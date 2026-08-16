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
  // oomph-lib does not ship a 1-point Gauss-Legendre rule; add the explicit specialization
  // (single knot at 0, weight 2, i.e. the midpoint rule on [-1,1]) since the periodic-orbit
  // collocation code below needs order-1 Gauss-Legendre integration as a special case.
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

  // Degenerate "integration rule" with a single, unweighted point, used as a placeholder
  // oomph::Integral when the periodic-orbit collocation mode does not actually need
  // quadrature (e.g. for element types that only require a single evaluation point).
  class POCollocationFakeIntegral: public oomph::Integral
  {
    public:
      unsigned nweight() const override {return 1;}
      double knot(const unsigned&, const unsigned&) const override {return -1;}
      double weight(const unsigned&) const override { return 1;}
  };
}

namespace pyoomph
{

  // Puts the rhs (and the vector that will hold the solution) of an upcoming resolve() on the
  // distribution the linear solver factorised on. Under mpirun oomph-lib assembles and factorises
  // on a uniform DISTRIBUTED layout even when the problem itself is not distributed, while the
  // parameter derivative these constructors resolve against comes back replicated. The solver would
  // redistribute the rhs itself -- but only after printing SuperLUSolver::resolve()'s "distribution
  // of rhs vector does not match that of the solver" warning once per rank, and it would leave the
  // result vector's distribution disagreeing with the rhs's, which PARANOID rejects outright.
  static void move_onto_solver_distribution(oomph::LinearSolver *const &linear_solver_pt,
                                            oomph::DoubleVector &rhs, oomph::DoubleVector &result)
  {
    oomph::LinearAlgebraDistribution *const solver_dist_pt = linear_solver_pt->distribution_pt();
    if (!solver_dist_pt || !solver_dist_pt->built()) return;
    if (*rhs.distribution_pt() == *solver_dist_pt) return;
    rhs.redistribute(solver_dist_pt);
    result.build(solver_dist_pt, 0.0);
  }

  // Rotates the complex eigenvector real_eigen+i*imag_eigen (in place) by a phase exp(-i*phi)
  // to a canonical orientation: phi is chosen (by sampling n_phi_samples initial guesses and
  // refining each with a few Newton iterations) so that <Re,Im>=0 and <Re,Re> is maximized,
  // i.e. all of the eigenvector's "length" is rotated into the real part. This removes the
  // arbitrary overall complex phase that a Hopf/azimuthal eigenvector is only defined up to,
  // giving reproducible output. Finally the (rotated) eigenvector is normalized so that
  // <Re,Re>=1.
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

  //////////////////////////////////////////////////////////
  // AugmentedDofDistributionHelper -- see bifurcation.hpp for the design rationale.
  //////////////////////////////////////////////////////////

  void AugmentedDofDistributionHelper::initialise(Problem *problem)
  {
    Problem_pt = problem;
#ifdef OOMPH_HAS_MPI
    Distributed = problem->distributed();
    if (Distributed)
    {
      // Keep the problem's original distribution OBJECT alive: the halo scheme built below stores
      // a raw pointer to it, and global_dof_pt() resolves base equations against it while the
      // problem itself carries the augmented distribution.
      Base_distribution_pt = problem->GetDofDistributionPt();
      problem->RebuildDofHaloScheme();
    }
    else
#endif
    {
      Base_distribution_copy.build(*problem->GetDofDistributionPt());
    }
  }

  oomph::LinearAlgebraDistribution *AugmentedDofDistributionHelper::base_dist_pt()
  {
#ifdef OOMPH_HAS_MPI
    if (Distributed) return Base_distribution_pt;
#endif
    return &Base_distribution_copy;
  }

  unsigned AugmentedDofDistributionHelper::base_nrow() const
  {
#ifdef OOMPH_HAS_MPI
    if (Distributed) return Base_distribution_pt->nrow();
#endif
    return Base_distribution_copy.nrow();
  }

  unsigned AugmentedDofDistributionHelper::base_nrow_local() const
  {
#ifdef OOMPH_HAS_MPI
    if (Distributed) return Base_distribution_pt->nrow_local();
#endif
    return Base_distribution_copy.nrow();
  }

  unsigned AugmentedDofDistributionHelper::base_first_row() const
  {
#ifdef OOMPH_HAS_MPI
    if (Distributed) return Base_distribution_pt->first_row();
#endif
    return 0;
  }

  void AugmentedDofDistributionHelper::build_base_vector(oomph::DoubleVectorWithHaloEntries &v) const
  {
#ifdef OOMPH_HAS_MPI
    if (Distributed)
    {
      v.build(Base_distribution_pt, 0.0);
      v.build_halo_scheme(Problem_pt->GetHaloSchemePt());
      return;
    }
#endif
    v.build(&Base_distribution_copy, 0.0);
  }

  void AugmentedDofDistributionHelper::setup_count_and_nelement(oomph::DoubleVectorWithHaloEntries &count, unsigned &nelement) const
  {
    unsigned n_element = Problem_pt->mesh_pt()->nelement();
    unsigned n_non_halo_element_local = 0;
    for (unsigned e = 0; e < n_element; e++)
    {
      GeneralisedElement *elem_pt = Problem_pt->mesh_pt()->element_pt(e);
#ifdef OOMPH_HAS_MPI
      // Halo elements are not assembled, so they must not enter the per-equation weights either --
      // otherwise the 1/Count factors of the normalization constraint no longer telescope to 1.
      if (elem_pt->is_halo()) continue;
#endif
      ++n_non_halo_element_local;
      unsigned n_var = elem_pt->ndof();
      for (unsigned n = 0; n < n_var; n++)
      {
        count.global_value(elem_pt->eqn_number(n)) += 1.0;
      }
    }
#ifdef OOMPH_HAS_MPI
    if (Distributed)
    {
      // An equation on a rank boundary is touched by elements of several ranks: make Count the
      // global element count per equation, and Nelement the global non-halo element count (the
      // -1/Nelement constants of the normalization residual must sum to -1 over ALL ranks).
      count.sum_all_halo_and_haloed_values();
      MPI_Allreduce(&n_non_halo_element_local, &nelement, 1, MPI_UNSIGNED, MPI_SUM,
                    Problem_pt->communicator_pt()->mpi_comm());
      return;
    }
#endif
    nelement = n_non_halo_element_local;
  }

  void AugmentedDofDistributionHelper::build_augmented_dofs(const std::vector<Block> &naive_layout)
  {
    const unsigned Ndof = base_nrow();
#ifdef OOMPH_HAS_MPI
    if (Distributed)
    {
      const unsigned my_rank = Problem_pt->communicator_pt()->my_rank();
      const unsigned nproc = Problem_pt->communicator_pt()->nproc();

      // Naive start offset of each block in the historical [base | k*Ndof+m] numbering
      std::vector<unsigned long> naive_start(naive_layout.size() + 1);
      naive_start[0] = Ndof; // the base dofs come first in every layout
      for (unsigned b = 0; b < naive_layout.size(); b++)
        naive_start[b + 1] = naive_start[b] + (naive_layout[b].vec ? Ndof : 1);
      const unsigned long total = naive_start[naive_layout.size()];

      // Per-rank interleaved translation table, modeled on upstream PitchForkHandler
      // (assembly_handler.cc): rank d's augmented rows are its base rows, then -- in naive block
      // order -- its rows of each eigenvector block, with each scalar as one row on rank 0.
      Global_eqn_number.assign(total, 0);
      unsigned long global_eqn_count = 0;
      unsigned long augmented_first_row = 0, augmented_n_row_local = 0;
      for (unsigned d = 0; d < nproc; d++)
      {
        if (my_rank == d) augmented_first_row = global_eqn_count;
        const unsigned n_row_local = Base_distribution_pt->nrow_local(d);
        const unsigned first_row = Base_distribution_pt->first_row(d);
        for (unsigned n = 0; n < n_row_local; n++)
        {
          Global_eqn_number[first_row + n] = global_eqn_count++;
        }
        for (unsigned b = 0; b < naive_layout.size(); b++)
        {
          if (naive_layout[b].vec)
          {
            for (unsigned n = 0; n < n_row_local; n++)
            {
              Global_eqn_number[naive_start[b] + first_row + n] = global_eqn_count++;
            }
          }
          else if (d == 0)
          {
            Global_eqn_number[naive_start[b]] = global_eqn_count++;
          }
        }
        if (my_rank == d) augmented_n_row_local = global_eqn_count - augmented_first_row;
      }

      // Push the locally owned augmented dofs in exactly the local row order of the table above
      // (the Newton update writes dx[l] onto Dof_pt[l] for l < nrow_local, so any mismatch here
      // silently applies increments to the wrong unknowns).
      const unsigned n_row_local = Base_distribution_pt->nrow_local();
      for (unsigned b = 0; b < naive_layout.size(); b++)
      {
        if (naive_layout[b].vec)
        {
          for (unsigned n = 0; n < n_row_local; n++)
          {
            Problem_pt->GetDofPtr().push_back(&(*naive_layout[b].vec)[n]);
          }
        }
        else if (my_rank == 0)
        {
          Problem_pt->GetDofPtr().push_back(naive_layout[b].scalar_pt);
        }
      }

      Augmented_distribution_pt = new oomph::LinearAlgebraDistribution(
          Problem_pt->communicator_pt(), (unsigned)augmented_first_row, (unsigned)augmented_n_row_local, (unsigned)total);
      Problem_pt->SetDofDistributionPt(Augmented_distribution_pt);
    }
    else
#endif
    {
      // Replicated: historical behavior -- every rank pushes everything, non-distributed in-place
      // build (see the comment in MyFoldHandler's constructor for why non-distributed is essential).
      unsigned long total = Ndof;
      for (unsigned b = 0; b < naive_layout.size(); b++)
      {
        if (naive_layout[b].vec)
        {
          for (unsigned n = 0; n < Ndof; n++)
          {
            Problem_pt->GetDofPtr().push_back(&(*naive_layout[b].vec)[n]);
          }
          total += Ndof;
        }
        else
        {
          Problem_pt->GetDofPtr().push_back(naive_layout[b].scalar_pt);
          total += 1;
        }
      }
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), (unsigned)total, false);
    }
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }

  double AugmentedDofDistributionHelper::allreduce_max(double local) const
  {
#ifdef OOMPH_HAS_MPI
    if (Distributed)
    {
      double global = local;
      MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, Problem_pt->communicator_pt()->mpi_comm());
      return global;
    }
#endif
    return local;
  }

  double AugmentedDofDistributionHelper::allreduce_sum(double local) const
  {
#ifdef OOMPH_HAS_MPI
    if (Distributed)
    {
      double global = local;
      MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, Problem_pt->communicator_pt()->mpi_comm());
      return global;
    }
#endif
    return local;
  }

  void AugmentedDofDistributionHelper::synchronise_scalars(std::initializer_list<double *> scalars) const
  {
#ifdef OOMPH_HAS_MPI
    if (!Distributed) return;
    std::vector<double> buf;
    buf.reserve(scalars.size());
    for (double *s : scalars) buf.push_back(*s);
    MPI_Bcast(buf.data(), (int)buf.size(), MPI_DOUBLE, 0, Problem_pt->communicator_pt()->mpi_comm());
    unsigned i = 0;
    for (double *s : scalars) *s = buf[i++];
#endif
  }

  void AugmentedDofDistributionHelper::restore_base_distribution()
  {
#ifdef OOMPH_HAS_MPI
    if (Distributed)
    {
      Problem_pt->GetDofPtr().resize(Base_distribution_pt->nrow_local());
      Problem_pt->SetDofDistributionPt(Base_distribution_pt);
      delete Augmented_distribution_pt;
      Augmented_distribution_pt = NULL;
      Global_eqn_number.clear();
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
      return;
    }
#endif
    Problem_pt->GetDofPtr().resize(Base_distribution_copy.nrow());
    Problem_pt->GetDofDistributionPt()->build(Base_distribution_copy);
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
  }


  //====================================================================
  /// Constructor: Initialise the hopf handler by solving for an initial guess of Phi
  /// (the real part of the null eigenvector) from J*Phi = dR/dparameter, deriving an
  /// orthogonal initial guess for Psi (the imaginary part) from it, setting Omega=0,
  /// and calculating Count. If the system changes, a new handler must be constructed.
  //===================================================================
  MyHopfHandler::MyHopfHandler(Problem *const &problem_pt,
                               double *const &parameter_pt) : Solve_which_system(0), Parameter_pt(parameter_pt), Omega(0.0)
  {
    // This constructor derives the eigenvector guess from a SERIAL linear solve on a replicated
    // vector -- not portable to a distributed problem, where a guess must be supplied explicitly.
    if (problem_pt->distributed())
    {
      throw_runtime_error("Hopf tracking on a distributed (--distribute) problem requires an explicit eigenvector guess -- solve an eigenproblem first or pass eigenvector=...");
    }
    call_param_change_handler = false;
    eigenweight = 1.0;
    // Set the problem pointer
    Problem_pt = problem_pt;
    // Set the number of non-augmented degrees of freedom
    Ndof = problem_pt->ndof();
    Dist_helper.initialise(problem_pt);

    // create the linear algebra distribution for this solver
    // currently only global (non-distributed) distributions are allowed
    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);

    // Resize the vectors of additional dofs
    Dist_helper.build_base_vector(Phi);
    Dist_helper.build_base_vector(Psi);
    C.resize(Ndof);
    Dist_helper.build_base_vector(Count);
    Dist_helper.setup_count_and_nelement(Count, Nelement);

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
    move_onto_solver_distribution(linear_solver_pt, input_x, x);
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

    // x comes back in whatever distribution the linear solver chose. The problem is not distributed
    // here, but on an mpirun oomph-lib still assembles the Jacobian over a UNIFORM DISTRIBUTED
    // layout (create_new_linear_algebra_distribution()), so x holds only nrow_local() doubles and
    // reading x[n] over the global dof range walks off the end of the buffer. That produced a NaN
    // guess and a Hopf Newton solve starting at inf on every mpirun while converging serially.
    std::vector<double> xg;
    Problem::gather_double_vector_to_global(x, xg);

    // Normalise the solution x
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += xg[n] * xg[n];
    }
    length = sqrt(length);

    // Now initialise the real part of the null space components (this constructor is
    // serial-by-construction, so Phi/Psi hold all Ndof rows here)
    // This is dumb at the moment ... fix with eigensolver?
    for (unsigned n = 0; n < Ndof; n++)
    {
      C[n] = Phi[n] = -xg[n] / length;
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

    // Add [Phi | Psi | param | Omega] to the problem unknowns and rebuild the augmented dof
    // distribution (non-distributed in place, as this constructor refuses --distribute above)
    Dist_helper.build_augmented_dofs({AugmentedDofDistributionHelper::Block::vector(&Phi),
                                      AugmentedDofDistributionHelper::Block::vector(&Psi),
                                      AugmentedDofDistributionHelper::Block::scalar(Parameter_pt),
                                      AugmentedDofDistributionHelper::Block::scalar(&Omega)});

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
    Dist_helper.initialise(problem_pt);

    // Resize the vectors of additional dofs
    Dist_helper.build_base_vector(Phi);
    Dist_helper.build_base_vector(Psi);
    C.resize(Ndof);
    Dist_helper.build_base_vector(Count);
    Dist_helper.setup_count_and_nelement(Count, Nelement);

    // Rotate/normalise the guess on REPLICATED copies (the guesses arrive full-length and
    // identical on every rank, and the rotation involves global dot products), then scatter the
    // owned rows into the -- possibly distributed -- unknowns.
    oomph::Vector<double> phi_rot(Ndof), psi_rot(Ndof);
    for (unsigned n = 0; n < Ndof; n++)
    {
      phi_rot[n] = phi[n];
      psi_rot[n] = psi[n];
    }
    rotate_complex_eigenvector_nicely(phi_rot, psi_rot);

    for (unsigned n = 0; n < Ndof; n++)
    {
      C[n] = phi_rot[n];
    }
    const unsigned n_row_local = Dist_helper.base_nrow_local();
    const unsigned first_row = Dist_helper.base_first_row();
    for (unsigned n = 0; n < n_row_local; n++)
    {
      Phi[n] = phi_rot[first_row + n];
      Psi[n] = psi_rot[first_row + n];
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      Phi.synchronise();
      Psi.synchronise();
    }
#endif

    // Add [Phi | Psi | param | Omega] and rebuild the augmented dof distribution (non-distributed
    // in place when replicated, pointer swap when distributed; see AugmentedDofDistributionHelper)
    Dist_helper.build_augmented_dofs({AugmentedDofDistributionHelper::Block::vector(&Phi),
                                      AugmentedDofDistributionHelper::Block::vector(&Psi),
                                      AugmentedDofDistributionHelper::Block::scalar(Parameter_pt),
                                      AugmentedDofDistributionHelper::Block::scalar(&Omega)});
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
    Dist_helper.restore_base_distribution();
  }

  void MyHopfHandler::set_eigenweight(double ew)
  {
    // Phi/Psi are the unknowns here: rescale the owned rows and refresh the halos
    const unsigned n_row_local = Dist_helper.distributed() ? Dist_helper.base_nrow_local() : Ndof;
    for (unsigned n = 0; n < n_row_local; n++)
    {
      Phi[n] *= ew / eigenweight;
      Psi[n] *= ew / eigenweight;
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      Phi.synchronise();
      Psi.synchronise();
    }
#endif
    eigenweight = ew;
  }

  //=============================================================
  /// Get the number of elemental degrees of freedom
  //=============================================================
  // The augmented block of a Hopf tracker, read off get_jacobian() below.
  //
  // Layout (Solve_which_system == 0, see eqn_number):
  //   [ base (raw) | Phi (raw) | Psi (raw) | parameter (1) | Omega (1) ]
  //
  // The augmented equations are
  //   R(u,p) = 0,   J.Phi - Omega M.Psi = 0,   J.Psi + Omega M.Phi = 0,   C.Phi - 1 = 0,   C.Psi = 0
  //
  //                  base        Phi         Psi         param    Omega
  //   base rows      J           -           -           dense    -
  //   Phi  rows      H           J           M           dense    dense
  //   Psi  rows      H           M           J           dense    dense
  //   param row      -           dense       -           -        -
  //   Omega row      -           -           dense       -        -
  //
  // The two base-column blocks of the eigenvector rows are Hessians: dJdU_Eig and dMdU_Eig, both of
  // which contributes_to_hessian covers -- it is marked whenever EITHER the Jacobian or the mass part
  // of the second derivative is non-zero, so a mass-matrix Hessian needs no separate kind.
  //
  // The (Phi,Phi) and (Psi,Psi) blocks are J, and pick up a further (*Parameter_pt)*M term in the
  // mass-augmented variant; that stays within the Jacobian pattern on the same grounds Phase 4 relies
  // on, that the mass matrix couples no field pair the Jacobian does not. If that ever fails, the
  // per-element verification says so rather than truncating.
  //
  // Neither scalar row has a diagonal: C.Phi - 1 involves neither the parameter nor Omega, so those
  // positions are Empty rather than Dense. Declaring them Dense would manufacture exactly the stored
  // zero diagonal that section 7c showed leads MUMPS onto a null pivot.
  bool MyHopfHandler::get_sparsity_pattern(GeneralisedElement *const &elem_pt, AugmentedBlockSpec &spec) const
  {
    if (Solve_which_system != 0) return false; // Only the full augmented system needs describing
    typedef AugmentedBlockSpec S;
    spec.resize(5);
    spec.group_is_scalar[0] = false; // u
    spec.group_is_scalar[1] = false; // Phi
    spec.group_is_scalar[2] = false; // Psi
    spec.group_is_scalar[3] = true;  // the bifurcation parameter
    spec.group_is_scalar[4] = true;  // Omega
    spec.set(0, 0, S::Jacobian);
    spec.set(0, 3, S::Dense);
    spec.set(1, 0, S::Hessian);
    spec.set(1, 1, S::Jacobian);
    spec.set(1, 2, S::MassMatrix);
    spec.set(1, 3, S::Dense);
    spec.set(1, 4, S::Dense);
    spec.set(2, 0, S::Hessian);
    spec.set(2, 1, S::MassMatrix);
    spec.set(2, 2, S::Jacobian);
    spec.set(2, 3, S::Dense);
    spec.set(2, 4, S::Dense);
    spec.set(3, 1, S::Dense);
    spec.set(4, 2, S::Dense);
    return true;
  }

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
    // Naive numbering -> per-rank interleaved augmented numbering when distributed (identity otherwise)
    return Dist_helper.global_eqn(global_eqn);
  }

  //==================================================================
  /// Get the residuals
  //=================================================================
  // Layout of the augmented per-element residual vector (raw_ndof = elem_pt->ndof()):
  //   [0, raw_ndof)              base residuals R(u)
  //   [raw_ndof, 2*raw_ndof)     real part of J*Phi - Omega*M*Psi   (eigen-equation, real)
  //   [2*raw_ndof, 3*raw_ndof)   real part of J*Psi + Omega*M*Phi   (eigen-equation, imag)
  //   3*raw_ndof                 normalization  sum(Phi.C)/nelement_sharing - 1/nelement (accumulated additively across elements)
  //   3*raw_ndof+1               normalization  sum(Psi.C)/nelement_sharing
  // If Parameter_pt is the special lambda-tracking parameter (used to directly track a
  // complex eigenvalue branch rather than a physical control parameter), an extra
  // (*Parameter_pt)*M*Phi / (*Parameter_pt)*M*Psi term is added to the eigen-equations,
  // since in that mode the parameter itself acts as an eigenvalue shift in J - lambda*M.
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
      residuals[3 * raw_ndof] = -Normalization_rhs / (double)Nelement * eigenweight;
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
              jacobian(i, j) * Phi.global_value(global_unknown) - Omega * M(i, j) * Psi.global_value(global_unknown);
          // Imaginary part
          residuals[2 * raw_ndof + i] +=
              jacobian(i, j) * Psi.global_value(global_unknown) + Omega * M(i, j) * Phi.global_value(global_unknown);
        }
        // Get the global equation number
        unsigned global_eqn = elem_pt->eqn_number(i);

        // Real part
        residuals[3 * raw_ndof] += (Phi.global_value(global_eqn) * C[global_eqn]) /
                                   Count.global_value(global_eqn);
        // Imaginary part
        residuals[3 * raw_ndof + 1] += (Psi.global_value(global_eqn) * C[global_eqn]) /
                                       Count.global_value(global_eqn);
      }

      if (lambda_tracking)
      {
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            unsigned global_unknown = elem_pt->eqn_number(j);
            residuals[raw_ndof + i] += (*Parameter_pt) * M(i, j) * Phi.global_value(global_unknown);
            residuals[2 * raw_ndof + i] +=  (*Parameter_pt) * M(i, j) * Psi.global_value(global_unknown);
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

  // Debugging aid: recomputes the augmented residuals/Jacobian for elem_pt once with the
  // analytic-Hessian path disabled (forcing finite-difference filling of the eigen-equation
  // derivatives) and once with it enabled, then prints every entry that disagrees by more
  // than eps between the two. Requires analytic Hessian products to be available/enabled.
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
  // The (u,u), (Phi,Phi)/(Phi,Psi) and (Psi,Phi)/(Psi,Psi) blocks are filled directly from
  // the base Jacobian J and mass matrix M (since d(J*Phi)/dPhi = J etc.). The blocks
  // involving derivatives of J and M themselves with respect to u (needed because J,M depend
  // on u) require the Hessian of the base residuals: if analytic Hessian-vector products are
  // available (ana_hessian) they are used directly, otherwise this falls back to finite
  // differences (see the loop over raw_ndof further below, perturbing each u-dof by FD_step
  // and re-evaluating get_residuals()). The parameter- and Omega-columns are filled
  // analogously, analytically if possible, by finite differences otherwise.
  void MyHopfHandler::get_jacobian(GeneralisedElement *const &elem_pt,
                                   Vector<double> &residuals,
                                   DenseMatrix<double> &jacobian)
  {

    bool lambda_tracking=(Parameter_pt==Problem_pt->get_lambda_tracking_real());
    bool ana_dparam = lambda_tracking || Problem_pt->is_dparameter_calculated_analytically(Parameter_pt);
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
            jacobian(raw_ndof + n, 3 * raw_ndof + 1) -= M(n, m) * Psi.global_value(global_eqn);
            jacobian(2 * raw_ndof + n, 3 * raw_ndof + 1) += M(n, m) * Phi.global_value(global_eqn);
          }

          unsigned local_eqn = elem_pt->eqn_number(n);
          jacobian(3 * raw_ndof, raw_ndof + n) = C[local_eqn] / Count.global_value(local_eqn);
          jacobian(3 * raw_ndof + 1, 2 * raw_ndof + n) = C[local_eqn] / Count.global_value(local_eqn);
        }

        const double FD_step = this->FD_step;

        Vector<double> newres_p(augmented_ndof), newres_m(augmented_ndof);

        //	 DenseMatrix<double> dJduPhi(raw_ndof,raw_ndof,0.0);
        //	 DenseMatrix<double> dJduPsi(raw_ndof,raw_ndof,0.0);

        // Loop over the dofs
        for (unsigned n = 0; n < raw_ndof; n++)
        {
          // Just do the x's -- perturb via the element's own BASE equation number (halo-aware
          // when distributed), not the handler's translated augmented number
          unsigned long base_eqn = elem_pt->eqn_number(n);
          double *unknown_pt = Problem_pt->global_dof_pt(base_eqn);
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
          this->get_dresiduals_dparameter(elem_pt, Parameter_pt, dres_dparam);
          for (unsigned m = 0; m < augmented_ndof - 2; m++)
          {
            jacobian(m, 3 * raw_ndof) = dres_dparam[m];
          }
        }
        else
        {
          // Now do the global parameter
          double *unknown_pt = Parameter_pt;
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
        // Analytic path: instead of finite-differencing get_residuals() once per perturbed
        // dof, request the Hessian-vector products dJdU_Eig=(d J/du)*Eig, dMdU_Eig=(d M/du)*Eig
        // (contracted directly with the eigenvector Eig_local=(Phi,Psi)) and the parameter
        // derivatives dJdParam/dMdParam/dRdParam from the generated element code in a single
        // combined assembly (multi_assm), then assemble all augmented Jacobian blocks from
        // these precomputed second-derivative contractions instead of finite differences.
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

          Eig_local[i] = Phi.global_value(global_eqn);
          Eig_local[raw_ndof + i] = Psi.global_value(global_eqn);
        }
        multi_assm.push_back(SinglePassMultiAssembleInfo(pyoomph_elem_pt->get_code_instance()->get_func_table()->current_res_jac, &residuals, &jacobian, &M));

        multi_assm.back().add_hessian(Eig_local, &dJdU_Eig, &dMdU_Eig);
        if (!lambda_tracking) multi_assm.back().add_param_deriv(Parameter_pt, &dRdParam, &dJdParam, &dMdParam);
        pyoomph_elem_pt->get_multi_assembly(multi_assm);

        // Residuals
        residuals[3 * raw_ndof] = -Normalization_rhs / (double)Nelement * eigenweight;
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
          residuals[3 * raw_ndof] += (Phi.global_value(global_eqn) * C[global_eqn]) / Count.global_value(global_eqn);
          residuals[3 * raw_ndof + 1] += (Psi.global_value(global_eqn) * C[global_eqn]) / Count.global_value(global_eqn);
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
          jacobian(3 * raw_ndof, raw_ndof + n) = C[local_eqn] / Count.global_value(local_eqn);         // dR[Param]/dPhi
          jacobian(3 * raw_ndof + 1, 2 * raw_ndof + n) = C[local_eqn] / Count.global_value(local_eqn); // dR[Omega]/dPsi
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
    // Otherwise the augmented complex case: solves only for the complex eigenvector
    // (Phi,Psi) block at fixed u and Omega, i.e. the 2*raw_ndof x 2*raw_ndof system
    // [[J,Omega*M],[-Omega*M,J]] * (Phi,Psi) = rhs, with the right-hand side chosen as
    // the mass-matrix-only terms below (residuals[n]=M*Psi, residuals[raw_ndof+n]=-M*Phi)
    // since those need no extra assembly beyond the mass matrix already computed here.
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
          residuals[n] += M(n, m) * Psi.global_value(global_unknown);
          // Imaginary part
          residuals[raw_ndof + n] -= M(n, m) * Phi.global_value(global_unknown);
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
              djac_dparam(i, j) * Phi.global_value(global_unknown) +
              Omega * dM_dparam(i, j) * Psi.global_value(global_unknown);
          // Imaginary part
          dres_dparam[2 * raw_ndof + i] +=
              djac_dparam(i, j) * Psi.global_value(global_unknown) -
              Omega * dM_dparam(i, j) * Phi.global_value(global_unknown);
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
      GeneralisedElement *const &,
      double *const &,
      Vector<double> &,
      DenseMatrix<double> &)
  {
    std::ostringstream error_stream;
    error_stream << "This function has not been implemented because it is not required\n";
    error_stream << "in standard problems.\n";
    error_stream << "If you find that you need it, you will have to implement it!\n\n";

    throw OomphLibError(error_stream.str(),
                        OOMPH_CURRENT_FUNCTION,
                        OOMPH_EXCEPTION_LOCATION);
  }

  // Simply forwards to the element's own Hessian-vector-product routine; the augmented
  // eigen-equations do not themselves contribute extra Hessian terms here.
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
    // There is a real and imaginary part of the null vector. The contract is a globally
    // replicated, non-distributed vector on every rank, so gather when distributed.
    eigenfunction.resize(2);
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false);
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      eigenfunction[0] = Phi;
      eigenfunction[1] = Psi;
      eigenfunction[0].redistribute(&dist);
      eigenfunction[1].redistribute(&dist);
      return;
    }
#endif
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
    // Gather the (possibly distributed) eigenvector into replicated copies first; the rotation
    // involves global dot products and its result must be identical on every rank.
    std::vector<std::complex<double>> eigenfunction(Ndof);
    oomph::Vector<double> PhiRot(Ndof), PsiRot(Ndof);
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      oomph::DoubleVector tmp_phi = Phi, tmp_psi = Psi;
      std::vector<double> g_phi, g_psi;
      Problem::gather_double_vector_to_global(tmp_phi, g_phi);
      Problem::gather_double_vector_to_global(tmp_psi, g_psi);
      for (unsigned n = 0; n < Ndof; n++)
      {
        PhiRot[n] = g_phi[n];
        PsiRot[n] = g_psi[n];
      }
    }
    else
#endif
    {
      for (unsigned n = 0; n < Ndof; n++)
      {
        PhiRot[n] = Phi[n];
        PsiRot[n] = Psi[n];
      }
    }
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
    // The block modes rebuild replicated in-place dof vectors -- refused when distributed
    // (only reachable through the BlockHopfLinearSolver, which is refused there anyway)
    if (Dist_helper.distributed()) throw_runtime_error("Hopf tracking block-solve modes are not supported on a distributed (--distribute) problem");
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
    if (Dist_helper.distributed()) throw_runtime_error("Hopf tracking block-solve modes are not supported on a distributed (--distribute) problem");
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
    if (Dist_helper.distributed() && Solve_which_system) throw_runtime_error("Hopf tracking block-solve modes are not supported on a distributed (--distribute) problem");
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

  // Re-normalizes the eigenvector (Phi,Psi) to unit weight (scaled by eigenweight) and resets
  // the normalization vector C to the (freshly rescaled) Phi, so that the normalization
  // equations C.Phi=1, C.Psi=0 stay well-conditioned as the eigenvector evolves during
  // continuation. Also prints diagnostic dot products of C against the current Phi/Psi.
  void MyHopfHandler::realign_C_vector()
  {
    // The dof entries Ndof+n / 2*Ndof+n point INTO Phi/Psi, so read the members directly (owned
    // rows plus a reduction when distributed) instead of the replicated dof-pointer table.
    double dot = 0.0;
    double doti = 0.0;
    double phisqr = 0.0;
    double psisqr = 0.0;
    const unsigned n_row_local = Dist_helper.distributed() ? Dist_helper.base_nrow_local() : Ndof;
    const unsigned first_row = Dist_helper.distributed() ? Dist_helper.base_first_row() : 0;
    for (unsigned n = 0; n < n_row_local; n++)
    {
      double phin = Phi[n];
      double psin = Psi[n];
      dot += C[first_row + n] * phin;
      phisqr += phin * phin;
      doti += C[first_row + n] * psin;
      psisqr += psin * psin;
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      double local_vals[4] = {dot, doti, phisqr, psisqr}, global_vals[4];
      MPI_Allreduce(local_vals, global_vals, 4, MPI_DOUBLE, MPI_SUM, Problem_pt->communicator_pt()->mpi_comm());
      dot = global_vals[0];
      doti = global_vals[1];
      phisqr = global_vals[2];
      psisqr = global_vals[3];
    }
#endif
    std::cerr << "DOT OF C and PHi is " << dot << " and PHi^2 = " << phisqr << std::endl;
    std::cerr << "DOT OF C and Psi is " << doti << " and Psi^2 = " << psisqr << std::endl;

    double lf = eigenweight / sqrt(phisqr);
    for (unsigned n = 0; n < n_row_local; n++)
    {
      Phi[n] *= lf;
      Psi[n] *= lf;
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      Phi.synchronise();
      Psi.synchronise();
      // C stays replicated: rebuild it from the gathered, freshly rescaled Phi
      oomph::DoubleVector tmp_phi = Phi;
      std::vector<double> g_phi;
      Problem::gather_double_vector_to_global(tmp_phi, g_phi);
      for (unsigned n = 0; n < Ndof; n++) C[n] = g_phi[n];
      return;
    }
#endif
    for (unsigned n = 0; n < Ndof; n++)
    {
      C[n] = Phi[n];
    }
  }

  // Opt-in rescaling of the normalisation constraint; see the comment on the declaration in
  // bifurcation.hpp for why. The max is taken over the real and imaginary parts JOINTLY, so the
  // complex eigenvector is scaled as one object and its phase is untouched.
  void MyHopfHandler::apply_maxabs_normalization()
  {
    const unsigned n_row_local = Dist_helper.distributed() ? Dist_helper.base_nrow_local() : Ndof;
    const unsigned first_row = Dist_helper.distributed() ? Dist_helper.base_first_row() : 0;
    double local_max = 0.0;
    for (unsigned n = 0; n < n_row_local; n++)
    {
      local_max = std::max(local_max, std::fabs(Phi[n]));
      local_max = std::max(local_max, std::fabs(Psi[n]));
    }
    const double m = Dist_helper.allreduce_max(local_max);
    if (m <= 0.0) return;
    for (unsigned n = 0; n < n_row_local; n++)
    {
      Phi[n] /= m;
      Psi[n] /= m;
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      Phi.synchronise();
      Psi.synchronise();
    }
#endif
    for (unsigned n = 0; n < Ndof; n++) C[n] /= m;
    double local_dot = 0.0;
    for (unsigned n = 0; n < n_row_local; n++) local_dot += C[first_row + n] * Phi[n];
    Normalization_rhs = Dist_helper.allreduce_sum(local_dot) / eigenweight;
  }

#ifdef OOMPH_HAS_MPI
  // After a Newton update only the owned rows of Phi/Psi (and, on rank 0, the parameter and
  // Omega) are current: refresh the halo copies and broadcast the scalars. Called from
  // Problem::synchronise_all_dofs at the end of each update.
  void MyHopfHandler::synchronise()
  {
    if (!Dist_helper.distributed()) return;
    Phi.synchronise();
    Psi.synchronise();
    Dist_helper.synchronise_scalars({Parameter_pt, &Omega});
  }
#endif

  /////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Constructs the fold handler and derives an initial guess for the null eigenvector Y by
  // solving J*x = dR/dparameter (i.e. reusing the already-factorized Jacobian) and normalizing
  // x; both Y and the fixed normalization vector Phi are initialized to this same direction.
  MyFoldHandler::MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt) : Solve_which_system(Full_augmented), Parameter_pt(parameter_pt)
  {
    // This constructor derives the eigenvector guess from a SERIAL linear solve on a replicated
    // vector -- not portable to a distributed problem, where a guess must be supplied explicitly.
    if (problem_pt->distributed())
    {
      throw_runtime_error("Fold tracking on a distributed (--distribute) problem requires an explicit eigenvector guess -- solve an eigenproblem first or pass eigenvector=...");
    }
    call_param_change_handler = false;
    eigenweight = 1.0;
    FD_step = 1e-8;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();
    Dist_helper.initialise(problem_pt);

    LinearAlgebraDistribution *dist_pt = new LinearAlgebraDistribution(problem_pt->communicator_pt(), Ndof, false);

    Phi.resize(Ndof);
    Dist_helper.build_base_vector(Y);
    Dist_helper.build_base_vector(Count);
    Dist_helper.setup_count_and_nelement(Count, Nelement);

    LinearSolver *const linear_solver_pt = problem_pt->linear_solver_pt();

    bool enable_resolve = linear_solver_pt->is_resolve_enabled();
    linear_solver_pt->enable_resolve();
    DoubleVector x(dist_pt, 0.0);
    linear_solver_pt->solve(problem_pt, x);
    problem_pt->get_derivative_wrt_global_parameter(parameter_pt, x);
    DoubleVector input_x(x);
    move_onto_solver_distribution(linear_solver_pt, input_x, x);
    linear_solver_pt->resolve(input_x, x);
    if (enable_resolve)
    {
      linear_solver_pt->enable_resolve();
    }
    else
    {
      linear_solver_pt->disable_resolve();
    }
    // Same trap as in MyHopfHandler's no-guess constructor: on an mpirun of a NON-distributed
    // problem the solver still works on a uniform distributed layout, so x[n] over the global dof
    // range reads past the local buffer. The fold guess came out NaN and the augmented Newton solve
    // reported "Initial Maximum residuals inf" straight away (kuramoto_sivanshinsky_bifurcation.py).
    std::vector<double> xg;
    Problem::gather_double_vector_to_global(x, xg);

    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += xg[n] * xg[n];
    }
    length = sqrt(length);

    for (unsigned n = 0; n < Ndof; n++)
    {
      Y[n] = Phi[n] = -xg[n] / length;
    }
    // The augmented dof vector must be built NON-distributed, like the base problem's own
    // (assign_eqn_numbers() builds it with false whenever the problem is not distributed; the
    // distributed case takes the pointer-swap path inside the helper instead). With true, every
    // rank of an mpirun took Dof_distribution_pt->nrow_local() as the length of its Newton update
    // and wrote it into GetDofPtr()[0...], so rank 1 applied the eigenvector block's increment to
    // the base dofs and nobody updated the parameter: the fold Newton solve diverged to inf under
    // MPI while converging serially. Inherited from upstream oomph-lib, whose handlers do the same.
    Dist_helper.build_augmented_dofs({AugmentedDofDistributionHelper::Block::scalar(Parameter_pt),
                                      AugmentedDofDistributionHelper::Block::vector(&Y)});
    delete dist_pt;
  }

  // Constructs the fold handler with an explicit initial guess for the eigenvector; both Y and
  // the normalization vector Phi are set to the (normalized) supplied eigenvector.
  MyFoldHandler::MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt, const DoubleVector &eigenvector) : Solve_which_system(Full_augmented), Parameter_pt(parameter_pt)
  {
    call_param_change_handler = false;
    eigenweight = 1.0;
    FD_step = 1e-8;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();
    Dist_helper.initialise(problem_pt);
    Phi.resize(Ndof);
    Dist_helper.build_base_vector(Y);
    Dist_helper.build_base_vector(Count);
    Dist_helper.setup_count_and_nelement(Count, Nelement);
    // The guess arrives replicated (full length, identical on every rank; see
    // start_bifurcation_tracking), so global sums and Phi need no communication.
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += eigenvector[n] * eigenvector[n];
    }
    length = sqrt(length);
    for (unsigned n = 0; n < Ndof; n++)
    {
      Phi[n] = eigenvector[n] / length;
    }
    const unsigned n_row_local = Dist_helper.base_nrow_local();
    const unsigned first_row = Dist_helper.base_first_row();
    for (unsigned n = 0; n < n_row_local; n++)
    {
      Y[n] = eigenvector[first_row + n] / length;
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed()) Y.synchronise();
#endif
    // Non-distributed in-place build when replicated / pointer swap to a genuinely distributed
    // augmented distribution otherwise; see the comment in the first constructor above.
    Dist_helper.build_augmented_dofs({AugmentedDofDistributionHelper::Block::scalar(Parameter_pt),
                                      AugmentedDofDistributionHelper::Block::vector(&Y)});
  }

  // Constructs the fold handler with an explicit eigenvector guess and an independently chosen
  // normalization vector Phi (rather than reusing the eigenvector itself for normalization),
  // e.g. to keep the normalization equation well-conditioned across a continuation run where
  // the eigenvector direction itself may vary a lot.
  MyFoldHandler::MyFoldHandler(Problem *const &problem_pt, double *const &parameter_pt, const DoubleVector &eigenvector, const DoubleVector &normalisation) : Solve_which_system(Full_augmented), Parameter_pt(parameter_pt)
  {
    call_param_change_handler = false;
    eigenweight = 1.0;
    FD_step = 1e-8;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();
    Dist_helper.initialise(problem_pt);
    Phi.resize(Ndof);
    Dist_helper.build_base_vector(Y);
    Dist_helper.build_base_vector(Count);
    Dist_helper.setup_count_and_nelement(Count, Nelement);
    // Both inputs arrive replicated (see the second constructor above).
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += eigenvector[n] * normalisation[n];
    }
    length = sqrt(length);
    for (unsigned n = 0; n < Ndof; n++)
    {
      Phi[n] = normalisation[n];
    }
    const unsigned n_row_local = Dist_helper.base_nrow_local();
    const unsigned first_row = Dist_helper.base_first_row();
    for (unsigned n = 0; n < n_row_local; n++)
    {
      Y[n] = eigenvector[first_row + n] / length;
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed()) Y.synchronise();
#endif
    // Non-distributed in-place build when replicated / pointer swap when distributed; see the
    // comment in the first constructor above.
    Dist_helper.build_augmented_dofs({AugmentedDofDistributionHelper::Block::scalar(Parameter_pt),
                                      AugmentedDofDistributionHelper::Block::vector(&Y)});
  }

  // Number of augmented dofs of the element, depending on Solve_which_system: the full
  // augmented system carries the original dofs plus the eigenvector Y plus the parameter
  // (2*raw_ndof+1); the "augmented J" block carries the original dofs plus the parameter only
  // (raw_ndof+1); the plain "J" block carries only the original dofs.
  // The augmented block of a fold tracker, read off get_jacobian() below.
  //
  // Layout (Full_augmented, see eqn_number): [ base (raw) | parameter (1) | eigenvector (raw) ].
  //
  //                  base cols        param col      eig cols
  //   base rows      J                 dense          -           get_jacobian(residuals, jacobian)
  //   param row      -                 -              dense       jacobian(raw, raw+1+n) = Phi[..]
  //   eig  rows      H (dJdU . Y)      dense          J           jacobian(raw+1+m, n) = dJduPhiH(m,n)
  //                                                               jacobian(raw+1+n, raw+1+m) = jacobian(n,m)
  //
  // The eigenvector-row/base-column block is the Hessian contracted with Y. It is NOT transposed: the
  // fold condition J.Y = 0 uses the RIGHT null vector, so d(J.Y)_m/du_n keeps the (m,n) orientation.
  // It gets the Hessian pattern rather than the Jacobian's, which is a subset relation but a loose one
  // -- linear terms of the residual are all of J and none of the Hessian.
  //
  // The parameter row's diagonal is deliberately Empty. The normalisation equation Phi.Y = 1 does not
  // involve the parameter, so nothing is written there today either -- and declaring it Dense would
  // manufacture a stored zero on the diagonal, which section 7c showed is exactly what invites MUMPS
  // to plan an elimination onto a null pivot.
  bool MyFoldHandler::get_sparsity_pattern(GeneralisedElement *const &elem_pt, AugmentedBlockSpec &spec) const
  {
    if (Solve_which_system != Full_augmented) return false; // Block_J needs no spec (it IS the raw block)
    typedef AugmentedBlockSpec S;
    spec.resize(3);
    spec.group_is_scalar[0] = false; // base dofs
    spec.group_is_scalar[1] = true;  // the bifurcation parameter
    spec.group_is_scalar[2] = false; // the eigenvector
    spec.set(0, 0, S::Jacobian);
    spec.set(0, 1, S::Dense);
    spec.set(2, 0, S::Hessian); // d(J.Y)/du: second derivatives only, hence far sparser than J itself
    spec.set(2, 1, S::Dense);
    spec.set(2, 2, S::Jacobian);
    spec.set(1, 2, S::Dense);
    return true;
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

  // Maps a local dof index to the global equation number in the augmented numbering:
  // [0,Ndof) original dofs, Ndof the bifurcation parameter, [Ndof+1,2*Ndof+1) the eigenvector Y.
  // Under --distribute this naive numbering is translated to the per-rank interleaved layout by
  // the helper (identity otherwise).
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
    return Dist_helper.global_eqn(global_eqn);
  }

  // Assembles the residuals for whichever block Solve_which_system selects. For Full_augmented,
  // in addition to the base residuals (or, for the lambda-tracking parameter, the base residuals
  // plus mass matrix), the eigen-equation residual J*Y and the normalization residual Phi.Y-1
  // are appended (filled further below in this function).
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
      residuals[raw_ndof] = -Normalization_rhs / Nelement * eigenweight;
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        residuals[raw_ndof + 1 + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y.global_value(elem_pt->eqn_number(j));
        }
        unsigned global_eqn = elem_pt->eqn_number(i);
        residuals[raw_ndof] += (Phi[global_eqn] * Y.global_value(global_eqn)) / Count.global_value(global_eqn);
      }

      if (Parameter_pt==Problem_pt->get_lambda_tracking_real())
      {
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            residuals[raw_ndof + 1 + i] += (*Parameter_pt)*mass_matrix(i, j) * Y.global_value(elem_pt->eqn_number(j));
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

  // Assembles the Jacobian for whichever block Solve_which_system selects (see ndof()/
  // get_residuals() above for what each block contains). For Block_augmented_J, the extra
  // parameter column is filled by finite differences and the extra normalization row directly
  // from Phi. For Full_augmented, the dY/dY block reuses the base Jacobian, while the blocks
  // coupling Y to u (i.e. d(J*Y)/du) and to the parameter require second derivatives of the
  // base residuals: these are obtained analytically via Hessian-vector products when available
  // (ana_hessian), otherwise by finite-differencing get_residuals() (see MyHopfHandler::get_jacobian
  // for the analogous, more heavily commented pattern).
  void MyFoldHandler::get_jacobian(GeneralisedElement *const &elem_pt,
                                   Vector<double> &residuals,
                                   DenseMatrix<double> &jacobian)
  {
    // If true, we do not track a fold by adjusting the parameter, but track an eigenvalue branch
    bool lambda_continuation=  (Parameter_pt==Problem_pt->get_lambda_tracking_real());
    bool ana_dparam = lambda_continuation || Problem_pt->is_dparameter_calculated_analytically(Parameter_pt);
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
        double *unknown_pt = Parameter_pt;
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
        jacobian(augmented_ndof - 1, n) = Phi[local_eqn] / Count.global_value(local_eqn);
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
          Y_local[_e] = Y.global_value(elem_pt->eqn_number(_e));
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
        residuals[raw_ndof] = -Normalization_rhs / Nelement * eigenweight;
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          residuals[raw_ndof + 1 + i] = 0.0;
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y_local[j];
          }
          unsigned global_eqn = elem_pt->eqn_number(i);
          residuals[raw_ndof] += (Phi[global_eqn] * Y.global_value(global_eqn)) / Count.global_value(global_eqn);
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
          jacobian(raw_ndof, raw_ndof + 1 + n) = Phi[global_eqn] / Count.global_value(global_eqn);
        }

         if (lambda_continuation)
         {
          for (unsigned n = 0; n < raw_ndof; n++)
          {
            for (unsigned m = 0; m < raw_ndof; m++)
            {
              residuals[raw_ndof + 1 + n] += (*Parameter_pt)*M(n, m) * Y_local[m];
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
          elem_pt->get_djacobian_dparameter(Parameter_pt, dres_dparam, djac_dparam);
          for (unsigned n = 0; n < raw_ndof; n++)
          {
            jacobian(n, raw_ndof) = dres_dparam[n];
            for (unsigned l = 0; l < raw_ndof; l++)
            {
              jacobian(raw_ndof + 1 + n, raw_ndof) += djac_dparam(n, l) * Y.global_value(elem_pt->eqn_number(l));
            }
          }
        }
        else
        {

          double FD_step = this->FD_step;
          {
            double *unknown_pt = Parameter_pt;
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
                  jacobian(raw_ndof + 1 + n, raw_ndof) += (newjac(n, l) - jacobian(n, l)) * Y.global_value(elem_pt->eqn_number(l)) /
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
                  jacobian(raw_ndof + 1 + n, raw_ndof) += (newjac(n, l) - newjac_m(n, l)) * Y.global_value(elem_pt->eqn_number(l)) /
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
          // The element's own (base) equation number, deliberately NOT the handler's translated
          // augmented one: we perturb a base dof, and global_dof_pt resolves base numbers halo-aware
          // when distributed (plain Dof_pt indexing otherwise, as before).
          unsigned long base_eqn = elem_pt->eqn_number(n);
          double *unknown_pt = Problem_pt->global_dof_pt(base_eqn);
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
                jacobian(raw_ndof + 1 + k, n) += (newjac(k, l) - jacobian(k, l)) * Y.global_value(elem_pt->eqn_number(l)) / FD_step;
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
                jacobian(raw_ndof + 1 + k, n) += (newjac(k, l) - newjac_m(k, l)) * Y.global_value(elem_pt->eqn_number(l)) / (2 * FD_step);
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
          jacobian(raw_ndof, raw_ndof + 1 + n) = Phi[global_eqn] / Count.global_value(global_eqn);
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
          dres_dparam[raw_ndof + 1 + i] += djac_dparam(i, j) * Y.global_value(elem_pt->eqn_number(j));
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

  void MyFoldHandler::get_djacobian_dparameter(GeneralisedElement *const &, double *const &, Vector<double> &, DenseMatrix<double> &)
  {
    std::ostringstream error_stream;
    error_stream << "This function has not been implemented because it is not required\n";
    error_stream << "in standard problems.\n";
    error_stream << "If you find that you need it, you will have to implement it!\n\n";

    throw OomphLibError(error_stream.str(),
                        OOMPH_CURRENT_FUNCTION,
                        OOMPH_EXCEPTION_LOCATION);
  }

  // Simply forwards to the element's own Hessian-vector-product routine.
  void MyFoldHandler::get_hessian_vector_products(GeneralisedElement *const &elem_pt, Vector<double> const &Y, DenseMatrix<double> const &C, DenseMatrix<double> &product)
  {
    elem_pt->get_hessian_vector_products(Y, C, product);
  }

  // Returns the (single, real) null eigenvector Y found at the fold. The contract is a globally
  // replicated, non-distributed vector on every rank (the nanobind binding and Python-side
  // consumers copy all Ndof entries), so when distributed the local rows are gathered first.
  void MyFoldHandler::get_eigenfunction(Vector<DoubleVector> &eigenfunction)
  {
    eigenfunction.resize(1);
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false);
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      eigenfunction[0] = Y; // copy of the distributed local rows (halo entries are not part of a DoubleVector)
      eigenfunction[0].redistribute(&dist);
      return;
    }
#endif
    eigenfunction[0].build(&dist, 0.0);
    for (unsigned n = 0; n < Ndof; n++)
    {
      eigenfunction[0][n] = Y[n];
    }
  }

  // See the shared comment on MyHopfHandler::apply_maxabs_normalization.
  void MyFoldHandler::apply_maxabs_normalization()
  {
    const unsigned n_row_local = Dist_helper.distributed() ? Dist_helper.base_nrow_local() : Ndof;
    const unsigned first_row = Dist_helper.distributed() ? Dist_helper.base_first_row() : 0;
    double local_max = 0.0;
    for (unsigned n = 0; n < n_row_local; n++) local_max = std::max(local_max, std::fabs(Y[n]));
    const double m = Dist_helper.allreduce_max(local_max);
    if (m <= 0.0) return; // a zero guess has nothing to rescale; leave the default constraint alone
    for (unsigned n = 0; n < n_row_local; n++) Y[n] /= m;
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed()) Y.synchronise();
#endif
    // Phi is the replicated normalisation vector. Rescale it too, so the constraint ROW is O(1) as
    // well -- with an unscaled unit-length Phi its Jacobian entries Phi_i/Count_i would still be
    // O(1/sqrt(N)), which is half of what this is meant to fix. Then take the right-hand side to be
    // the dot product the rescaled guess actually has, so it satisfies the constraint exactly.
    // (Phi.Y is Phi.Phi in the usual case where Phi was set from the guess, but the third
    // constructor takes an independent normalisation vector and this stays correct for it.)
    for (unsigned n = 0; n < Ndof; n++) Phi[n] /= m;
    double local_dot = 0.0;
    for (unsigned n = 0; n < n_row_local; n++) local_dot += Phi[first_row + n] * Y[n];
    Normalization_rhs = Dist_helper.allreduce_sum(local_dot) / eigenweight;
  }

#ifdef OOMPH_HAS_MPI
  // After a Newton update only the owned rows of Y (and, on rank 0, the parameter) are current:
  // refresh the halo copies and broadcast the parameter value so every rank assembles with the
  // same state. Called from Problem::synchronise_all_dofs at the end of each update.
  void MyFoldHandler::synchronise()
  {
    if (!Dist_helper.distributed()) return;
    Y.synchronise();
    Dist_helper.synchronise_scalars({Parameter_pt});
  }
#endif

  // Restores the problem to its original (non-augmented) size, undoing whatever
  // Solve_which_system mode it was left in.
  MyFoldHandler::~MyFoldHandler()
  {
    AugmentedBlockFoldLinearSolver *block_fold_solver_pt = dynamic_cast<AugmentedBlockFoldLinearSolver *>(
        Problem_pt->linear_solver_pt());

    if (block_fold_solver_pt)
    {
      Problem_pt->linear_solver_pt() = block_fold_solver_pt->linear_solver_pt();
      delete block_fold_solver_pt;
    }
    Dist_helper.restore_base_distribution();
  }

  // Switches to the Block_augmented_J mode (original dofs + parameter, no eigenvector) - used
  // e.g. by AugmentedBlockFoldLinearSolver's block-elimination scheme for the fold system.
  void MyFoldHandler::solve_augmented_block_system()
  {
    // The block modes rebuild the dof vector with every-rank scalars and replicated in-place
    // distributions -- meaningless on a distributed problem (blocksolve is refused there anyway).
    if (Dist_helper.distributed()) throw_runtime_error("Fold tracking block-solve modes are not supported on a distributed (--distribute) problem");
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

  // Switches to the Block_J mode (original dofs only, plain Jacobian, no augmentation at all).
  void MyFoldHandler::solve_block_system()
  {
    if (Dist_helper.distributed()) throw_runtime_error("Fold tracking block-solve modes are not supported on a distributed (--distribute) problem");
    if (Solve_which_system != Block_J)
    {
      Problem_pt->GetDofPtr().resize(Ndof);
      Problem_pt->GetDofDistributionPt()->build(Problem_pt->communicator_pt(), Ndof, false);
      Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);
      Solve_which_system = Block_J;
    }
  }

  // Switches back to the Full_augmented mode (original dofs + eigenvector Y + parameter).
  void MyFoldHandler::solve_full_system()
  {
    if (Dist_helper.distributed() && Solve_which_system != Full_augmented) throw_runtime_error("Fold tracking block-solve modes are not supported on a distributed (--distribute) problem");
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

  // Re-derives the member copy of the eigenvector Y from the current dof values (phin, read
  // back from the problem's dof pointers rather than the member array) and rescales it; also
  // prints diagnostic dot/length values against the previous Y for debugging. Note: this
  // divides by phisqr (the sum of squares) rather than sqrt(phisqr) (the vector norm) used in
  // the analogous MyHopfHandler::realign_C_vector() above - left as-is since the exact intended
  // normalization convention here was not verified against the original oomph-lib FoldHandler.
  void MyFoldHandler::realign_C_vector()
  {
    // The dof entries Ndof+1+n point INTO Y itself, so this always was a rescale of Y by the sum
    // of its own squares -- now read directly from the member (owned rows + reduction when
    // distributed) instead of through the replicated dof-pointer table that no longer exists there.
    double dot = 0.0;
    double phisqr = 0.0;
    const unsigned n_row_local = Dist_helper.distributed() ? Dist_helper.base_nrow_local() : Ndof;
    for (unsigned n = 0; n < n_row_local; n++)
    {
      double phin = Y[n];
      dot += Y[n] * phin;
      phisqr += phin * phin;
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      double local_vals[2] = {dot, phisqr}, global_vals[2];
      MPI_Allreduce(local_vals, global_vals, 2, MPI_DOUBLE, MPI_SUM, Problem_pt->communicator_pt()->mpi_comm());
      dot = global_vals[0];
      phisqr = global_vals[1];
    }
#endif
    std::cerr << "DOT OF C and PHi is " << dot << " and PHi^2 = " << phisqr << std::endl;

    for (unsigned n = 0; n < n_row_local; n++)
    {
      Y[n] = Y[n] / phisqr; // Renormalize the c vector
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed()) Y.synchronise();
#endif
  }

  //////////////////////////////////////////////////

  // Constructs the pitchfork handler. The supplied symmetry_vector defines both the initial
  // guess for the null eigenvector Y and (normalized, as Psi/C) the fixed symmetry constraint
  // vector used to pin the amplitude Sigma of the symmetry-breaking component of u. Sigma
  // itself starts at 0 (the symmetric branch). If improved_pitchfork_tracking_on_unstructured_meshes
  // is enabled, the symmetry constraint is instead assembled as an integral <U,Psi> via
  // dedicated generated-code residual contributions (see setup_U_times_Psi_residual_indices()),
  // which is more accurate on meshes where a plain dof-wise dot product is a poor approximation
  // to the L2 inner product.
  MyPitchForkHandler::MyPitchForkHandler(Problem *const &problem_pt, double *const &parameter_pt, const oomph::DoubleVector &symmetry_vector) : Parameter_pt(parameter_pt)
  {
    call_param_change_handler = false;
    eigenweight = 1.0;
    symmetryweight = 1.0;
    Problem_pt = problem_pt;
    Ndof = problem_pt->ndof();
    Dist_helper.initialise(problem_pt);
    Psi.resize(Ndof);
    C.resize(Ndof);
    Dist_helper.build_base_vector(Y);
    Dist_helper.build_base_vector(Count);
    Dist_helper.setup_count_and_nelement(Count, Nelement);
    unsigned n_element = problem_pt->mesh_pt()->nelement();
    // The symmetry vector arrives replicated (full length, identical on every rank)
    double length = 0.0;
    for (unsigned n = 0; n < Ndof; n++)
    {
      length += symmetry_vector[n] * symmetry_vector[n];
    }
    length = sqrt(length);
    for (unsigned n = 0; n < Ndof; n++)
    {
      C[n] = Psi[n] = symmetry_vector[n] / length;
    }
    const unsigned n_row_local = Dist_helper.base_nrow_local();
    const unsigned first_row = Dist_helper.base_first_row();
    for (unsigned n = 0; n < n_row_local; n++)
    {
      Y[n] = symmetry_vector[first_row + n] / length;
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed()) Y.synchronise();
#endif

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
#ifdef OOMPH_HAS_MPI
          if (elem_pt->is_halo()) continue; // halo elements would double-count the integral
#endif
          DenseMatrix<double> psi_i_times_psi_j(elem_pt->ndof());
          initial_orthogonality += this->get_integrated_U_dot_Psi(elem_pt, psi_i_times_psi_j);
        }
      }
      else
      {
        // Dof_pt holds only the local base rows here (nothing has been pushed yet)
        for (unsigned n = 0; n < n_row_local; n++)
        {
          initial_orthogonality += Psi[first_row + n] * (*problem_pt->GetDofPtr()[n]);
        }
      }
#ifdef OOMPH_HAS_MPI
      if (Dist_helper.distributed())
      {
        double global_orth = 0.0;
        MPI_Allreduce(&initial_orthogonality, &global_orth, 1, MPI_DOUBLE, MPI_SUM, problem_pt->communicator_pt()->mpi_comm());
        initial_orthogonality = global_orth;
      }
#endif
      std::cout << "Initial pitchfork symmetry breaking orthogonality <psi,u>=" << initial_orthogonality << std::endl;
    }
    Sigma = 0.0;

    // [u | param | Y | Sigma]: non-distributed in-place build when replicated (see MyFoldHandler's
    // first constructor for what 'true' did under MPI), pointer swap when distributed.
    Dist_helper.build_augmented_dofs({AugmentedDofDistributionHelper::Block::scalar(Parameter_pt),
                                      AugmentedDofDistributionHelper::Block::vector(&Y),
                                      AugmentedDofDistributionHelper::Block::scalar(&Sigma)});
  }

  // Rescales the fixed symmetry vector Psi by ew/eigenweight and records the new weight.
  void MyPitchForkHandler::set_eigenweight(double ew)
  {
    for (unsigned n = 0; n < Ndof; n++)
    {
      Psi[n] *= ew / eigenweight;
    }
    eigenweight = ew;
  }

  // Restores the problem to its original (non-augmented) size.
  MyPitchForkHandler::~MyPitchForkHandler()
  {
    Dist_helper.restore_base_distribution();
  }

  // Computes the element-local contribution to integral(U.Psi) via the mass-matrix-like
  // "residual mode 1" contribution (see set_assembled_residual()/setup_U_times_Psi_residual_indices()),
  // contracting the current dof values U with the fixed symmetry vector Psi through the
  // psi_i*psi_j shape-function-product matrix; used for the improved (mesh-quadrature-based)
  // symmetry constraint on unstructured meshes.
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

  // Augmented dofs of the element: original dofs + eigenvector Y + parameter + Sigma.
  // The augmented block of a pitchfork tracker, read off get_jacobian() below.
  //
  // Layout (see eqn_number): [ base (raw) | parameter (1) | Y (raw) | Sigma (1) ]
  //
  //                base         param       Y           Sigma
  //   base rows    J (+H_sym)   dense       -           dense
  //   param row    dense        -           -           -
  //   Y    rows    H            dense       J           -
  //   Sigma row    -            -           dense       -
  //
  // Two residuals are in play, which is why every term names one: the base state and the symmetry
  // constraint that PitchForkResidualContributionList calls the "mass matrix" residual. With
  // improved_pitchfork_tracking_on_unstructured_meshes the base block picks up
  // Sigma * symmetryDADU_times_Psi, a Hessian of that second residual, ON TOP of the base Jacobian --
  // the case the single-kind spec could not express.
  bool MyPitchForkHandler::get_sparsity_pattern(oomph::GeneralisedElement *const &elem_pt, AugmentedBlockSpec &spec) const
  {
    auto *el = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!el) return false;
    // The residual map only exists when improved_pitchfork_tracking_on_unstructured_meshes is on --
    // that is the only mode with a separate symmetry residual. Without it there is one residual, the
    // one the element is already assembling, which is what -1 means.
    int r_base = -1, r_sym = -1;
    auto it = residual_contribution_indices.find(el->get_code_instance()->get_code());
    if (it != residual_contribution_indices.end())
    {
      r_base = it->second.residual_indices[0];
      r_sym = it->second.residual_indices[1];
      if (r_base < 0) return false; // This element has no base contribution to describe
    }

    typedef AugmentedBlockSpec S;
    spec.resize(4);
    spec.group_is_scalar[0] = false; // u
    spec.group_is_scalar[1] = true;  // the bifurcation parameter
    spec.group_is_scalar[2] = false; // Y
    spec.group_is_scalar[3] = true;  // Sigma
    spec.set(0, 0, S::Jacobian, r_base);
    if (Problem_pt && Problem_pt->improved_pitchfork_tracking_on_unstructured_meshes && r_sym >= 0)
      spec.add(0, 0, S::Hessian, r_sym); // Sigma * d(A.Psi)/dU, on top of the base Jacobian
    spec.set(0, 1, S::Dense);
    spec.set(0, 3, S::Dense);
    spec.set(1, 0, S::Dense);
    spec.set(2, 0, S::Hessian, r_base);
    spec.set(2, 1, S::Dense);
    spec.set(2, 2, S::Jacobian, r_base);
    spec.set(3, 2, S::Dense);
    return true;
  }

  unsigned MyPitchForkHandler::ndof(oomph::GeneralisedElement *const &elem_pt)
  {
    unsigned raw_ndof = elem_pt->ndof();
    return (2 * raw_ndof + 2);
  }

  // Maps a local dof index to the global equation number in the augmented numbering:
  // [0,Ndof) original dofs, Ndof the bifurcation parameter, [Ndof+1,2*Ndof+1) the eigenvector Y,
  // 2*Ndof+1 the symmetry-breaking amplitude Sigma.
  unsigned long MyPitchForkHandler::eqn_number(oomph::GeneralisedElement *const &elem_pt, const unsigned &ieqn_local)
  {
    // Naive numbering -> per-rank interleaved augmented numbering when distributed (identity otherwise)
    unsigned raw_ndof = elem_pt->ndof();
    if (ieqn_local < raw_ndof)
    {
      return Dist_helper.global_eqn(elem_pt->eqn_number(ieqn_local));
    }
    // The bifurcation parameter equation
    else if (ieqn_local == raw_ndof)
    {
      return Dist_helper.global_eqn(Ndof);
    }
    else if (ieqn_local < (2 * raw_ndof + 1))
    {
      return Dist_helper.global_eqn(Ndof + 1 + elem_pt->eqn_number(ieqn_local - 1 - raw_ndof));
    }
    else
    {
      return Dist_helper.global_eqn(2 * Ndof + 1);
    }
  }

  // Assembles the augmented residuals: base residuals, the null-eigenvector equation J*Y=0,
  // the normalization Y.C-1=0 (via Count-weighted accumulation across elements), and (further
  // below) the symmetry constraint <U,Psi>-Sigma=0. On unstructured meshes with the improved
  // tracking option, base residuals/Jacobian and the symmetry-constraint contribution are
  // assembled together via a combined multi-assembly call instead of a plain dof-wise dot product.
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
    residuals[2 * raw_ndof + 1] = -Normalization_rhs / Nelement * eigenweight;
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      unsigned local_eqn = elem_pt->eqn_number(i);
      residuals[raw_ndof + 1 + i] = 0.0;
      for (unsigned j = 0; j < raw_ndof; j++)
      {
        unsigned local_unknown = elem_pt->eqn_number(j);
        residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y.global_value(local_unknown);
      }
      residuals[2 * raw_ndof + 1] += (Y.global_value(local_eqn) * C[local_eqn]) / Count.global_value(local_eqn);
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
        residuals[i] += Sigma * Psi[local_eqn] / Count.global_value(local_eqn);
        residuals[raw_ndof] += ((*Problem_pt->global_dof_pt(local_eqn)) * Psi[local_eqn]) / Count.global_value(local_eqn);
      }
    }
  }

  // Assembles the augmented Jacobian for (u, Y, parameter, Sigma). If analytic Hessian-vector
  // products and parameter derivatives are available (ana_hessian && ana_dparam), all
  // second-derivative blocks (d(J*Y)/du, d(J*Y)/dparameter, and - for the improved unstructured-
  // mesh symmetry constraint - d(<U,Psi>)/du) are obtained from a single combined multi-assembly
  // call; otherwise this falls back to finite-differencing get_residuals()/get_jacobian() per dof
  // (see the FD loops further below), analogous to MyFoldHandler::get_jacobian().
  void MyPitchForkHandler::get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
  {
    bool ana_dparam = Problem_pt->is_dparameter_calculated_analytically(Parameter_pt);
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
        Y_local[_e] = Y.global_value(elem_pt->eqn_number(_e));
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

      residuals[2 * raw_ndof + 1] = -Normalization_rhs / Nelement * eigenweight;
      // Fill augmented residuals
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        unsigned local_eqn = elem_pt->eqn_number(i);
        residuals[raw_ndof + 1 + i] = 0.0;
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          residuals[raw_ndof + 1 + i] += jacobian(i, j) * Y_local[j];
        }

        residuals[2 * raw_ndof + 1] += (Y.global_value(local_eqn) * C[local_eqn]) / Count.global_value(local_eqn);
      }

      // And the Jacobian
      for (unsigned n = 0; n < raw_ndof; n++)
      {
        unsigned local_eqn = elem_pt->eqn_number(n);
        jacobian(n, raw_ndof) = dres_dparam[n];
        jacobian(2 * raw_ndof + 1, raw_ndof + 1 + n) = C[local_eqn] / Count.global_value(local_eqn);
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
          residuals[i] += Sigma * Psi[local_eqn] / Count.global_value(local_eqn);
          jacobian(i, 2 * raw_ndof + 1) += Psi[local_eqn] / Count.global_value(local_eqn);
          residuals[raw_ndof] += ((*Problem_pt->global_dof_pt(local_eqn)) * Psi[local_eqn]) / Count.global_value(local_eqn);
          jacobian(raw_ndof, i) = Psi[local_eqn] / Count.global_value(local_eqn);
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
        jacobian(2 * raw_ndof + 1, raw_ndof + 1 + n) = C[local_eqn] / Count.global_value(local_eqn);
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
          jacobian(i, 2 * raw_ndof + 1) = Psi[local_eqn] / Count.global_value(local_eqn);
          jacobian(raw_ndof, i) = Psi[local_eqn] / Count.global_value(local_eqn);
        }
      }
      const double FD_step = 1.0e-8;
      Vector<double> newres_p(augmented_ndof);

      for (unsigned n = 0; n < raw_ndof; ++n)
      {
        // The element's own BASE equation number -- the handler's eqn_number() now returns the
        // TRANSLATED augmented number when distributed, which global_dof_pt must not be fed with
        unsigned long base_eqn = elem_pt->eqn_number(n);
        double *unknown_pt = Problem_pt->global_dof_pt(base_eqn);
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

  // Derivative of the augmented residuals with respect to the bifurcation parameter, needed
  // for arclength continuation; the symmetry-constraint and normalization rows do not depend
  // on the parameter directly and are set to zero.
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
            djac_dparam(i, j) * Y.global_value(local_unknown);
      }
    }
  }
  void MyPitchForkHandler::get_djacobian_dparameter(oomph::GeneralisedElement *const &, double *const &, oomph::Vector<double> &, oomph::DenseMatrix<double> &)
  {
    throw_runtime_error("implement");
  }
  // Simply forwards to the element's own Hessian-vector-product routine.
  void MyPitchForkHandler::get_hessian_vector_products(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product)
  {
    elem_pt->get_hessian_vector_products(Y, C, product);
  }
  // Returns the (single, real) null eigenvector Y found at the pitchfork. Globally replicated on
  // every rank (gathered when distributed), like the other handlers' get_eigenfunction.
  void MyPitchForkHandler::get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction)
  {
    eigenfunction.resize(1);
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false);
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      eigenfunction[0] = Y;
      eigenfunction[0].redistribute(&dist);
      return;
    }
#endif
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

  // See the shared comment on MyHopfHandler::apply_maxabs_normalization. Only Y and the
  // normalisation vector C are touched; Psi defines the symmetry constraint, a different equation,
  // and rescaling it would change what that constraint means.
  void MyPitchForkHandler::apply_maxabs_normalization()
  {
    const unsigned n_row_local = Dist_helper.distributed() ? Dist_helper.base_nrow_local() : Ndof;
    const unsigned first_row = Dist_helper.distributed() ? Dist_helper.base_first_row() : 0;
    double local_max = 0.0;
    for (unsigned n = 0; n < n_row_local; n++) local_max = std::max(local_max, std::fabs(Y[n]));
    const double m = Dist_helper.allreduce_max(local_max);
    if (m <= 0.0) return;
    for (unsigned n = 0; n < n_row_local; n++) Y[n] /= m;
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed()) Y.synchronise();
#endif
    for (unsigned n = 0; n < Ndof; n++) C[n] /= m;
    double local_dot = 0.0;
    for (unsigned n = 0; n < n_row_local; n++) local_dot += C[first_row + n] * Y[n];
    Normalization_rhs = Dist_helper.allreduce_sum(local_dot) / eigenweight;
  }

#ifdef OOMPH_HAS_MPI
  // After a Newton update only the owned rows of Y (and, on rank 0, the parameter and Sigma) are
  // current: refresh the halo copies and broadcast the scalars. Called from
  // Problem::synchronise_all_dofs at the end of each update.
  void MyPitchForkHandler::synchronise()
  {
    if (!Dist_helper.distributed()) return;
    Y.synchronise();
    Dist_helper.synchronise_scalars({Parameter_pt, &Sigma});
  }
#endif

  // Inspects every generated bulk element code once and records, for each, the residual-form
  // index of the base state and (if present) of the "_simple_mass_matrix_of_defined_fields"
  // variant, used by the improved unstructured-mesh symmetry constraint to switch elements
  // between assembling the base residual and the mass-matrix-like form needed for <U,Psi>.
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

  // Looks up (without switching) the residual-form index for elem_pt's generated code that
  // corresponds to residual_mode (0: base state, 1: mass matrix), as set up by
  // setup_U_times_Psi_residual_indices().
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

  // Switches elem_pt's generated code to assemble the residual form given by residual_mode
  // (0: base state, 1: mass matrix); returns false if that element has no such contribution
  // (residual_indices[residual_mode]<0), in which case the caller should skip it.
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
    Dist_helper.initialise(problem_pt);

    // Resize the vectors of additional dofs
    Dist_helper.build_base_vector(real_eigenvector);
    Dist_helper.build_base_vector(imag_eigenvector);
    normalization_vector.resize(Ndof,0);
    Dist_helper.build_base_vector(Count);
    Dist_helper.setup_count_and_nelement(Count, Nelement);

    // Rotate/normalise the guess on REPLICATED copies (the guesses arrive full-length and
    // identical on every rank, and the rotation involves global dot products), then scatter the
    // owned rows into the -- possibly distributed -- unknowns.
    oomph::Vector<double> re_rot(Ndof), im_rot(Ndof);
    for (unsigned n=0;n<Ndof;n++)
    {
      re_rot[n]=real_eigen[n];
      im_rot[n]=imag_eigen[n];
    }
    rotate_complex_eigenvector_nicely(re_rot,im_rot);
    //std::cout << "BIFTRACKER GOT " << (*parameter_pt) << " " << Omega << " HAS IMAG " << has_imaginary_part << std::endl;
    for (unsigned n=0;n<Ndof;n++) normalization_vector[n]=re_rot[n];
    const unsigned n_row_local = Dist_helper.base_nrow_local();
    const unsigned first_row = Dist_helper.base_first_row();
    for (unsigned n=0;n<n_row_local;n++)
    {
      real_eigenvector[n]=re_rot[first_row+n];
      imag_eigenvector[n]=im_rot[first_row+n];
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      real_eigenvector.synchronise();
      imag_eigenvector.synchronise();
    }
#endif

    // [u | Re(v) | (Im(v)) | param | (Omega)]: non-distributed in-place build when replicated,
    // pointer swap when distributed (see AugmentedDofDistributionHelper)
    std::vector<AugmentedDofDistributionHelper::Block> blocks;
    blocks.push_back(AugmentedDofDistributionHelper::Block::vector(&real_eigenvector));
    if (has_imaginary_part) blocks.push_back(AugmentedDofDistributionHelper::Block::vector(&imag_eigenvector));
    blocks.push_back(AugmentedDofDistributionHelper::Block::scalar(Parameter_pt));
    if (has_imaginary_part) blocks.push_back(AugmentedDofDistributionHelper::Block::scalar(&Omega));
    Dist_helper.build_augmented_dofs(blocks);
  }

  // Destructor (used for cleaning up memory)
  AzimuthalSymmetryBreakingHandler::~AzimuthalSymmetryBreakingHandler()
  {
    // Now return the problem to its original size
    Dist_helper.restore_base_distribution();
  }

  // This will return the degrees of freedom of a single element of the augmented system
  // We will have to take the degrees of freedom of the original element and add a few more for the eigenvector values (Re and Im)
  // The augmented block of the azimuthal symmetry-breaking tracker, read off get_jacobian() below.
  //
  // Layout with an imaginary part (see eqn_number):
  //   [ base (raw) | real eig (raw) | imag eig (raw) | parameter (1) | Omega (1) ]
  // and without it:
  //   [ base (raw) | real eig (raw) | parameter (1) ]
  //
  // THREE residuals are live at once -- base, real azimuthal, imaginary azimuthal -- and the
  // eigenvector blocks mix them: jacobian_real(m,n) - Omega*M_imag(m,n) is the real azimuthal
  // Jacobian OR'd with the imaginary azimuthal mass matrix. That is what the per-term residual index
  // and the OR'd term lists exist for; nothing here could be said in a one-kind-per-block vocabulary.
  //
  // The eigenvector-row base-column blocks combine JHess and MHess of both azimuthal residuals, all of
  // which contributes_to_hessian covers -- it is marked whenever either the Jacobian or the mass part
  // of a second derivative is non-zero.
  bool AzimuthalSymmetryBreakingHandler::get_sparsity_pattern(oomph::GeneralisedElement *const &elem_pt, AugmentedBlockSpec &spec) const
  {
    auto *el = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!el) return false;
    auto it = residual_contribution_indices.find(el->get_code_instance()->get_code());
    if (it == residual_contribution_indices.end()) return false;
    const int r_base = it->second.residual_indices[0];
    const int r_re = it->second.residual_indices[1];
    const int r_im = it->second.residual_indices[2];

    // A negative index means this element has NO contribution to that residual -- not that the block
    // is undescribable. Its blocks are then structurally empty, which is a perfectly good description,
    // so the terms are simply omitted. Returning false instead would abandon the pattern for the whole
    // MESH, because build_frozen_sparsity gives up as soon as one element cannot be described: on the
    // Rayleigh-Benard azimuthal case that was every element, 33012 declines and no frozen path at all.
    auto addT = [&spec](unsigned r, unsigned c, AugmentedBlockSpec::Kind k, int resid) {
      if (resid >= 0) spec.add(r, c, k, resid);
    };

    typedef AugmentedBlockSpec S;
    if (!has_imaginary_part)
    {
      spec.resize(3);
      spec.group_is_scalar[0] = false; // u
      spec.group_is_scalar[1] = false; // the (real) eigenvector
      spec.group_is_scalar[2] = true;  // the bifurcation parameter
    // The axis boundary conditions for the azimuthal mode overwrite whole rows with an identity, for
    // dofs in base_dofs_forced_zero / eigen_dofs_forced_zero. Those entries come from no residual at
    // all, so no coupling table can predict them; the diagonals of the base and eigenvector blocks are
    // therefore declared explicitly. It is a per-dof decision at runtime and a per-block statement
    // here, so a diagonal entry may be stored for dofs that are not forced -- a handful of explicit
    // zeros, and the section 7c hazard they carry is why MUMPS null-pivot detection stays on.
      addT(0, 0, S::Jacobian, r_base);
      spec.add(0, 0, S::Diagonal);
      spec.set(0, 2, S::Dense);
      addT(1, 0, S::Hessian, r_re);
      addT(1, 1, S::Jacobian, r_re);
      spec.add(1, 1, S::Diagonal);
      spec.set(1, 2, S::Dense);
      spec.set(2, 1, S::Dense); // the normalisation row
      return true;
    }

    spec.resize(5);
    spec.group_is_scalar[0] = false; // u
    spec.group_is_scalar[1] = false; // real part of the eigenvector
    spec.group_is_scalar[2] = false; // imaginary part
    spec.group_is_scalar[3] = true;  // the bifurcation parameter
    spec.group_is_scalar[4] = true;  // Omega
    // The axis boundary conditions for the azimuthal mode overwrite whole rows with an identity, for
    // dofs in base_dofs_forced_zero / eigen_dofs_forced_zero. Those entries come from no residual at
    // all, so no coupling table can predict them; the diagonals of the base and eigenvector blocks are
    // therefore declared explicitly. It is a per-dof decision at runtime and a per-block statement
    // here, so a diagonal entry may be stored for dofs that are not forced -- a handful of explicit
    // zeros, and the section 7c hazard they carry is why MUMPS null-pivot detection stays on.
    addT(0, 0, S::Jacobian, r_base);
    spec.add(0, 0, S::Diagonal);
    spec.set(0, 3, S::Dense);
    // Real eigenvector rows
    addT(1, 0, S::Hessian, r_re);
    addT(1, 0, S::Hessian, r_im);
    addT(1, 1, S::Jacobian, r_re);
    addT(1, 1, S::MassMatrix, r_im);
    spec.add(1, 1, S::Diagonal);
    addT(1, 2, S::Jacobian, r_im);
    addT(1, 2, S::MassMatrix, r_re);
    spec.set(1, 3, S::Dense);
    spec.set(1, 4, S::Dense);
    // Imaginary eigenvector rows
    addT(2, 0, S::Hessian, r_re);
    addT(2, 0, S::Hessian, r_im);
    addT(2, 1, S::Jacobian, r_im);
    addT(2, 1, S::MassMatrix, r_re);
    addT(2, 2, S::Jacobian, r_re);
    addT(2, 2, S::MassMatrix, r_im);
    spec.add(2, 2, S::Diagonal);
    spec.set(2, 3, S::Dense);
    spec.set(2, 4, S::Dense);
    // The two normalisation rows. Neither has a diagonal, deliberately: they constrain the
    // eigenvector only, and a Dense entry there would manufacture the stored zero diagonal of §7c.
    spec.set(3, 1, S::Dense);
    spec.set(4, 2, S::Dense);
    return true;
  }

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
    unsigned long global_eqn=0;
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
    // Naive numbering -> per-rank interleaved augmented numbering when distributed (identity otherwise)
    return Dist_helper.global_eqn(global_eqn);
  }

  // This will calculate the residual contribution of the original weak form by calling the function of the element
  // Layout of the augmented residual vector (raw_ndof = elem_pt->ndof()), analogous to
  // MyHopfHandler::get_residuals() but with the azimuthal mode number m folded into the
  // generated code's "residual mode" (0: base/axisymmetric state, 1: real part of the m!=0
  // Jacobian/mass matrix, 2: imaginary part - selected here via set_assembled_residual()):
  //   [0, raw_ndof)              base (axisymmetric) residuals R(u)
  //   [raw_ndof, 2*raw_ndof)     real part of (J_real+i*J_imag)*(Re+i*Im) - i*Omega*(M_real+i*M_imag)*(Re+i*Im)
  //   [2*raw_ndof, 3*raw_ndof)   imag part of the same (only if has_imaginary_part)
  //   3*raw_ndof [+1]            normalization equations, accumulated additively across elements
  // If has_imaginary_part is false, Omega is fixed at 0 and only the real eigen-equation/
  // normalization are assembled, shifting the layout by one block (see the ndof()/eqn_number()
  // comments for the exact offsets in that case). Degrees of freedom listed in
  // base_dofs_forced_zero/eigen_dofs_forced_zero (fixed by the axis boundary condition for the
  // given m) have their residuals patched to zero after the general assembly.
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
      residuals[3 * raw_ndof] = -Normalization_rhs /(double)Nelement* eigenweight;
      residuals[3 * raw_ndof + 1] = 0.0;
    }
    else
    {
      residuals[2 * raw_ndof] = -Normalization_rhs /(double)Nelement* eigenweight;
    }

    // Now multiply to fill in the residuals
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      residuals[raw_ndof + i] = 0.0;
      if (has_imaginary_part) residuals[2 * raw_ndof + i] = 0.0;
      for (unsigned j = 0; j < raw_ndof; j++)
      {
        unsigned global_unknown = elem_pt->eqn_number(j);
        const double re_ev = real_eigenvector.global_value(global_unknown);
        const double im_ev = has_imaginary_part ? imag_eigenvector.global_value(global_unknown) : 0.0;
        // First residual
        if (has_imaginary_part)
        {
          residuals[raw_ndof + i] +=jacobian_real(i, j) * re_ev - jacobian_imag(i, j) * im_ev - Omega * (M_real(i, j) * im_ev + M_imag(i, j) * re_ev);
          residuals[2 * raw_ndof + i] += jacobian_real(i, j) * im_ev + jacobian_imag(i, j) * re_ev + Omega * (M_real(i, j) * re_ev - M_imag(i, j) * im_ev);
        }
        else
        {
          residuals[raw_ndof + i] +=jacobian_real(i, j) * re_ev;
        }
      }
      // Get the global equation number
      unsigned global_eqn = elem_pt->eqn_number(i);

      if (has_imaginary_part)
      {
        residuals[3 * raw_ndof] += (real_eigenvector.global_value(global_eqn) * normalization_vector[global_eqn]) / Count.global_value(global_eqn);
        // Imaginary eigenvector normalization
        residuals[3 * raw_ndof + 1] += (imag_eigenvector.global_value(global_eqn) * normalization_vector[global_eqn]) / Count.global_value(global_eqn);
      }
      else
      {
        residuals[2 * raw_ndof] += (real_eigenvector.global_value(global_eqn) * normalization_vector[global_eqn]) / Count.global_value(global_eqn);
      }
    }

    if (lambda_tracking)
    {
      for (unsigned i = 0; i < raw_ndof; i++)
      {
        for (unsigned j = 0; j < raw_ndof; j++)
        {
          unsigned global_unknown = elem_pt->eqn_number(j);
          const double re_ev = real_eigenvector.global_value(global_unknown);
          if (has_imaginary_part)
          {
            const double im_ev = imag_eigenvector.global_value(global_unknown);
            residuals[raw_ndof + i] +=(*Parameter_pt) * (M_real(i, j) * re_ev - M_imag(i, j) * im_ev);
            residuals[2 * raw_ndof + i] += (*Parameter_pt) * (M_real(i, j) * im_ev + M_imag(i, j) * re_ev);
          }
          else
          {
            residuals[raw_ndof + i] +=(*Parameter_pt) *M_real(i, j) * re_ev;
          }
        }
      }
    }

    //=======Correct residuals according to boundary conditions dependent on m=======//

    // Loop through the RAW dofs
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      // The forced-zero sets hold BASE equation numbers, so use the element's own numbering --
      // the handler's eqn_number() returns the TRANSLATED augmented number when distributed
      unsigned long global_eqn = elem_pt->eqn_number(i);
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
  // Two independent code paths, selected by whether analytic Hessian-vector products and
  // parameter derivatives are available (ana_hessian): the !ana_hessian branch fills the
  // eigen-equation/normalization blocks directly from the real/imag Jacobian and mass-matrix
  // contributions (obtained by switching the generated code's residual mode via
  // set_assembled_residual()) and finite-differences the remaining u- and parameter-derivative
  // blocks by perturbing get_residuals(); the analytic branch instead requests Hessian-vector
  // products (JHess_real/imag, MHess_real/imag = second derivatives of J/M contracted with the
  // eigenvector Eig=(Re,Im)) and parameter derivatives directly from the generated code via a
  // combined multi-assembly call, avoiding any finite differencing.
  void AzimuthalSymmetryBreakingHandler::get_jacobian(oomph::GeneralisedElement *const &elem_pt, oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
  {
    bool lambda_tracking=(Parameter_pt==Problem_pt->get_lambda_tracking_real());
    // (This used to read GetDofPtr()[3*Ndof], which was even out of bounds without an imaginary part)
    bool ana_dparam = lambda_tracking || Problem_pt->is_dparameter_calculated_analytically(Parameter_pt);
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
            jacobian(raw_ndof + n, 3 * raw_ndof + 1) += M_real(n, m) * imag_eigenvector.global_value(global_eqn) +
                                                        M_imag(n, m) * real_eigenvector.global_value(global_eqn);
            jacobian(2 * raw_ndof + n, 3 * raw_ndof + 1) += M_real(n, m) * real_eigenvector.global_value(global_eqn) -
                                                            M_imag(n, m) * imag_eigenvector.global_value(global_eqn);
          }
          else
          {
            jacobian(raw_ndof + n, raw_ndof + m) = jacobian_real(n, m);            
          }
        }

        unsigned local_eqn = elem_pt->eqn_number(n);
        if (has_imaginary_part)
        {
          jacobian(3 * raw_ndof, raw_ndof + n) = normalization_vector[local_eqn] / Count.global_value(local_eqn);
          jacobian(3 * raw_ndof + 1, 2 * raw_ndof + n) = normalization_vector[local_eqn] / Count.global_value(local_eqn);
        }
        else
        {
          jacobian(2 * raw_ndof, raw_ndof + n) = normalization_vector[local_eqn] / Count.global_value(local_eqn);
        }
      }

      const double FD_step = 1.0e-8;

      Vector<double> newres_p(augmented_ndof), newres_m(augmented_ndof);

      // Loop over the dofs
      for (unsigned n = 0; n < raw_ndof; n++)
      {
        // Just do the x's -- perturb via the element's own BASE equation number (halo-aware when
        // distributed), not the handler's translated augmented number
        unsigned long base_eqn = elem_pt->eqn_number(n);
        double *unknown_pt = Problem_pt->global_dof_pt(base_eqn);
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
        this->get_dresiduals_dparameter(elem_pt, Parameter_pt, dres_dparam);
        for (unsigned m = 0; m < augmented_ndof - (has_imaginary_part ? 2 : 1); m++)
        {
          jacobian(m, (has_imaginary_part ? 3 : 2) * raw_ndof) = dres_dparam[m];
        }
      }
      else
      {
        double *unknown_pt = Parameter_pt;
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
        Eig[_e] = real_eigenvector.global_value(index);
        if (has_imaginary_part) Eig[raw_ndof + _e] = imag_eigenvector.global_value(index);
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
      residuals[(has_imaginary_part ? 3 : 2) * raw_ndof] = -Normalization_rhs / (double)Nelement* eigenweight;
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
            residuals[2 * raw_ndof + i] += jacobian_real(i, j) * Eig[raw_ndof + j] + jacobian_imag(i, j) * Eig[j] + Omega * (M_real(i, j) * Eig[j] - M_imag(i, j) * Eig[raw_ndof + j]);
          }
          else
          {
            residuals[raw_ndof + i] += jacobian_real(i, j) * Eig[j] ;
          }
        }
        unsigned global_eqn = elem_pt->eqn_number(i);
        residuals[(has_imaginary_part ? 3 : 2) * raw_ndof] += (real_eigenvector.global_value(global_eqn) * normalization_vector[global_eqn]) / Count.global_value(global_eqn);
        if (has_imaginary_part) residuals[3 * raw_ndof + 1] += (imag_eigenvector.global_value(global_eqn) * normalization_vector[global_eqn]) / Count.global_value(global_eqn);
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

            jacobian(2 * raw_ndof + m, n) = JHess_real(raw_ndof + m, n) + JHess_imag(m, n) + Omega * (MHess_real(m, n) - MHess_imag(raw_ndof + m, n));
            jacobian(2 * raw_ndof + m, raw_ndof + n) = jacobian_imag(m, n) + Omega * M_real(m, n);
            jacobian(2 * raw_ndof + m, 2 * raw_ndof + n) = jacobian_real(m, n) - Omega * M_imag(m, n);
            jacobian(2 * raw_ndof + m, 3 * raw_ndof) += dJ_real_dparam(m, n) * Eig[raw_ndof + n] + dJ_imag_dparam(m, n) * Eig[n] + Omega * (dM_real_dparam(m, n) * Eig[n] - dM_imag_dparam(m, n) * Eig[raw_ndof + n]);
            jacobian(2 * raw_ndof + m, 3 * raw_ndof + 1) += M_real(m, n) * Eig[n] - M_imag(m, n) * Eig[raw_ndof + n];            
          }
          else
          {
            jacobian(raw_ndof + m, n) = JHess_real(m, n);
            jacobian(raw_ndof + m, raw_ndof + n) = jacobian_real(m, n);
            jacobian(raw_ndof + m, 2 * raw_ndof) += dJ_real_dparam(m, n) * Eig[n];
          }
        }
        unsigned local_eqn = elem_pt->eqn_number(m);
        jacobian((has_imaginary_part ? 3 : 2) * raw_ndof, raw_ndof + m) = normalization_vector[local_eqn] / Count.global_value(local_eqn);
        if (has_imaginary_part)   jacobian(3 * raw_ndof + 1, 2 * raw_ndof + m) = normalization_vector[local_eqn] / Count.global_value(local_eqn);
      }


      if (lambda_tracking)
      {
        for (unsigned i = 0; i < raw_ndof; i++)
        {
          for (unsigned j = 0; j < raw_ndof; j++)
          {
            unsigned global_unknown = elem_pt->eqn_number(j);
            const double re_ev = real_eigenvector.global_value(global_unknown);
            if (has_imaginary_part)
            {
              const double im_ev = imag_eigenvector.global_value(global_unknown);
              residuals[raw_ndof + i] +=(*Parameter_pt) * (M_real(i, j) * re_ev - M_imag(i, j) * im_ev);
              residuals[2 * raw_ndof + i] += (*Parameter_pt) * (M_real(i, j) * im_ev + M_imag(i, j) * re_ev);
              //jacobian(raw_ndof + i,j)+=(*Parameter_pt) * (MHess_real(i, j) * re_ev - MHess_imag(i, j) * im_ev);
              //jacobian(2*raw_ndof + i,j)+=(*Parameter_pt) * (MHess_real(i, j) * im_ev + MHess_imag(i, j) * re_ev);
              jacobian(raw_ndof + i,j)+=(*Parameter_pt) * (MHess_real(i, j)  - MHess_imag(i+raw_ndof, j) );
              jacobian(2*raw_ndof + i,j)+=(*Parameter_pt) * (MHess_real(i+raw_ndof, j) + MHess_imag(i, j));
              jacobian(raw_ndof + i,raw_ndof + j)+=(*Parameter_pt)*M_real(i, j);
              jacobian(raw_ndof + i,2*raw_ndof + j)+=-(*Parameter_pt)*M_imag(i, j);
              jacobian(2*raw_ndof + i,raw_ndof + j)+=(*Parameter_pt)*M_imag(i, j);
              jacobian(2*raw_ndof + i,2*raw_ndof + j)+=(*Parameter_pt)*M_real(i, j);
              jacobian(raw_ndof + i, 3 * raw_ndof)+=(M_real(i, j) * re_ev - M_imag(i, j) * im_ev);
              jacobian(2*raw_ndof + i, 3 * raw_ndof)+=(M_real(i, j) * im_ev + M_imag(i, j) * re_ev);
            }
            else
            {
              residuals[raw_ndof + i] +=(*Parameter_pt) *M_real(i, j) * re_ev;
              jacobian(raw_ndof + i,j)+=(*Parameter_pt) * (MHess_real(i, j) );
              jacobian(raw_ndof + i,raw_ndof + j)+=(*Parameter_pt)*M_real(i, j);
              jacobian(raw_ndof + i, 2 * raw_ndof)+=M_real(i, j) * re_ev;
            }
          }
        }
      }
    }


    // Enforce the axis boundary conditions for the current azimuthal mode m: for dofs listed in
    // base_dofs_forced_zero/eigen_dofs_forced_zero, overwrite their Jacobian row with the
    // identity (and zero the residual, for the analytic-Hessian path where residuals were
    // already filled above) so that Newton's method simply keeps them at zero rather than
    // solving the (physically meaningless, for this m) original equation there.
    // Loop through the dofs
    for (unsigned i = 0; i < raw_ndof; i++)
    {
      // The forced-zero sets hold BASE equation numbers, so use the element's own numbering --
      // the handler's eqn_number() returns the TRANSLATED augmented number when distributed
      unsigned long global_eqn = elem_pt->eqn_number(i);
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
        const double re_ev = real_eigenvector.global_value(global_unknown);
        // Real part
        if (has_imaginary_part)
        {
          const double im_ev = imag_eigenvector.global_value(global_unknown);
          if (lambda_tracking && parameter_pt==Parameter_pt)
          {
              dres_dparam[raw_ndof + i] += (dM_real_dparam(i, j) * re_ev - dM_imag_dparam(i, j) * im_ev);
              dres_dparam[2 * raw_ndof + i] += (dM_real_dparam(i, j) * im_ev + dM_imag_dparam(i, j) * re_ev);
          }
          else
          {
            dres_dparam[raw_ndof + i] +=
                djac_real_dparam(i, j) * re_ev - djac_imag_dparam(i, j) * im_ev -
                Omega * (dM_real_dparam(i, j) * im_ev + dM_imag_dparam(i, j) * re_ev);
            // Imaginary part
            dres_dparam[2 * raw_ndof + i] +=
                djac_real_dparam(i, j) * im_ev + djac_imag_dparam(i, j) * re_ev +
                Omega * (dM_real_dparam(i, j) * re_ev - dM_imag_dparam(i, j) * im_ev);
            if (lambda_tracking)
            {
              dres_dparam[raw_ndof + i] += (*Parameter_pt) * (dM_real_dparam(i, j) * re_ev - dM_imag_dparam(i, j) * im_ev);
              dres_dparam[2 * raw_ndof + i] += (*Parameter_pt) * (dM_real_dparam(i, j) * im_ev + dM_imag_dparam(i, j) * re_ev);
            }
          }

        }
        else
        {
          if (lambda_tracking && parameter_pt==Parameter_pt)
          {
            dres_dparam[raw_ndof + i] += dM_real_dparam(i, j) * re_ev;
          }
          else
          {
            dres_dparam[raw_ndof + i] +=djac_real_dparam(i, j) * re_ev;
            if (lambda_tracking)
            {
              dres_dparam[raw_ndof + i] += (*Parameter_pt) * dM_real_dparam(i, j) * re_ev;
            }
          }
        }
      }
    }
  }

  // Derivative of the augmented Jacobian with respect to the parameter
  void AzimuthalSymmetryBreakingHandler::get_djacobian_dparameter(oomph::GeneralisedElement *const &, double *const &, oomph::Vector<double> &, oomph::DenseMatrix<double> &)
  {
    throw_runtime_error("AzimuthalSymmetryBreakingHandler::get_djacobian_dparameter(oomph::GeneralisedElement* const &elem_pt,double* const &parameter_pt,oomph::Vector<double> &dres_dparam,oomph::DenseMatrix<double> &djac_dparam)");
    // TODO: Fill it
    // Is it required?
  }

  // Get the eigenfunction
  void AzimuthalSymmetryBreakingHandler::get_eigenfunction(oomph::Vector<oomph::DoubleVector> &eigenfunction)
  {
    // There is a real and imaginary part of the eigen vector. The contract is a globally
    // replicated, non-distributed vector on every rank, so gather when distributed.
    eigenfunction.resize((has_imaginary_part ? 2 : 1)); // So we must return two real vectors (Re and Im)
    // build a distribution for the storage of the eigenvector parts
    LinearAlgebraDistribution dist(Problem_pt->communicator_pt(), Ndof, false); // The eigenvectors have Ndof entries, i.e. the number of dofs of the original problem
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      eigenfunction[0] = real_eigenvector;
      eigenfunction[0].redistribute(&dist);
      if (has_imaginary_part)
      {
        eigenfunction[1] = imag_eigenvector;
        eigenfunction[1].redistribute(&dist);
      }
      return;
    }
#endif
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

  // See the shared comment on MyHopfHandler::apply_maxabs_normalization. The max is taken over the
  // real and imaginary parts jointly, so the complex eigenvector's phase is untouched.
  void AzimuthalSymmetryBreakingHandler::apply_maxabs_normalization()
  {
    const unsigned n_row_local = Dist_helper.distributed() ? Dist_helper.base_nrow_local() : Ndof;
    const unsigned first_row = Dist_helper.distributed() ? Dist_helper.base_first_row() : 0;
    double local_max = 0.0;
    for (unsigned n = 0; n < n_row_local; n++)
    {
      local_max = std::max(local_max, std::fabs(real_eigenvector[n]));
      if (has_imaginary_part) local_max = std::max(local_max, std::fabs(imag_eigenvector[n]));
    }
    const double m = Dist_helper.allreduce_max(local_max);
    if (m <= 0.0) return;
    for (unsigned n = 0; n < n_row_local; n++)
    {
      real_eigenvector[n] /= m;
      if (has_imaginary_part) imag_eigenvector[n] /= m;
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      real_eigenvector.synchronise();
      if (has_imaginary_part) imag_eigenvector.synchronise();
    }
#endif
    for (unsigned n = 0; n < Ndof; n++) normalization_vector[n] /= m;
    double local_dot = 0.0;
    for (unsigned n = 0; n < n_row_local; n++) local_dot += normalization_vector[first_row + n] * real_eigenvector[n];
    Normalization_rhs = Dist_helper.allreduce_sum(local_dot) / eigenweight;
  }

#ifdef OOMPH_HAS_MPI
  // After a Newton update only the owned rows of the eigenvector parts (and, on rank 0, the
  // parameter and Omega) are current: refresh the halo copies and broadcast the scalars.
  // Called from Problem::synchronise_all_dofs at the end of each update.
  void AzimuthalSymmetryBreakingHandler::synchronise()
  {
    if (!Dist_helper.distributed()) return;
    real_eigenvector.synchronise();
    if (has_imaginary_part) imag_eigenvector.synchronise();
    if (has_imaginary_part)
      Dist_helper.synchronise_scalars({Parameter_pt, &Omega});
    else
      Dist_helper.synchronise_scalars({Parameter_pt});
  }
#endif

   void AzimuthalSymmetryBreakingHandler::set_eigenweight(double ew)
  {
    // The eigenvector parts are the unknowns here: rescale the owned rows and refresh the halos
    const unsigned n_row_local = Dist_helper.distributed() ? Dist_helper.base_nrow_local() : Ndof;
    for (unsigned n = 0; n < n_row_local; n++)
    {
      real_eigenvector[n] *= ew / eigenweight;
      imag_eigenvector[n] *= ew / eigenweight;
    }
#ifdef OOMPH_HAS_MPI
    if (Dist_helper.distributed())
    {
      real_eigenvector.synchronise();
      imag_eigenvector.synchronise();
    }
#endif
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

  // Looks up (without switching) the residual-form index for elem_pt's generated code that
  // corresponds to residual_mode (0: base/axisymmetric, 1: real azimuthal, 2: imag azimuthal),
  // as set up by setup_solved_azimuthal_contributions().
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

  // Switches elem_pt's generated code to assemble the residual form given by residual_mode
  // (0: base/axisymmetric, 1: real azimuthal, 2: imag azimuthal); returns false if that element
  // has no such contribution, in which case the caller should skip it.
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



  // A 1D oomph::Node used purely as a bookkeeping device for the periodic-orbit time mesh:
  // its single "spatial" coordinate stores the normalized orbit coordinate s in [0,1), and
  // 'index' records which entry of the augmented dof set (which discrete time point / Tadd
  // slot) this node corresponds to.
  class TimeNode : public oomph::Node
  {
    protected:
      unsigned index;
    public:
      TimeNode(double s,unsigned _index) : oomph::Node(1,1,1), index(_index) {this->x(0)=s;}
      unsigned get_index() {return index;}
  };

  

  


  //////// PERIODIC ORBIT TRACKER

  // Sets up the augmented dof set for periodic-orbit tracking: the current problem dofs
  // (at s=0) become x0, and 'tadd' (the caller's initial guess for the orbit at further
  // discrete time points) is pushed onto the problem's dof pointers as additional unknowns,
  // together with the period T itself. The Poincare-plane constraint (x0, n0, d_plane) used
  // to fix the phase of the orbit (when T_constraint_mode==0) is derived from the initial
  // guess: n0 is the (normalized) direction from the first to the last extra time point,
  // i.e. approximately the orbit's velocity direction at s=0, so that the plane x.n0=d_plane
  // cuts transversally through the orbit near the starting point. The remainder of the
  // constructor (below) sets up whichever discretization (B-spline / Floquet / collocation)
  // was requested; see the class comment in bifurcation.hpp for the three modes.
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
    // Non-distributed: see MyFoldHandler's first constructor for what true does under MPI.
    problem_pt->GetDofDistributionPt()->build(problem_pt->communicator_pt(), Ndof * (Tadd.size()+1)+1 , false); 
    Problem_pt->GetSparcseAssembleWithArraysPA().resize(0);

    
    
    // If not given explicitly, distribute the knots (normalized orbit coordinates s in [0,1])
    // uniformly over the discrete time points; otherwise just validate the caller-supplied knots.
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
    // For the (non-B-spline, non-collocation) modes, precompute the finite-difference weights
    // and neighboring-knot indices used to approximate du/ds at each knot: bspline_order==-1
    // gives a first-order-accurate central difference, bspline_order==-2 a second-order-accurate
    // backward difference, and bspline_order==0 a simple forward difference between consecutive
    // knots (used in the plain Floquet/time-nodal mode). Indices wrap periodically via
    // get_periodic_knot_index()/get_knot_value() since the orbit is periodic in s.
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
      // Orthogonal-collocation ("time mesh") mode: build a 1D finite-element mesh whose
      // "spatial" coordinate is the orbit parameter s, with elements of collocation order
      // 'order' (encoded as bspline_order=-(order+2)) and Gauss-Legendre integration of order
      // gl_order=order within each element; time_mesh/collocation_gl are then used by
      // get_residuals_collocation_mode()/get_jacobian_collocation_mode().
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

  // Returns the s-value of knot index i, extended periodically: indices outside [0,s_knots.size()-1)
  // wrap around by +-L (L being the total period length in s, i.e. 1.0) so that finite-difference
  // stencils near the start/end of the orbit can transparently reference "neighboring" knots on
  // the other side of the periodic boundary.
  double PeriodicOrbitHandler::get_knot_value(int i)
  {
    double L=this->s_knots.back()-this->s_knots.front();
    double offs=0.0;
    while (i<0) { i+=this->s_knots.size()-1; offs-=L; }
    while (i>=(int)(this->s_knots.size())-1) { i-=(int)this->s_knots.size()-1; offs+=L; }
    return this->s_knots[i]+offs;
  }

  // Like get_knot_value(), but returns the wrapped-around knot *index* (without an offset),
  // e.g. to index into Tadd/dof arrays for a periodically-extended stencil position.
  unsigned PeriodicOrbitHandler::get_periodic_knot_index(int i)
  {
    while (i<0) { i+=this->s_knots.size()-1; }
    while (i>=(int)this->s_knots.size()-1) { i-=(int)this->s_knots.size()-1;  }
    return i;
  }

  // Frees the time mesh/collocation integral/B-spline basis (if any) and restores the problem
  // to its original (non-augmented) size.
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
    if (basis) {delete this->basis; this->basis=NULL;}
  }
  // Maps a local dof index to the global equation number: the first nT*raw_ndof local dofs
  // are the element's raw dofs replicated over all nT=n_tsteps() discrete time points
  // (tindex = which time point, local_eqn = which raw dof; the global numbering groups all
  // dofs of one time point together, offset by Ndof*tindex), and the final local dof is the
  // shared unknown period T (T_global_eqn).
  // Describes the augmented block of a periodic-orbit solve for the frozen sparsity machinery.
  //
  // The augmented unknowns are nT copies of the raw dof set -- one per time node around the orbit --
  // followed by the single scalar period T, so the block is an (nT+1)x(nT+1) grid of groups. Within a
  // pair of coupled time nodes the pattern is just the base Jacobian and mass matrix of the underlying
  // element, so the only thing this has to describe is WHICH TIME NODES COUPLE. That is what differs
  // between the discretisations (see the class comment), and both answers are read off the very data
  // the assembly loops use rather than re-derived here:
  //
  //   * B-spline / collocation (basis != NULL): two time nodes couple iff they are supported on a
  //     common basis element. get_integration_info() reports that support as `indices`, exactly as
  //     get_jacobian_bspline_mode() consumes it.
  //   * Floquet / plain finite differences (basis == NULL): a node couples to itself, to its
  //     neighbours, and to whatever the ds stencil FD_ds_inds[t] reaches.
  //
  // Deliberately conservative in two places, because over-describing only costs stored zeros while
  // under-describing would be wrong: the wrap-around identity that closes the orbit is declared for
  // every mode (it is written only in Floquet mode, where the last block row is flushed first), and
  // the base coupling is left on that flushed row too rather than special-cased per mode.
  bool PeriodicOrbitHandler::get_sparsity_pattern(oomph::GeneralisedElement *const &elem_pt, AugmentedBlockSpec &spec) const
  {
    pyoomph::BulkElementBase *pyoomph_elem_pt = dynamic_cast<pyoomph::BulkElementBase *>(elem_pt);
    if (!pyoomph_elem_pt) return false;
    auto *ft = pyoomph_elem_pt->get_code_instance()->get_func_table();
    if (!ft) return false;
    // The orbit assembles J and M together from this residual in a single multi-assemble pass; if the
    // element does not have it, we cannot say what its pattern is.
    const int resind = (int)ft->current_res_jac;
    if (resind < 0) return false;

    const unsigned nT = this->n_tsteps();
    if (nT < 2) return false;
    const unsigned Tgrp = nT; // the scalar period

    spec.resize(nT + 1);
    for (unsigned t = 0; t < nT; t++) spec.group_is_scalar[t] = false;
    spec.group_is_scalar[Tgrp] = true;

    // Couples time node a to time node b with the base Jacobian and mass-matrix patterns. Both are
    // needed: the residual carries J*u from the spatial operator and M*du/ds from the time derivative.
    // Also records that a carries an equation at all, which the period column below is gated on --
    // not every time node does (see the collocation branch).
    std::vector<bool> is_eqn_row(nT, false);
    auto couple = [&](unsigned a, unsigned b)
    {
      if (a >= nT || b >= nT) return;
      is_eqn_row[a] = true;
      spec.add(a, b, AugmentedBlockSpec::Jacobian, resind);
      spec.add(a, b, AugmentedBlockSpec::MassMatrix, resind);
    };

    // Same priority order as get_residuals()/get_jacobian(): collocation, Floquet, nodal FD, B-spline.
    if (!this->basis)
    {
      if (this->time_mesh)
      {
        // Collocation (the default). Rows and columns do NOT run over the same node set, which is the
        // whole subtlety here: a time element with nnode() nodes carries only nnode()-1 collocation
        // points, and get_jacobian_collocation_mode() writes the equation of collocation point `inode`
        // into the row of node `inode` -- so only the FIRST nnode()-1 nodes of an element are equation
        // rows, while all nnode() of them are interpolated and hence appear as columns. Declaring the
        // last node as a row too is what made every element-boundary block (3,0), (6,3), (9,6), ...
        // come out 100% structural zeros. The time index a node stands for is TimeNode::get_index(),
        // which is what the assembly indexes the block with.
        for (unsigned ie = 0; ie < this->time_mesh->nelement(); ie++)
        {
          oomph::FiniteElement *el = dynamic_cast<oomph::FiniteElement *>(this->time_mesh->element_pt(ie));
          if (!el || el->nnode() < 2) return false;
          const unsigned nn = el->nnode();
          for (unsigned in = 0; in + 1 < nn; in++) // rows: collocation points only
          {
            TimeNode *ni = dynamic_cast<TimeNode *>(el->node_pt(in));
            if (!ni) return false;
            for (unsigned in2 = 0; in2 < nn; in2++) // columns: every node of the element
            {
              TimeNode *nj = dynamic_cast<TimeNode *>(el->node_pt(in2));
              if (!nj) return false;
              couple(ni->get_index(), nj->get_index());
            }
          }
        }
      }
      else if (this->floquet_mode)
      {
        // Floquet: trapezoidal between consecutive time nodes, so a node couples to itself and to its
        // successor only -- both through J and M. The last block row carries no equation of its own;
        // it is flushed and replaced by the wrap-around identity declared below, which is also why it
        // must not be marked as an equation row here (it has no period column either).
        for (unsigned t = 0; t + 1 < nT; t++)
        {
          couple(t, t);
          couple(t, t + 1);
        }
      }
      else
      {
        // Plain nodal finite differences (central, BDF2). Unlike Floquet there is NO t -> t+1 term:
        // the spatial operator is evaluated at the node itself, and the only coupling to other time
        // nodes is the dU/ds stencil, whose reach is FD_ds_inds[t]. That stencil multiplies the MASS
        // matrix alone, so declaring the Jacobian pattern there as well would be pure over-inclusion
        // (it cost +44% nnz for central and +89% for BDF2 when this branch was a guess).
        for (unsigned t = 0; t < nT; t++)
        {
          couple(t, t);
          if (t < this->FD_ds_inds.size())
            for (unsigned k = 0; k < this->FD_ds_inds[t].size(); k++)
            {
              const unsigned b = this->FD_ds_inds[t][k];
              if (b < nT) { is_eqn_row[t] = true; spec.add(t, b, AugmentedBlockSpec::MassMatrix, resind); }
            }
        }
      }
    }
    else
    {
      // B-spline: two time nodes couple iff they are supported on a common basis element, which the
      // basis reports as `indices` -- exactly what get_jacobian_bspline_mode() indexes the block with.
      for (unsigned ie = 0; ie < this->basis->get_num_elements(); ie++)
      {
        std::vector<double> w;
        std::vector<unsigned> indices;
        std::vector<std::vector<double>> psi_s, dpsi_ds;
        this->basis->get_integration_info(ie, w, indices, psi_s, dpsi_ds);
        for (unsigned l = 0; l < indices.size(); l++)
          for (unsigned l2 = 0; l2 < indices.size(); l2++)
            couple(indices[l], indices[l2]);
      }
    }

    // The period column dR/dT (from the M*dU/ds terms, which carry a 1/T), and the phase/plane
    // constraint row that pins the orbit's time origin. Neither has a pattern of its own.
    for (unsigned t = 0; t < nT; t++)
    {
      // dR/dT comes from the M*dU/ds term, so it exists exactly where there is an equation to
      // differentiate; a time node that is only ever interpolated (collocation's element-boundary
      // node) has no period column.
      if (is_eqn_row[t]) spec.set(t, Tgrp, AugmentedBlockSpec::Dense);
      spec.set(Tgrp, t, AugmentedBlockSpec::Dense);
    }
    spec.set(Tgrp, Tgrp, AugmentedBlockSpec::Dense);

    // The wrap-around u(nT-1) - u(0) = 0 closing the orbit in Floquet mode. It is an identity, so it
    // lands on the DIAGONAL of the off-diagonal block (nT-1, 0) -- not on a Jacobian pattern, and not
    // on a square block, which is why AugmentedBlockSpec::Diagonal is not restricted to gr == gc.
    spec.add(nT - 1, nT - 1, AugmentedBlockSpec::Diagonal);
    spec.add(nT - 1, 0, AugmentedBlockSpec::Diagonal);

    return true;
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
  
 

  // Augmented dofs of the element: its raw dofs replicated at each of the nT discrete time
  // points making up the orbit, plus one shared dof for the period T.
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

  // Orthogonal-collocation residual assembly. The periodic BVP being discretized is
  // (1/T) dU/ds = f(U) (f being the original problem's residual-implied time derivative,
  // s in [0,1) the normalized orbit coordinate, T the unknown period), enforced pointwise at
  // the Gauss-Legendre collocation points within each element of time_mesh. At each collocation
  // point: U(s) and dU/ds(s) are interpolated from the surrounding nodal dof values (dof_backup
  // for the node aliased to the problem's current dofs, Tadd[index-1] for the other nodes), the
  // interpolated U is temporarily written into the problem's dof pointers so that the original
  // element's get_jacobian_and_mass_matrix() (or its parameter-derivative variant, if
  // parameter_pt!=NULL) evaluates f and M at that state, and the weighted residual
  // f(U)*w + M(U)/T*(dU/ds)*w is accumulated into the residual block belonging to that
  // collocation node's time index. Afterwards the temporarily overwritten dofs are restored.
  // If floquet_mode, an extra closure residual enforces periodicity between the last explicit
  // time point and the (aliased) starting point. Finally, unless computing a parameter
  // derivative, one additional scalar residual enforces the phase-fixing constraint: either the
  // Poincare-plane condition (T_constraint_mode==0, using x0/n0/d_plane) or, for
  // T_constraint_mode==1, orthogonality of the current orbit's velocity to the reference orbit's
  // velocity du0ds (integrated collocation-wise inside the main loop above).
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

  
  // Floquet-mode residual assembly: a trapezoidal-like (midpoint-rule) discretization of
  // (1/T) dU/ds = f(U) between each pair of consecutive discrete time points (U0 at ti, Uplus
  // at ti+1): evaluates f and M at the midpoint state U=(U0+Uplus)/2 and forms
  // f(U) + M(U)/T*dUds, with dUds obtained from the finite-difference weight FD_ds_weights[ti][0]
  // precomputed in the constructor. The final explicit time point is not looped over directly;
  // instead a closure residual ties it to the (aliased) starting point, enforcing periodicity.
  // As in get_residuals_collocation_mode(), one extra scalar residual fixes the phase, via
  // either the Poincare-plane condition or orthogonality to the reference orbit's velocity du0ds.
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


  // Analytic-Hessian counterpart of get_residuals_floquet_mode(): requires analytic Hessian-
  // vector products (the du/ds-contracted Hessian dMdU_dUdsterm is requested per midpoint via a
  // multi-assembly call) since finite-differencing the whole periodic-orbit Jacobian would be
  // far too expensive; reuses the same midpoint/finite-difference discretization as the residual
  // routine to fill in the corresponding Jacobian blocks between each pair of time points.
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

  // Plain nodal finite-difference residual assembly (used for bspline_order in {-1,-2}, i.e.
  // central or backward-difference discretizations of dU/ds): unlike the Floquet-mode midpoint
  // rule, here f(U) and M(U) are evaluated directly at each nodal time point ti (not a midpoint),
  // and dU/ds at that point is formed from the precomputed FD_ds_weights[ti]/FD_ds_inds[ti]
  // stencil (a linear combination of dof_backup and the relevant Tadd entries). Otherwise
  // mirrors get_residuals_floquet_mode(): accumulates f(U)+M(U)/T*dUds into the residual block
  // for time index ti, plus the phase-fixing constraint residual.
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

  // Weighted-residual (Galerkin) assembly for the B-spline discretization: for each B-spline
  // basis element and each of its Gauss-Legendre integration points (weights/shape functions
  // psi_s/dpsi_ds and the contributing dof indices given by this->basis->get_integration_info()),
  // interpolate U(s)/dU/ds(s), evaluate f(U) and M(U) there, and scatter the weighted residual
  // f(U)*w + M(U)/T*dUds*w into the residual blocks of every time index the basis function
  // touches (weighted by its shape-function value psi_s), i.e. a standard Galerkin projection of
  // (1/T) dU/ds = f(U) onto the periodic B-spline space. As in the other modes, one extra scalar
  // residual fixes the orbit's phase (Poincare-plane or reference-orbit-orthogonality constraint).
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
            //unsigned glob_eq=elem_pt->eqn_number(i);
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

  // Dispatches to the residual-assembly routine matching the active discretization: collocation
  // (time_mesh set), Floquet (explicit periodic node), plain nodal finite differences, or
  // B-spline (basis set), in that priority order. Under PYOOMPH_BIFURCATION_HANDLER_DEBUG, also
  // cross-checks the result against the corresponding get_jacobian_*_mode()'s own residual
  // output as a consistency check between the two.
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

  // Analytic-Hessian counterpart of get_residuals_collocation_mode(): re-evaluates the same
  // per-collocation-point residual (f(U)+M(U)/T*dUds) while additionally requesting the
  // Hessian-vector product of J and M contracted with dUds (via multi-assembly, when the mass
  // matrix is not provably constant) so that the du/ds-dependent Jacobian terms can be filled
  // analytically instead of by finite differences; the resulting element Jacobian and mass
  // matrix are scattered into the (time-index, time-index) blocks weighted by the collocation
  // shape functions, mirroring the residual scatter pattern of the residual routine.
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


  // Analytic-Hessian counterpart of get_residuals_bspline_mode(): requires analytic Hessian-
  // vector products; re-runs the same Galerkin quadrature loop over B-spline basis elements
  // and Gauss-Legendre points, additionally scattering the element Jacobian/mass-matrix (and,
  // where the mass matrix is not constant, the du/ds-contracted Hessian) into the Jacobian
  // blocks for every pair of time indices touched by the current integration point, weighted
  // by products of shape-function values/derivatives.
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

  // Snapshots the current orbit (the reference orbit at the time this is called, typically
  // right after a converged solve) into du0ds, the nodal values used by the T_constraint_mode==1
  // phase constraint (orthogonality of the new orbit's velocity to the reference orbit's
  // velocity). Only relevant when T_constraint_mode==1; a no-op otherwise. In the collocation
  // and B-spline modes, the *values* of the reference orbit are stored (its derivative is later
  // formed via the same shape-function derivatives used for the unknown orbit, inside the
  // residual/Jacobian routines); in the Floquet and plain finite-difference modes, the
  // (unscaled, since any constant ds-factor cancels in the resulting integral) finite-difference
  // derivative is precomputed directly here.
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

  // Analytic-Hessian counterpart of get_residuals_time_nodal_mode(): re-evaluates f(U) and M(U)
  // at each nodal time point (not a midpoint) and fills in the corresponding Jacobian blocks
  // using the precomputed FD_ds_weights/FD_ds_inds stencil for the du/ds dependence, requesting
  // Hessian-vector products where the mass matrix is not provably constant.
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

  // Dispatches to the Jacobian-assembly routine matching the active discretization (same
  // priority order as get_residuals()). Under PYOOMPH_BIFURCATION_HANDLER_DEBUG, both a finite-
  // difference Jacobian (obtained by perturbing every augmented dof and calling get_residuals())
  // and the analytic one from the dispatched *_mode() routine are computed and compared
  // entry-by-entry, printing any mismatch; the analytic values are then used regardless. This
  // debug path is expensive (one extra residual evaluation per dof) and only active when the
  // macro is defined.
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

  // Saves the current (s=0) problem dof values into backed_up_dofs, so they can be temporarily
  // overwritten (via set_dofs_to_interpolated_values()) and later restored with restore_dofs().
  // Throws if a backup is already in progress (nested backup/restore is not supported).
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
  // Restores the problem dofs saved by backup_dofs() and clears the backup.
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
  // Overwrites the problem's dofs with the orbit state interpolated at normalized orbit
  // coordinate s (wrapped periodically into [0,1) via clamped_s), using linear interpolation
  // between the two bracketing discrete time points in the non-B-spline modes, or the B-spline
  // basis's own interpolation (basis->get_interpolation_info()) otherwise. Requires backup_dofs()
  // to have been called first, since the s=0 state is read from backed_up_dofs rather than the
  // (already overwritten, in a typical sampling loop) live dof values.
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

  // Returns quadrature samples (s_i, w_i) such that integral_0^1 f(U(s)) ds ~= sum_i f(U(s_i))*w_i,
  // matching whichever discretization is active: midpoint-rule samples between knots for the
  // collocation/Floquet modes, knot-centered samples with a locally averaged spacing for the
  // plain finite-difference mode, or the true Gauss-Legendre points/weights of the B-spline basis.
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


  // Derivative of the augmented residuals with respect to a parameter: dispatches to the same
  // *_mode() residual routines used by get_residuals(), but with parameter_pt!=NULL so that
  // they evaluate the parameter-derivative variant of the element residual/mass matrix instead
  // (see the "if (!parameter_pt) ... else ..." branches inside each get_residuals_*_mode()).
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


  void PeriodicOrbitHandler::get_djacobian_dparameter(GeneralisedElement *const &,double *const &,Vector<double> &,DenseMatrix<double> &)
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
        if (static_cast<size_t>(hessian_vector_indices[i])>=hessian_vectors.size()) throw_runtime_error("Hessian vector index out of bounds");
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
  
  void CustomMultiAssembleHandler::get_residuals(oomph::GeneralisedElement* const&,Vector<double>&)
  {
    throw_runtime_error("Residual called");
  }

  void CustomMultiAssembleHandler::get_jacobian(oomph::GeneralisedElement* const&,oomph::Vector<double>&,oomph::DenseMatrix<double>&)
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


