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
 This file is strongly related to the file timestepper.h from oomph-lib
 We merge all time steppers in a single class here, i.e. we store the weights of all different time steppers from oomph-lib
*/

#pragma once
#include "oomph_lib.hpp"
namespace pyoomph
{
  // A time stepper that provides multiple weights, namely
  // First derivative: BDF1, BDF2, BDF12 (First step: First order, then second order), Newmark2
  // Second derivative: Newmark2
  // Temporal adaptivity can be evaluated based on First derivative BDF2
  // Nodal storage is as follows:
  // t=0	:	current value
  // t=1	:	previous time step
  // t=2	:	value two time steps ago
  // t=3	:  Newmark2 velo
  // t=4	:  Newmark2 accel
  // IF ADAPTIVE:
  //	t=5	:	BDF2 velocity
  //	t=6	:	Predictor

  // A time stepper that computes and stores several sets of finite-difference weights
  // (BDF1, BDF2, Newmark2) simultaneously on the same nodal/data time history, rather than
  // committing to a single scheme. The scheme actually used to advance the solution is
  // whichever weight matrix (`Weight`, inherited from oomph::TimeStepper) is filled by
  // set_weights() - currently Newmark2 - while the other weight matrices (WeightBDF1,
  // WeightBDF2) are kept around so e.g. a BDF2-based temporal error estimate can be formed
  // without re-deriving history-dependent weights from scratch. In the adaptive case, an
  // additional predictor step (AB2-like) and error weight are computed to support adaptive
  // timestepping (see set_predictor_weights/set_error_weights/temporal_error_in_*).
  class MultiTimeStepper : public oomph::TimeStepper
  {
  protected:
    static const unsigned NSTEPS;
    static const unsigned NWEIGHT; // For non-adaptive case: Same as Newmark
    static const unsigned MAXDERIV;

    double NewmarkBeta1;
    double NewmarkBeta2;

    oomph::Vector<double> Predictor_weight;
    double Error_weight;

    unsigned unsteady_steps_done_for_degrading; // How many unsteady steps have been done (required for degrading)

    oomph::DenseMatrix<double> WeightBDF1, WeightBDF2, WeightNewmark2; // Weight matrices for each scheme, laid out like the inherited Weight matrix (,WeightBDF12;)
  public:
    MultiTimeStepper(const bool &adaptive = false) : oomph::TimeStepper(NWEIGHT, MAXDERIV), NewmarkBeta1(0.5), NewmarkBeta2(0.5), unsteady_steps_done_for_degrading(0)
    {
      Type = "MultiTimeStepper";
      if (adaptive)
      {
        Adaptive_Flag = true;
        Predictor_weight.resize(NSTEPS + 2);
        Weight.resize(3, NSTEPS + 5, 0.0);
        Predictor_storage_index = NSTEPS + 4;
      }
      WeightBDF1.resize(Weight.nrow(), Weight.ncol(), 0.0);
      WeightBDF2.resize(Weight.nrow(), Weight.ncol(), 0.0);
      //    WeightBDF12.resize(Weight.nrow(),Weight.ncol(),0.0);
      WeightNewmark2.resize(Weight.nrow(), Weight.ncol(), 0.0);
      Weight(0, 0) = 1.0;
      WeightBDF1(0, 0) = 1.0;
      WeightBDF2(0, 0) = 1.0;
      //    WeightBDF12(0,0) = 1.0;
      WeightNewmark2(0, 0) = 1.0;
    }

    MultiTimeStepper(const MultiTimeStepper &)
    {
      oomph::BrokenCopy::broken_copy("MultiTimeStepper");
    }

    void operator=(const MultiTimeStepper &)
    {
      oomph::BrokenCopy::broken_assign("MultiTimeStepper");
    }

    unsigned order() const
    {
      /*		std::string error_message =
           "Can't remember the order of the MultiTimeStepper scheme";
          error_message += " -- I think it's 2nd order...\n";

          oomph::OomphLibWarning(error_message,"MultiTimeStepper::order()",OOMPH_EXCEPTION_LOCATION);*/
      return 2;
    }

    unsigned nprev_values() const { return NSTEPS; }
    unsigned ndt() const { return NSTEPS; }

    virtual double weightBDF1(const unsigned &i, const unsigned &j) const { return WeightBDF1(i, j); }
    virtual double weightBDF2(const unsigned &i, const unsigned &j) const { return WeightBDF2(i, j); }
    virtual double weightNewmark2(const unsigned &i, const unsigned &j) const { return WeightNewmark2(i, j); }
    virtual void setNewmark2Coeffs(const double & p1,const double & p2) {NewmarkBeta1=p1;NewmarkBeta2=p2;}

    void shift_time_values(oomph::Data *const &data_pt);    // Push data_pt's value history back by one step and store the new Newmark2 (and, if adaptive, BDF2) velocity/acceleration
    void shift_time_positions(oomph::Node *const &node_pt); // Same as shift_time_values, but for a node's position history
    void set_weights();                                     // (Re-)compute WeightBDF1/WeightBDF2/WeightNewmark2/Weight from the current and previous timestep sizes

    void set_predictor_weights();                                         // Compute the AB2-like predictor weights used for adaptive-timestep error estimation
    void calculate_predicted_positions(oomph::Node *const &node_pt);      // Store the predicted (uncorrected) position at Predictor_storage_index
    void calculate_predicted_values(oomph::Data *const &data_pt);         // Store the predicted (uncorrected) value at Predictor_storage_index
    void set_error_weights();                                             // Compute the scalar Error_weight used to scale predictor-vs-corrector differences into a timestep error estimate
    double temporal_error_in_position(oomph::Node *const &node_pt, const unsigned &i);  // Estimated temporal error for position component i, based on the predictor/corrector difference
    double temporal_error_in_value(oomph::Data *const &data_pt, const unsigned &i);      // Estimated temporal error for value i, based on the predictor/corrector difference

    void assign_initial_values_impulsive(oomph::Data *const &data_pt); // Fill data_pt's entire time history with its current value and zero velocity/acceleration (impulsive start)
    void assign_initial_positions_impulsive(oomph::Node *const &node_pt); // Same as assign_initial_values_impulsive, but for a node's position history

    void set_num_unsteady_steps_done(unsigned n) { unsteady_steps_done_for_degrading = n; }
    void increment_num_unsteady_steps_done() { unsteady_steps_done_for_degrading++; }
    unsigned get_num_unsteady_steps_done() const { return unsteady_steps_done_for_degrading; }
  };
}
