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

  class MultiTimeStepper : public oomph::TimeStepper
  {
  protected:
    static const unsigned NSTEPS = 2;
    static const unsigned NWEIGHT = 2 + 3; // For non-adaptive case: Same as Newmark
    static const unsigned MAXDERIV = 2;

    double NewmarkBeta1;
    double NewmarkBeta2;

    oomph::Vector<double> Predictor_weight;
    double Error_weight;

    unsigned unsteady_steps_done_for_degrading; // How many unsteady steps have been done (required for degrading)

    oomph::DenseMatrix<double> WeightBDF1, WeightBDF2, WeightNewmark2; //,WeightBDF12;
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

    void shift_time_values(oomph::Data *const &data_pt);
    void shift_time_positions(oomph::Node *const &node_pt);
    void set_weights();

    void set_predictor_weights();
    void calculate_predicted_positions(oomph::Node *const &node_pt);
    void calculate_predicted_values(oomph::Data *const &data_pt);
    void set_error_weights();
    double temporal_error_in_position(oomph::Node *const &node_pt, const unsigned &i);
    double temporal_error_in_value(oomph::Data *const &data_pt, const unsigned &i);

    void assign_initial_values_impulsive(oomph::Data *const &data_pt); //{}
    void assign_initial_positions_impulsive(oomph::Node *const &node_pt); //{}

    void set_num_unsteady_steps_done(unsigned n) { unsteady_steps_done_for_degrading = n; }
    void increment_num_unsteady_steps_done() { unsteady_steps_done_for_degrading++; }
    unsigned get_num_unsteady_steps_done() const { return unsteady_steps_done_for_degrading; }
  };
}
