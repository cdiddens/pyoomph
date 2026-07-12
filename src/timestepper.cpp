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
 This file is strongly related to the file timestepper.cc from oomph-lib
 We merge all time steppers in a single class here, i.e. we store the weights of all different time steppers from oomph-lib
*/
#include "timestepper.hpp"

namespace pyoomph
{

	// (Re-)derive all weight matrices (BDF1, BDF2, Newmark2) from the current (dt) and
	// previous (dtprev) timestep sizes. Row indices into the Weight-style matrices are the
	// derivative order (0=value, 1=first derivative/velocity, 2=second derivative/accel),
	// column indices are the time-history slot (0=current, 1=previous step, 2=two steps ago,
	// NSTEPS+1/NSTEPS+2=stored Newmark2 velocity/acceleration slots, and in the adaptive
	// case NSTEPS+3/NSTEPS+4 the predictor velocity/offset slots - see the storage layout
	// comment on MultiTimeStepper in timestepper.hpp). Only the first-derivative row is
	// filled for BDF1/BDF2 (they are only used for value/velocity, not acceleration); the
	// actually-used `Weight` matrix (inherited from oomph::TimeStepper) is NOT updated here
	// - see the Newmark2 weights below, which populate WeightNewmark2 only.
	void MultiTimeStepper::set_weights()
	{
		double dt = Time_pt->dt(0);
		double dtprev = Time_pt->dt(1);

		// BDF2
		WeightBDF2(1, 0) = 1.0 / dt + 1.0 / (dt + dtprev);
		WeightBDF2(1, 1) = -(dt + dtprev) / (dt * dtprev);
		WeightBDF2(1, 2) = dt / ((dt + dtprev) * dtprev);
		WeightBDF2(1, 3) = 0.0;
		WeightBDF2(1, 4) = 0.0;
		if (adaptive_flag())
		{
			WeightBDF2(1, 5) = 0.0;
			WeightBDF2(1, 6) = 0.0;
		}

		// BDF1
		WeightBDF1(1, 0) = 1.0 / dt;
		WeightBDF1(1, 1) = -1.0 / dt;
		WeightBDF1(1, 2) = 0.0;
		WeightBDF1(1, 3) = 0.0;
		WeightBDF1(1, 4) = 0.0;
		if (adaptive_flag())
		{
			WeightBDF1(1, 5) = 0.0;
			WeightBDF1(1, 6) = 0.0;
		}

		// Newmark2
		WeightNewmark2(2, 0) = 2.0 / (NewmarkBeta2 * dt * dt);
		WeightNewmark2(2, 1) = -2.0 / (NewmarkBeta2 * dt * dt);
		for (unsigned t = 2; t <= NSTEPS; t++)
		{
			WeightNewmark2(2, t) = 0.0;
		}
		WeightNewmark2(2, NSTEPS + 1) = -2.0 / (dt * NewmarkBeta2);
		WeightNewmark2(2, NSTEPS + 2) = (NewmarkBeta2 - 1.0) / NewmarkBeta2;

		WeightNewmark2(1, 0) = NewmarkBeta1 * dt * WeightNewmark2(2, 0);
		WeightNewmark2(1, 1) = NewmarkBeta1 * dt * WeightNewmark2(2, 1);
		for (unsigned t = 2; t <= NSTEPS; t++)
		{
			WeightNewmark2(1, t) = 0.0;
		}
		WeightNewmark2(1, NSTEPS + 1) = 1.0 + NewmarkBeta1 * dt * WeightNewmark2(2, NSTEPS + 1);
		WeightNewmark2(1, NSTEPS + 2) = dt * (1.0 - NewmarkBeta1) + NewmarkBeta1 * dt * WeightNewmark2(2, NSTEPS + 2);
	}

	//========================================================================
	// Compute the weights for an explicit second-order (Adams-Bashforth-like) predictor of
	// the value at the new time, used only in the adaptive case to obtain an independent
	// estimate to compare the (implicit, corrector) Newmark2 result against.
	void MultiTimeStepper::set_predictor_weights()
	{
		if (adaptive_flag())
		{
			double dt = Time_pt->dt(0);
			double dtprev = Time_pt->dt(1);
			Predictor_weight[0] = 0.0;
			Predictor_weight[1] = 1.0 - (dt * dt) / (dtprev * dtprev);
			Predictor_weight[2] = (dt * dt) / (dtprev * dtprev);
			Predictor_weight[3] = (1.0 + dt / dtprev) * dt;
		}
	}

	// Evaluate the predictor (see set_predictor_weights) for every non-copy value of data_pt
	// and store it at Predictor_storage_index, to be compared against the corrected value
	// later in temporal_error_in_value.
	void MultiTimeStepper::calculate_predicted_values(oomph::Data *const &data_pt)
	{
		if (adaptive_flag())
		{
			unsigned n_value = data_pt->nvalue();
			for (unsigned j = 0; j < n_value; j++)
			{
				if (data_pt->is_a_copy(j) == false)
				{
					double predicted_value = data_pt->value(NSTEPS + 3, j) * Predictor_weight[3];
					for (unsigned i = 1; i < 3; i++)
					{
						predicted_value += data_pt->value(i, j) * Predictor_weight[i];
					}
					data_pt->set_value(Predictor_storage_index, j, predicted_value);
				}
			}
		}
	}

	// Same as calculate_predicted_values, but for a node's position coordinates.
	void MultiTimeStepper::calculate_predicted_positions(oomph::Node *const &node_pt)
	{
		if (adaptive_flag())
		{
			unsigned n_dim = node_pt->ndim();
			for (unsigned j = 0; j < n_dim; j++)
			{
				if (node_pt->position_is_a_copy(j) == false)
				{
					double predicted_value = node_pt->x(NSTEPS + 3, j) * Predictor_weight[3];
					for (unsigned i = 1; i < 3; i++)
					{
						predicted_value += node_pt->x(i, j) * Predictor_weight[i];
					}
					node_pt->x(Predictor_storage_index, j) = predicted_value;
				}
			}
		}
	}

	// Compute the scalar factor that converts a predictor/corrector difference into a local
	// temporal error estimate, following the standard variable-step predictor-corrector
	// error formula (depends only on the ratio of previous to current timestep size).
	void MultiTimeStepper::set_error_weights()
	{
		if (adaptive_flag())
		{
			double dt = Time_pt->dt(0);
			double dtprev = Time_pt->dt(1);
			Error_weight = pow((1.0 + dtprev / dt), 2.0) / (1.0 + 3.0 * (dtprev / dt) + 4.0 * pow((dtprev / dt), 2.0) + 2.0 * pow((dtprev / dt), 3.0));
		}
	}

	// Return the estimated local temporal error for position component i of node_pt: the
	// (scaled) difference between the actually-computed value and the explicit predictor's
	// value stored at Predictor_storage_index. Returns 0 when not running adaptively.
	double MultiTimeStepper::temporal_error_in_position(oomph::Node *const &node_pt, const unsigned &i)
	{
		if (adaptive_flag())
		{
			return Error_weight * (node_pt->x(i) - node_pt->x(Predictor_storage_index, i));
		}
		else
		{
			return 0.0;
		}
	}

	// Same as temporal_error_in_position, but for a Data value.
	double MultiTimeStepper::temporal_error_in_value(oomph::Data *const &data_pt, const unsigned &i)
	{
		if (adaptive_flag())
		{
			return Error_weight * (data_pt->value(i) - data_pt->value(Predictor_storage_index, i));
		}
		else
		{
			return 0.0;
		}
	}

	// "Impulsive start": fill all history slots of every non-copy value of data_pt with its
	// current value (i.e. assume it was constant in the past) and zero the Newmark2
	// velocity/acceleration slots (and, if adaptive, the predictor slots).
	void MultiTimeStepper::assign_initial_values_impulsive(oomph::Data *const &data_pt)
	{
			unsigned n_value = data_pt->nvalue();  
			for (unsigned j = 0; j < n_value; j++)
			{
				if (data_pt->is_a_copy(j) == false)
				{
					for (unsigned t = 1; t <= NSTEPS; t++)
					{
						data_pt->set_value(t, j, data_pt->value(j));
					}
					// Newmark velo and accel
					data_pt->set_value(NSTEPS + 1, j, 0.0);
					data_pt->set_value(NSTEPS + 2, j, 0.0);
					if (adaptive_flag())
					{
						// Adaptive velocity and prediction
						data_pt->set_value(NSTEPS + 3, j, 0.0);
						data_pt->set_value(NSTEPS + 4, j, data_pt->value(j));
					}
				}
			}
	}
	// Same as assign_initial_values_impulsive, but for a node's (possibly higher-order,
	// i.e. Hermite-type) position history across all position types and dimensions.
	void MultiTimeStepper::assign_initial_positions_impulsive(oomph::Node *const &node_pt)
	{
		unsigned n_dim = node_pt->ndim();
		unsigned n_position_type = node_pt->nposition_type();    
		for (unsigned i = 0; i < n_dim; i++)
		{
			if (node_pt->position_is_a_copy(i) == false)
			{     
				for (unsigned k = 0; k < n_position_type; k++)
				{       
					for (unsigned t = 1; t <= NSTEPS; t++)
					{
						node_pt->x_gen(t, k, i) = node_pt->x_gen(k, i);
					}         
					// Newmark velo and accel
					node_pt->x_gen(NSTEPS + 1, k, i) = 0.0;
					node_pt->x_gen(NSTEPS + 2, k, i) = 0.0;

					if (adaptive_flag())
					{
							// Predicted velocity and offset
							node_pt->x_gen(NSTEPS + 3, k, i) = 0.0;
							node_pt->x_gen(NSTEPS + 4, k, i) = node_pt->x_gen(k, i);
					}
				}
			}
		}
	}

	// Called once per completed timestep to advance data_pt's time history: computes the
	// Newmark2 velocity/acceleration (and, if adaptive, the BDF2 velocity used for error
	// estimation at the *next* step) from the about-to-be-shifted history using the current
	// weights, then shifts every non-copy value back by one slot (t <- t-1) and stores the
	// freshly computed derivative values in their dedicated slots.
	void MultiTimeStepper::shift_time_values(oomph::Data *const &data_pt)
	{
		// BDF2 part
		unsigned n_value = data_pt->nvalue();
		double velocityBDF2[n_value];
		double velocityNewmark2[n_value];
		double accelNewmark2[n_value];
		const unsigned nt_value = this->ntstorage(); // TODO: This correct?
		if (adaptive_flag())
		{
			for (unsigned i = 0; i < n_value; i++)
			{
				velocityBDF2[i] = 0.0;
				for (unsigned t = 0; t < nt_value; t++)
				{
					velocityBDF2[i] += WeightBDF2(1, t) * data_pt->value(t, i);
				}
			}
		}
		for (unsigned i = 0; i < n_value; i++)
		{
			velocityNewmark2[i] = 0.0;
			accelNewmark2[i] = 0.0;
			for (unsigned t = 0; t < nt_value; t++)
			{
				velocityNewmark2[i] += WeightNewmark2(1, t) * data_pt->value(t, i);
				accelNewmark2[i] += WeightNewmark2(2, t) * data_pt->value(t, i);
			}
			//     std::cout << "NEWMARK BELO UNST: " << unsteady_steps_done_for_degrading << "   " << velocityNewmark2[0] << "  " << accelNewmark2[0] << std::endl;
		}
		for (unsigned j = 0; j < n_value; j++)
		{
			if (data_pt->is_a_copy(j) == false)
			{
				for (unsigned t = NSTEPS; t > 0; t--)
				{
					data_pt->set_value(t, j, data_pt->value(t - 1, j));
				}
				data_pt->set_value(NSTEPS + 1, j, velocityNewmark2[j]);
				data_pt->set_value(NSTEPS + 2, j, accelNewmark2[j]);
				if (adaptive_flag())
				{
					data_pt->set_value(Predictor_storage_index - 1, j, velocityBDF2[j]); // This correct?
				}
			}
		}
	}

	// Same as shift_time_values, but for a node's generalized position history (x_gen,
	// indexed by time slot, position type and spatial dimension).
	void MultiTimeStepper::shift_time_positions(oomph::Node *const &node_pt)
	{
		// BDF2 part
		unsigned n_dim = node_pt->ndim();
		unsigned n_position_type = node_pt->nposition_type();
		unsigned n_tstorage = ntstorage();
		double velocityBDF2[n_position_type][n_dim];
		double velocityNewmark2[n_position_type][n_dim];
		double accelNewmark2[n_position_type][n_dim];
		if (adaptive_flag())
		{
			for (unsigned i = 0; i < n_dim; i++)
			{
				for (unsigned k = 0; k < n_position_type; k++)
				{
					velocityBDF2[k][i] = 0.0;
					for (unsigned t = 0; t < n_tstorage; t++)
					{
						velocityBDF2[k][i] += WeightBDF2(1, t) * node_pt->x_gen(t, k, i);
					}
				}
			}
		}
		for (unsigned i = 0; i < n_dim; i++)
		{
			for (unsigned k = 0; k < n_position_type; k++)
			{
				velocityNewmark2[k][i] = 0.0;
				accelNewmark2[k][i] = 0.0;
				for (unsigned t = 0; t < n_tstorage; t++)
				{
					velocityNewmark2[k][i] += WeightNewmark2(1, t) * node_pt->x_gen(t, k, i);
					accelNewmark2[k][i] += WeightNewmark2(2, t) * node_pt->x_gen(t, k, i);
				}
			}
		}
		for (unsigned i = 0; i < n_dim; i++)
		{
			if (node_pt->position_is_a_copy(i) == false)
			{
				for (unsigned k = 0; k < n_position_type; k++)
				{
					for (unsigned t = NSTEPS; t > 0; t--)
					{
						node_pt->x_gen(t, k, i) = node_pt->x_gen(t - 1, k, i);
					}
					node_pt->x_gen(NSTEPS + 1, k, i) = velocityNewmark2[k][i];
					node_pt->x_gen(NSTEPS + 2, k, i) = accelNewmark2[k][i];
					if (adaptive_flag())
					{
						node_pt->x_gen(Predictor_storage_index - 1, k, i) = velocityBDF2[k][i];
					}
				}
			}
		}
	}

}
