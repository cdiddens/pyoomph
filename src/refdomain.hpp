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

// "Is this local coordinate inside its element, and if not, where is the nearest point that is?"
//
// Every element family answers that differently, and two unrelated pieces of machinery need the
// answer: the point locator, when deciding whether a Newton inversion landed in the element it was
// tried against, and the tracer advection, when deciding whether a sub-step left the element. The
// predicates live here rather than in either of them so the two cannot drift apart - the tracers
// previously carried their own copy as a per-element-class
// factor_when_local_coordinate_becomes_invalid, which existed for 1d and 2d only and threw for
// every 3d element.
//
// Deliberately keyed on an explicitly passed RefDomain rather than re-running the dynamic_casts:
// the classification is stable for the lifetime of an element, and both callers already cache it
// per element. reference_domain_kind() is the one place that does the casts.

#pragma once

#include "oomph_lib.hpp"

namespace pyoomph
{
  // Which reference domain a local coordinate has to land in to count as inside.
  enum class RefDomain : unsigned char
  {
    Unknown = 0, // no containment test available - the caller must fall back to something else
    Simplex = 1, // s_i >= 0 and sum s_i <= 1
    Box = 2,     // s_min <= s_i <= s_max
    Prism = 3,   // wedge: s0,s1 >= 0, s0+s1 <= 1, s2 in [s_min,s_max]
    Pyramid = 4  // s2 in [s_min,s_max], s0 and s1 in [0, 1-s2]
  };

  // Classify one element. Unknown is not an error: it means no containment test is available, so a
  // caller has to decide for itself what to do (the locator falls back to a seeded Newton).
  RefDomain reference_domain_kind(const oomph::FiniteElement *e);

  // `tol` is in LOCAL coordinate units, which are O(1), so it is a fraction of an element.
  //
  // It must not be at machine epsilon. A point sitting exactly on an element's face - a tracer that
  // has just been handed over from the neighbour, a query on a node shared by two elements -
  // inverts to s = 1 + a few ulp as often as to s = 1 - a few ulp, and rejecting the first reports
  // the point as outside for a purely numerical reason.
  bool inside_reference_domain(RefDomain kind, const oomph::FiniteElement *e, unsigned element_dim,
                               const double *s, double tol);

  // Move s to a point of the reference domain. Not the exact Euclidean projection onto it, but it
  // lands inside, which is all a step limiter needs. No-op for RefDomain::Unknown.
  void clamp_to_reference_domain(RefDomain kind, const oomph::FiniteElement *e, unsigned element_dim,
                                 double *s);
}
