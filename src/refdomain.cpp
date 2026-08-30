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

#include "refdomain.hpp"

#include "Telements.h"
#include "wedges_and_pyramids.hpp"

#include <algorithm>

namespace pyoomph
{

  RefDomain reference_domain_kind(const oomph::FiniteElement *e)
  {
    if (dynamic_cast<const oomph::TElementBase *>(e))
      return RefDomain::Simplex;
    if (dynamic_cast<const oomph::QElementGeometricBase *>(e))
      return RefDomain::Box;
    if (dynamic_cast<const oomph::WedgeElementBase *>(e))
      return RefDomain::Prism;
    if (dynamic_cast<const oomph::PyramidElementBase *>(e))
      return RefDomain::Pyramid;
    return RefDomain::Unknown;
  }

  bool inside_reference_domain(RefDomain kind, const oomph::FiniteElement *e, unsigned element_dim,
                               const double *s, double tol)
  {
    switch (kind)
    {
    case RefDomain::Simplex:
    {
      double sum = 0.0;
      for (unsigned d = 0; d < element_dim; d++)
      {
        if (s[d] < -tol)
          return false;
        sum += s[d];
      }
      return sum <= 1.0 + tol;
    }
    case RefDomain::Box:
      for (unsigned d = 0; d < element_dim; d++)
        if (s[d] < e->s_min() - tol || s[d] > e->s_max() + tol)
          return false;
      return true;
    // Both of these are documented on the element classes in wedges_and_pyramids.hpp: the wedge is
    // a triangular prism, and the pyramid's cross-section shrinks with s2 ("s[0] and s[1] run from
    // 0 to 1-s[2]").
    case RefDomain::Prism:
      if (s[0] < -tol || s[1] < -tol || s[0] + s[1] > 1.0 + tol)
        return false;
      return s[2] >= e->s_min() - tol && s[2] <= e->s_max() + tol;
    case RefDomain::Pyramid:
    {
      if (s[2] < e->s_min() - tol || s[2] > e->s_max() + tol)
        return false;
      const double lim = 1.0 - s[2];
      return s[0] >= -tol && s[0] <= lim + tol && s[1] >= -tol && s[1] <= lim + tol;
    }
    default:
      return false;
    }
  }

  void clamp_to_reference_domain(RefDomain kind, const oomph::FiniteElement *e, unsigned element_dim,
                                 double *s)
  {
    switch (kind)
    {
    case RefDomain::Simplex:
    {
      for (unsigned d = 0; d < element_dim; d++)
        s[d] = std::max(0.0, s[d]);
      double sum = 0.0;
      for (unsigned d = 0; d < element_dim; d++)
        sum += s[d];
      if (sum > 1.0 && sum > 0.0)
        for (unsigned d = 0; d < element_dim; d++)
          s[d] /= sum;
      break;
    }
    case RefDomain::Box:
      for (unsigned d = 0; d < element_dim; d++)
        s[d] = std::min(std::max(s[d], e->s_min()), e->s_max());
      break;
    case RefDomain::Prism:
      s[2] = std::min(std::max(s[2], e->s_min()), e->s_max());
      s[0] = std::max(0.0, s[0]);
      s[1] = std::max(0.0, s[1]);
      if (s[0] + s[1] > 1.0)
      {
        const double sum = s[0] + s[1];
        s[0] /= sum;
        s[1] /= sum;
      }
      break;
    case RefDomain::Pyramid:
    {
      s[2] = std::min(std::max(s[2], e->s_min()), e->s_max());
      const double lim = 1.0 - s[2];
      s[0] = std::min(std::max(s[0], 0.0), std::max(lim, 0.0));
      s[1] = std::min(std::max(s[1], 0.0), std::max(lim, 0.0));
      break;
    }
    default:
      break;
    }
  }

}
