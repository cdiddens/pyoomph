/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha

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

// Split-scheme ("refinement pattern") abstraction for adaptive mesh refinement.
//
// A RefinementPattern describes ONE way to subdivide a bulk element into son
// elements when it is flagged for h-refinement. Historically pyoomph had exactly
// one implicit scheme: isotropic subdivision into required_nsons() sons of the
// SAME element type (a quad -> 4 quads, a hex -> 8 hexes, a line -> 2 lines),
// hard-coded inside BulkElementBase::dynamic_split(). This header factors that
// decision out into an explicit object so the engine can later support
//   * heterogeneous offspring (e.g. a pyramid -> pyramids + tetrahedra, whose
//     refinement is not shape-closed), and
//   * multiple / anisotropic schemes for one element (e.g. splitting a quad into
//     two quads in either direction instead of four).
//
// This first landing only introduces the abstraction and reproduces the historical
// behaviour via IsotropicSameTypeRefinementPattern; it is a no-behaviour-change
// refactor. See dev_docs/mixed_adaptive_meshes.md for the full plan.

#pragma once

#include <string>

namespace pyoomph
{
  class BulkElementBase;

  // Base class for a split scheme. A pattern is stateless with respect to a
  // particular element: all queries take the parent element as an argument, so a
  // single pattern instance can be shared (as a singleton) across all elements
  // that use it.
  class RefinementPattern
  {
  public:
    virtual ~RefinementPattern() {}

    // Number of son elements produced when `parent` is split by this pattern.
    virtual unsigned nsons(const BulkElementBase *parent) const = 0;

    // Construct son number `ison` (0 <= ison < nsons(parent)) as a fresh, empty
    // element of the appropriate type. Its nodes/geometry are filled in later by
    // the oomph-lib build() machinery. Ownership of the returned element passes to
    // the caller (the tree/mesh).
    virtual BulkElementBase *construct_son(const BulkElementBase *parent, unsigned ison) const = 0;

    // Short, stable identifier for this scheme. Stored on refined elements so that
    // unrefinement (rebuild_from_sons) can invert the scheme that produced them.
    virtual std::string name() const = 0;
  };

  // The historical default scheme: N = parent->required_nsons() sons, each created
  // by parent->create_son_instance() (i.e. same element type as the parent). This
  // exactly reproduces the pre-refactor BulkElementBase::dynamic_split() behaviour
  // for lines/quads/hexes. Stateless -> exposed as a process-wide singleton.
  class IsotropicSameTypeRefinementPattern : public RefinementPattern
  {
  public:
    unsigned nsons(const BulkElementBase *parent) const override;
    BulkElementBase *construct_son(const BulkElementBase *parent, unsigned ison) const override;
    std::string name() const override { return "isotropic"; }

    static const IsotropicSameTypeRefinementPattern *instance();
  };
}
