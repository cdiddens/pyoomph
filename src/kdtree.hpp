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


// Thin public-facing wrapper around a nanoflann-based k-d tree (see kdtree.cpp for the
// actual nanoflann glue code in ImplementedKDTree and its Dynamic/Static subclasses).
// Supports 1d/2d/3d point clouds, either "dynamic" (points can be added incrementally,
// used e.g. while incrementally building a mesh's spatial index) or "static" (built once
// from a fixed coordinate array, which additionally enables radius_search - nanoflann's
// dynamic adaptor does not support radius queries).

#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>

namespace pyoomph
{

  class ImplementedKDTree;
  // Public k-d tree handle. Delegates all actual work to an ImplementedKDTree of the
  // appropriate dimension (1d/2d/3d) and kind (dynamic/static), stored in `tree`. A
  // dynamic tree can grow its dimension on the fly (see add_point in kdtree.cpp): it
  // starts as 1d and is transparently rebuilt as 2d/3d once a nonzero y/z coordinate is
  // added.
  class KDTree
  {
  protected:
    unsigned dim;
    bool static_tree;
    ImplementedKDTree *tree;
    // Index of the first point that is in the point cloud but not yet in the index, or -1 when
    // there is none. See add_point_deferred(): indexing points one at a time makes nanoflann's
    // dynamic adaptor rebuild a sub-index per call, which is the dominant cost when a caller knows
    // up front that it is about to add hundreds of thousands of points and will not query in
    // between. Every read path flushes first, so the deferral is invisible from outside.
    long first_deferred_point = -1;

  public:
    KDTree(unsigned _dim = 1);                              // Create a dynamic tree
    KDTree(std::vector<double> &coordarray, unsigned _dim); // Create a static tree coordarray[line*dim+coordindex_of_line]
    virtual ~KDTree();
    void reset(unsigned _dim);                                                                          // Discard all points and start over as an empty dynamic tree of dimension _dim
    unsigned add_point(double x, double y = 0.0, double z = 0.0);                                        // Add a point to a dynamic tree, returning its index; upgrades dim if y/z is nonzero
    unsigned add_point_deferred(double x, double y = 0.0, double z = 0.0);                               // Like add_point, but leaves the point unindexed until the next query (or index_deferred_points()); returns the same index add_point would have
    void index_deferred_points();                                                                        // Index everything added by add_point_deferred() in one go; called automatically by every query
    unsigned add_point_if_not_present(double x, double y = 0.0, double z = 0.0, double epsilon = 1e-8);  // Like add_point, but reuses an existing point within epsilon if one exists
    int point_present(double x, double y = 0.0, double z = 0.0, double epsilon = 1e-8);                  // Return the index of a point within epsilon of (x,y,z), or -1 if none
    int nearest_point(double x, double y = 0.0, double z = 0.0, double *distret = NULL);                 // Return the index of the nearest point, optionally the (Euclidean) distance in *distret
    std::vector<double> get_point_coordinate_by_index(unsigned index);
    std::vector<std::pair<uint32_t, double>> radius_search(double radius, double x, double y = 0.0, double z = 0.0); // All points within `radius`, as (index, distance) pairs; static trees only
    // The k nearest points, nearest first, appended to `indices`; returns how many were found
    // (fewer than k if the tree is smaller). Static trees only. Preferred over radius_search when
    // the right radius is not known in advance, since a too-large one returns most of the tree.
    unsigned k_nearest(unsigned k, double x, double y, double z, std::vector<uint32_t> &indices);
  };

}
