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


#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>

namespace pyoomph
{

  class ImplementedKDTree;
  class KDTree
  {
  protected:
    unsigned dim;
    bool static_tree;
    ImplementedKDTree *tree;

  public:
    KDTree(unsigned _dim = 1);                              // Create a dynamic tree
    KDTree(std::vector<double> &coordarray, unsigned _dim); // Create a static tree coordarray[line*dim+coordindex_of_line]
    virtual ~KDTree();
    void reset(unsigned _dim);
    unsigned add_point(double x, double y = 0.0, double z = 0.0);
    unsigned add_point_if_not_present(double x, double y = 0.0, double z = 0.0, double epsilon = 1e-8);
    int point_present(double x, double y = 0.0, double z = 0.0, double epsilon = 1e-8);
    int nearest_point(double x, double y = 0.0, double z = 0.0, double *distret = NULL);
    std::vector<double> get_point_coordinate_by_index(unsigned index);
    std::vector<std::pair<uint32_t, double>> radius_search(double radius, double x, double y = 0.0, double z = 0.0);
  };

}
