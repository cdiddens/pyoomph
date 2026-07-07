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


#pragma once
#include <map>
#include <vector>
namespace pyoomph
{

  class SparseRank3Tensor
  {
  protected:
    struct map_index_comp
    {
      bool operator()(const std::pair<int, int> &a, const std::pair<int, int> &b) const { return a.first < b.first || (a.first == b.first && a.second < b.second); }
    };

    bool symmetric;                                                          // Symmetric in the second&third index, like a Hessian
    std::vector<std::map<std::pair<int, int>, double, map_index_comp>> data; // [i](j,k)->value
    int tens_size;
    std::vector<std::vector<std::pair<int, double>>> vector_prod_contribs;
    std::vector<int> matrix_col_index;
    std::vector<int> matrix_row_start;

  public:
    SparseRank3Tensor(unsigned size, bool _symmetric = false);

    unsigned size() const
    {
      if (tens_size < 0)
        return data.size();
      else
        return tens_size;
    }

    std::vector<double> right_vector_mult(const std::vector<double> &v); // Returns a CSR matrix

    void accumulate(const unsigned &i, const unsigned &j, const unsigned &k, const double &val)
    {
      if (symmetric && j > k)
      {
        // std::tuple<int,int> index={k,j};
        return; // Will be accumulated elsewise
      }
      else
      {
        std::pair<int, int> index = {j, k};
        if (data[i].count(index))
          data[i][index] += val;
        else
          data[i][index] = val;
      }
    }

    std::tuple<std::vector<int>, std::vector<int>> finalize_for_vector_product();

    std::vector<std::tuple<int, int, int, double>> get_entries() const;
  };

}
