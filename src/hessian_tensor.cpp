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


#include "hessian_tensor.hpp"
#include "exception.hpp"
namespace pyoomph
{

  // Allocates 'size' empty per-row maps (one per first index i); the tensor starts
  // completely empty and is filled up via accumulate().
  SparseRank3Tensor::SparseRank3Tensor(unsigned size, bool _symmetric) : symmetric(_symmetric), data(size), tens_size(-1)
  {
  }

  // Already calculate the indices for the CSR format to make a quick vector product
  // Walks the per-row (j,k)->value maps (which are already sorted lexicographically by (j,k),
  // see map_index_comp) and flattens them into a CSR-like structure of (i,j) "matrix" entries,
  // where each entry stores the list of (k,value) pairs contributing to that (i,j) slot.
  // This lets right_vector_mult() below avoid repeated map lookups on every call.
  std::tuple<std::vector<int>, std::vector<int>> SparseRank3Tensor::finalize_for_vector_product()
  {
    matrix_col_index.clear();
    matrix_row_start.clear();
    vector_prod_contribs.clear();
    matrix_row_start.push_back(0);
    size_t size_accu = 0;
    for (unsigned int i = 0; i < data.size(); i++)
    {
      int last_col = -1;
      for (auto &entry : data[i]) // entry.first => j,k   and  entry.second=contrib
      {
        // A new distinct j value starts a new (i,j) CSR entry; all k-contributions for the
        // same (i,j) are collected in vector_prod_contribs.back() until j changes again.
        if (entry.first.first > last_col)
        {
          matrix_col_index.push_back(entry.first.first);
          vector_prod_contribs.push_back(std::vector<std::pair<int, double>>());
          size_accu += sizeof(std::vector<std::pair<int, double>>);
          last_col = entry.first.first;
        }
        size_accu += sizeof(std::pair<int, float>);
        vector_prod_contribs.back().push_back(std::make_pair(entry.first.second, entry.second));
      }
      matrix_row_start.push_back(vector_prod_contribs.size());
    }
    tens_size = data.size();
    data.clear(); // Get some space, we desperatey need it!
    std::cout << "Hessian Tensor Storage size " << size_accu / (1024 * 1024) << " MB" << std::endl;
    return std::make_tuple(matrix_col_index, matrix_row_start);
  }

  // M_ij = T_ijk v_k
  /*
  The arrays V and COL_INDEX are of length NNZ, and contain the non-zero values and the column indices of those values respectively.
  The array ROW_INDEX is of length m + 1 and encodes the index in V and COL_INDEX where the given row starts. This is equivalent to ROW_INDEX[j] encoding the total number of nonzeros above row j. The last element is NNZ , i.e., the fictitious index in V immediately after the last valid index NNZ - 1
  */
  // Contracts the tensor with vector v over the last index, giving the entries (in the CSR
  // layout produced by finalize_for_vector_product()) of the matrix M_ij = sum_k T_ijk v_k.
  std::vector<double> SparseRank3Tensor::right_vector_mult(const std::vector<double> &v) // Returns a CSR matrix
  {
    if (this->size() != v.size())
      throw_runtime_error("Mismatch in tensor and vector size");
    if (matrix_row_start.empty())
      throw_runtime_error("Call finalize_for_vector_product first");
    std::vector<double> vals(matrix_row_start.back(), 0.0);

    if (!symmetric)
    {
      // General case: each flat CSR entry c (corresponding to one (i,j) pair) sums its
      // stored (k,value) contributions against v[k].
      for (int c = 0; c < matrix_row_start.back(); c++)
      {
        for (auto contrib : vector_prod_contribs[c])
        {
          vals[c] += v[contrib.first] * contrib.second;
        }
      }
    }
    else
    {
      // Symmetric case: only the j<=k half of each row was accumulated (see accumulate()),
      // so each stored (k,value) contribution is applied both to its own CSR entry and, via
      // contrib.first used as the "mirrored" index, to the counterpart entry that was skipped
      // during accumulation.
      for (int c = 0; c < matrix_row_start.back(); c++)
      {
        for (auto contrib : vector_prod_contribs[c])
        {
          vals[c] += v[contrib.first] * contrib.second;
          vals[contrib.first] += v[c] * contrib.second;
        }
      }
    }

    return vals;
  }

  // Dumps every stored nonzero entry as an explicit (i,j,k,value) tuple, e.g. for exporting
  // the tensor to Python or for debugging; not intended for performance-critical use.
  std::vector<std::tuple<int, int, int, double>> SparseRank3Tensor::get_entries() const
  {
    std::vector<std::tuple<int, int, int, double>> res;
    for (unsigned int i = 0; i < data.size(); i++)
    {
      const auto &d = data[i];
      for (const auto &me : d)
      {
        res.push_back(std::make_tuple(i, me.first.first, me.first.second, me.second));
      }
    }
    return res;
  }

}
