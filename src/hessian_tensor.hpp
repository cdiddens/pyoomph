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
#include <algorithm>
#include <vector>
namespace pyoomph
{

  // Sparse storage for a rank-3 tensor T_ijk, used to hold the Hessian of the residuals
  // with respect to the degrees of freedom, i.e. T_ijk = d^2 R_i / (du_j du_k).
  // This tensor is needed e.g. for Hessian-vector products in bifurcation tracking
  // (second-order derivatives of the residual enter the derivative of the Jacobian
  // with respect to the parameter/eigenvector). Storage is sparse in (j,k) per row i,
  // since only a small subset of dof pairs actually contribute to each residual entry.
  class SparseRank3Tensor
  {
  protected:
    // One stored contribution to T_i(j,k). Rows are kept as flat vectors rather than as ordered maps:
    // the tensor is written once, in a burst of ~10^6 accumulate() calls, and only afterwards read in
    // (j,k) order. A std::map paid a red-black-tree descent and a node allocation per contribution for
    // an ordering nothing needed until the end; append-then-sort pays neither
    // (dev_docs/code_generation.md 9.4.10).
    struct Rank3Entry
    {
      int j, k;
      double value;
    };
    static bool entry_less(const Rank3Entry &a, const Rank3Entry &b) { return a.j < b.j || (a.j == b.j && a.k < b.k); }

    bool symmetric;                          // Symmetric in the second&third index, like a Hessian
    // Mutable because consolidation is a representation detail: get_entries() is logically const but
    // must compact before it can report anything.
    mutable std::vector<std::vector<Rank3Entry>> data;  // [i] -> unordered (j,k,value) contributions
    mutable std::vector<size_t> compacted_size;         // [i] -> row length just after its last compaction

    // Sort row i by (j,k) and sum duplicate (j,k) contributions in place.
    void compact_row(unsigned i) const;
    // Compact every row, so that each is sorted by (j,k) with no duplicates.
    void compact_all() const;
    int tens_size;                                                           // Cached size (number of rows i); -1 until finalize_for_vector_product() has run, then equals data.size() before it was cleared
    // CSR-like storage built by finalize_for_vector_product(), used to evaluate right_vector_mult() quickly:
    // vector_prod_contribs[c] holds the (k,value) pairs for the column index matrix_col_index[c].
    std::vector<std::vector<std::pair<int, double>>> vector_prod_contribs;
    std::vector<int> matrix_col_index;
    std::vector<int> matrix_row_start;

  public:
    // Allocate a tensor with 'size' rows (i.e. i in [0,size) ). If _symmetric is true,
    // the tensor is assumed symmetric in its last two indices (j,k), so only the j<=k
    // half needs to be accumulated (see accumulate()).
    SparseRank3Tensor(unsigned size, bool _symmetric = false);

    // Number of rows (i.e. the size of the first index i, matching the number of residuals/dofs).
    unsigned size() const
    {
      if (tens_size < 0)
        return data.size();
      else
        return tens_size;
    }

    // Contracts the tensor with a vector over the last index: result_ij = sum_k T_ijk * v_k,
    // returned in the CSR layout established by finalize_for_vector_product(). Requires that
    // finalize_for_vector_product() has already been called.
    std::vector<double> right_vector_mult(const std::vector<double> &v); // Returns a CSR matrix

    // Add 'val' to entry T_ijk. If the tensor is symmetric and j>k, the contribution is skipped
    // here since it will be accounted for by the corresponding (i,k,j) call instead (the caller
    // is expected to only ever add either (j,k) or (k,j), never both, for symmetric tensors).
    void accumulate(const unsigned &i, const unsigned &j, const unsigned &k, const double &val)
    {
      if (symmetric && j > k)
      {
        // std::tuple<int,int> index={k,j};
        return; // Will be accumulated elsewise
      }
      else
      {
        // Append; duplicates are summed when the row is compacted. Compaction is triggered once a row
        // has doubled since it was last compacted, which bounds the memory held by duplicates at ~2x
        // the compacted size while keeping the amortised cost per contribution constant.
        std::vector<Rank3Entry> &row = data[i];
        row.push_back(Rank3Entry{(int)j, (int)k, val});
        if (row.size() >= 2 * compacted_size[i] + 64)
          compact_row(i);
      }
    }

    // Converts the map-based storage into a flat CSR-like layout (matrix_col_index/matrix_row_start
    // plus vector_prod_contribs) to allow fast repeated right_vector_mult() calls, and frees the
    // map storage ('data') afterwards to reduce memory footprint. Must be called once before any
    // right_vector_mult() call, and no further accumulate() calls are allowed afterwards.
    std::tuple<std::vector<int>, std::vector<int>> finalize_for_vector_product();

    // Returns all stored nonzero entries as (i,j,k,value) tuples, e.g. for exporting/debugging.
    std::vector<std::tuple<int, int, int, double>> get_entries() const;
  };

}
