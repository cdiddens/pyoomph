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


#include "kdtree.hpp"
#include "thirdparty/nanoflann.hpp"
#include <iostream>
#include "exception.hpp"

namespace pyoomph
{

  template <typename T>
  struct PointCloud
  {
    struct Point
    {
      T x, y, z;
    };

    using coord_t = T; //!< The type of each coordinate

    std::vector<Point> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate
    // value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
      if (dim == 0)
        return pts[idx].x;
      else if (dim == 1)
        return pts[idx].y;
      else
        return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard
    // bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned
    //   in "bb" so it can be avoided to redo it again. Look at bb.size() to
    //   find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const
    {
      return false;
    }
  };

  using num_t = double;
  using kd_tree_1d = nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>, PointCloud<num_t>, 1>;
  using kd_tree_2d = nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>, PointCloud<num_t>, 2>;
  using kd_tree_3d = nanoflann::KDTreeSingleIndexDynamicAdaptor<nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>, PointCloud<num_t>, 3>;

  using static_kd_tree_1d = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>, PointCloud<num_t>, 1>;
  using static_kd_tree_2d = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>, PointCloud<num_t>, 2>;
  using static_kd_tree_3d = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>, PointCloud<num_t>, 3>;

  class ImplementedKDTree
  {
  protected:
    friend class KDTree;
    const unsigned max_leaf = 10;
    PointCloud<num_t> cloud;

  public:
    PointCloud<num_t> &get_cloud() { return cloud; }
    ImplementedKDTree() {}
    ImplementedKDTree(ImplementedKDTree &old) : cloud(old.get_cloud()) {}
    ImplementedKDTree(std::vector<double> coords, unsigned dim)
    {
      cloud.pts.resize(coords.size() / dim);
      //  std::cout << "CREATEING CLOUD " << dim << "  " << coords.size() << std::endl;
      if (dim == 1)
      {
        for (size_t i = 0; i < coords.size(); i++)
        {
          cloud.pts[i].x = coords[i];
          cloud.pts[i].y = 0;
          cloud.pts[i].z = 0;
        }
      }
      else if (dim == 2)
      {
        for (size_t i = 0; i < coords.size(); i += 2)
        {
          cloud.pts[i / 2].x = coords[i];
          cloud.pts[i / 2].y = coords[i + 1];
          cloud.pts[i / 2].z = 0;
        }
      }
      else if (dim == 3)
      {
        for (size_t i = 0; i < coords.size(); i += 3)
        {
          cloud.pts[i / 3].x = coords[i];
          cloud.pts[i / 3].y = coords[i + 1];
          cloud.pts[i / 3].z = coords[i + 2];
        }
      }
    }
    virtual ~ImplementedKDTree() {}
    virtual void addIndex(unsigned index) = 0;
    virtual void addIndices(unsigned start, unsigned end) = 0;
    virtual int point_present(double x, double y, double z, double epsilon = 1e-8) = 0;
    virtual int nearest_point(double x, double y, double z, double *distret) = 0;
    virtual std::vector<std::pair<uint32_t, double>> radius_search(double radius, double x, double y, double z) = 0;
  };

  template <typename tree_t, int DIM>
  class DynamicImplementedKDTreeNDIM : public ImplementedKDTree
  {
  protected:
    tree_t tree;

  public:
    DynamicImplementedKDTreeNDIM<tree_t, DIM>() : tree(DIM, cloud, {max_leaf}) {}
    DynamicImplementedKDTreeNDIM<tree_t, DIM>(ImplementedKDTree &lower_dim) : ImplementedKDTree(lower_dim), tree(DIM, cloud, {max_leaf})
    {
      if (cloud.pts.size())
      {
        addIndices(0, cloud.pts.size() - 1);
      }
    }
    void addIndex(unsigned index) override { tree.addPoints(index, index); }
    void addIndices(unsigned start, unsigned end) override { tree.addPoints(start, end); }

    int point_present(double x, double y, double z, double epsilon = 1e-8) override
    {
      if (cloud.pts.empty())
        return -1;
      const size_t num_results = 1;
      size_t ret_index;
      num_t out_dist_sqr;
      nanoflann::KNNResultSet<num_t> resultSet(num_results);
      resultSet.init(&ret_index, &out_dist_sqr);
      num_t query_pt[3] = {x, y, z};
      tree.findNeighbors(resultSet, query_pt, {10});
      if (out_dist_sqr < epsilon * epsilon)
      {
        return ret_index;
      }
      else
      {
        return -1;
      }
    }

    int nearest_point(double x, double y, double z, double *distret) override
    {
      if (cloud.pts.empty())
        return -1;
      const size_t num_results = 1;
      size_t ret_index;
      num_t out_dist_sqr;
      nanoflann::KNNResultSet<num_t> resultSet(num_results);
      resultSet.init(&ret_index, &out_dist_sqr);
      num_t query_pt[3] = {x, y, z};
      tree.findNeighbors(resultSet, query_pt, {10});
      if (distret)
        *distret = sqrt(out_dist_sqr);
      return ret_index;
    }

    std::vector<std::pair<uint32_t, double>> radius_search(double radius, double x, double y, double z) override
    {
      throw_runtime_error("Cannot perform a radius_search on a dynamic KDTree");
      return std::vector<std::pair<uint32_t, double>>();
    }
  };

  template <typename tree_t, int DIM>
  class StaticImplementedKDTreeNDIM : public ImplementedKDTree
  {
  protected:
    tree_t tree;

  public:
    StaticImplementedKDTreeNDIM<tree_t, DIM>(std::vector<double> coordarray) : ImplementedKDTree(coordarray, DIM), tree(DIM, cloud, {max_leaf}) {}
    void addIndex(unsigned index) override { throw_runtime_error("Cannot add points to a static tree"); }
    void addIndices(unsigned start, unsigned end) override { throw_runtime_error("Cannot add points to a static tree"); }
    int point_present(double x, double y, double z, double epsilon = 1e-8) override
    {
      if (cloud.pts.empty())
        return -1;
      const size_t num_results = 1;
      size_t ret_index;
      num_t out_dist_sqr;
      nanoflann::KNNResultSet<num_t> resultSet(num_results);
      resultSet.init(&ret_index, &out_dist_sqr);
      num_t query_pt[3] = {x, y, z};
      tree.findNeighbors(resultSet, query_pt, {10});
      if (out_dist_sqr < epsilon * epsilon)
      {
        return ret_index;
      }
      else
      {
        return -1;
      }
    }

    int nearest_point(double x, double y, double z, double *distret) override
    {
      if (cloud.pts.empty())
        return -1;
      const size_t num_results = 1;
      size_t ret_index;
      num_t out_dist_sqr;
      nanoflann::KNNResultSet<num_t> resultSet(num_results);
      resultSet.init(&ret_index, &out_dist_sqr);
      num_t query_pt[3] = {x, y, z};
      tree.findNeighbors(resultSet, query_pt, {10});
      if (distret)
        *distret = sqrt(out_dist_sqr);
      return ret_index;
    }
    std::vector<std::pair<uint32_t, double>> radius_search(double radius, double x, double y, double z) override
    {
      std::vector<std::pair<uint32_t, double>> ret_matches;

      nanoflann::SearchParams params;
      // params.sorted = false;
      num_t query_pt[3] = {x, y, z};
      const size_t nMatches = this->tree.radiusSearch(&query_pt[0], radius * radius, ret_matches, params);
      for (size_t i = 0; i < nMatches; i++)
        ret_matches[i].second = sqrt(ret_matches[i].second);
      return ret_matches;
    }
  };

  typedef DynamicImplementedKDTreeNDIM<kd_tree_1d, 1> DynamicImplementedKDTree1d;
  typedef DynamicImplementedKDTreeNDIM<kd_tree_2d, 2> DynamicImplementedKDTree2d;
  typedef DynamicImplementedKDTreeNDIM<kd_tree_3d, 3> DynamicImplementedKDTree3d;

  typedef StaticImplementedKDTreeNDIM<static_kd_tree_1d, 1> StaticImplementedKDTree1d;
  typedef StaticImplementedKDTreeNDIM<static_kd_tree_2d, 2> StaticImplementedKDTree2d;
  typedef StaticImplementedKDTreeNDIM<static_kd_tree_3d, 3> StaticImplementedKDTree3d;

  KDTree::KDTree(unsigned _dim) : dim(_dim), static_tree(false), tree(NULL)
  {
    if (dim == 3)
      tree = new DynamicImplementedKDTree3d();
    else if (dim == 2)
      tree = new DynamicImplementedKDTree2d();
    else
      tree = new DynamicImplementedKDTree1d();
  };

  void KDTree::reset(unsigned _dim)
  {
    delete tree;
    if (_dim == 3)
      tree = new DynamicImplementedKDTree3d();
    else if (_dim == 2)
      tree = new DynamicImplementedKDTree2d();
    else
      tree = new DynamicImplementedKDTree1d();
  }

  KDTree::KDTree(std::vector<double> &coordarray, unsigned _dim) : dim(_dim), static_tree(true), tree(NULL)
  {
    if (dim == 3)
      tree = new StaticImplementedKDTree3d(coordarray);
    else if (dim == 2)
      tree = new StaticImplementedKDTree2d(coordarray);
    else
      tree = new StaticImplementedKDTree1d(coordarray);
    // std::cout << "CONSTRUCTED  STATIC TREE " << tree << " WITH DIM " << dim << std::endl;
  }

  KDTree::~KDTree()
  {
    delete tree;
  }

  unsigned KDTree::add_point(double x, double y, double z)
  {
    if (static_tree)
      throw_runtime_error("Cannot add a point to a static tree");
    // std::cout << "ADDING POINT " << x << "  "<< y<< "  " << z <<  "  CURRENT DIM " << dim << std::endl;
    if (z && dim < 3)
    {
      auto *oldtree = tree;
      tree = new DynamicImplementedKDTree3d(*oldtree);
      //   tree->cloud=oldtree->cloud;
      delete oldtree;
      dim = 3;
    }
    else if (y && dim < 2)
    {
      auto *oldtree = tree;
      // std::cout << "CHANING TREE " << std::endl;
      tree = new DynamicImplementedKDTree2d(*oldtree);
      // tree->cloud=oldtree->cloud;
      delete oldtree;
      dim = 2;
    }
    unsigned index = tree->cloud.pts.size();
    tree->cloud.pts.push_back({x, y, z});

    tree->addIndex(index);

    return index;
  }

  std::vector<double> KDTree::get_point_coordinate_by_index(unsigned index)
  {
    auto &p = tree->cloud.pts[index];
    if (dim == 1)
      return {p.x};
    else if (dim == 1)
      return {p.x, p.y};
    else if (dim == 2)
      return {p.x, p.y, p.z};
    return {};
  }

  int KDTree::point_present(double x, double y, double z, double epsilon)
  {
    if (dim < 3 && fabs(z) > epsilon)
      return -1;
    if (dim < 2 && fabs(y) > epsilon)
      return -1;
    return tree->point_present(x, y, z, epsilon);
  }

  int KDTree::nearest_point(double x, double y, double z, double *distret)
  {
    // std::cout << "CALLING NEAREST POINT ON " << tree << std::endl;
    return tree->nearest_point(x, y, z, distret);
  }

  unsigned KDTree::add_point_if_not_present(double x, double y, double z, double epsilon)
  {
    int res = point_present(x, y, z, epsilon);
    if (res >= 0)
      return res;
    else
      return add_point(x, y, z);
  }

  std::vector<std::pair<uint32_t, double>> KDTree::radius_search(double radius, double x, double y, double z)
  {
    return tree->radius_search(radius, x, y, z);
  }

}
