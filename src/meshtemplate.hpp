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

#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "exception.hpp"
#include "problem.hpp"
#include "nodes.hpp"
#include "kdtree.hpp"

#include "oomph_lib.hpp"
#include "Tmacroelements.hpp"
#include <vector>
#include <map>
#include <set>
#include <functional>
#include <algorithm>

namespace pyoomph
{

  // Forward declaration: MeshTemplate is the top-level object (defined below)
  // that owns all nodes/facets/element-collections of a mesh description.
  class MeshTemplate;

  class BulkElementBase;
  // Index type used to refer to nodes within a MeshTemplate's node vector
  // (rather than storing raw pointers everywhere, elements/facets just store
  // indices into MeshTemplate::nodes).
  typedef std::size_t nodeindex_t;

  // Mesh templates share a list of nodes
  // They have information of subdomains and added elements

  class MeshTemplateElementCollection;

  // A single node of a mesh template, i.e. the geometric description of a
  // mesh point before the actual oomph-lib Node object is created.
  // Stores the (up to 3d) coordinates, which boundaries it lies on, whether
  // it sits on a curved facet (and therefore needs special treatment when
  // interpolating its position), and (for periodic meshes) the index of its
  // periodic master node (-1 if none).
  class MeshTemplateNode
  {
  public:
    double x, y, z;
    nodeindex_t index;
    Node *oomph_node; // Pointer to the actual oomph-lib node once it has been built (NULL before that)
    int periodic_master; // Index of the periodic master node, or -1 if this node is not periodic
    //	std::vector<nodeindex_t> periodic_master_of;
    bool on_curved_facet;
    std::set<unsigned int> on_boundaries;
    std::set<MeshTemplateElementCollection *> part_of_domain;
    MeshTemplateNode(double _x, double _y, double _z) : x(_x), y(_y), z(_z), oomph_node(NULL), periodic_master(-1), on_curved_facet(false) {}
    MeshTemplateNode(double _x, double _y) : x(_x), y(_y), z(0.0), oomph_node(NULL), periodic_master(-1), on_curved_facet(false) {}
    MeshTemplateNode(double _x) : x(_x), y(0.0), z(0.0), oomph_node(NULL), periodic_master(-1), on_curved_facet(false) {}
  };

  // Abstract base class describing a curved geometric entity (e.g. a circle
  // arc, cylinder mantle, sphere patch or spline) that a mesh facet can be
  // attached to. It provides the mapping between an intrinsic parametric
  // coordinate (of dimension `dim`, e.g. angle for a circle) and the actual
  // Eulerian position, in both directions, so that mid-side/boundary nodes
  // can be placed exactly on the curved geometry rather than on the
  // polygonal/polyhedral approximation implied by the straight-sided facet.
  class MeshTemplateCurvedEntity
  {
  protected:
    unsigned dim; // Dimension of the parametric coordinate (1 for curves, 2 for surfaces)
    // Helper used by get_information_string() to serialize a coordinate
    // vector as "<size> <v0> <v1> ..." for later reloading via load_from_strings().
    virtual void write_vector_information(const std::vector<double> v, std::ostream &os)
    {
      os << v.size();
      for (unsigned int i = 0; i < v.size(); i++)
      {
        os << "\t" << v[i];
      }
      os << std::endl;
    }

  public:
    MeshTemplateCurvedEntity(unsigned d) : dim(d) {}
    // Factory: reconstructs a set of curved entities (keyed by an integer id)
    // from their serialized string representation (as written by
    // get_information_string()), starting at line `currline` of `s`.
    static std::map<int, MeshTemplateCurvedEntity *> load_from_strings(const std::vector<std::string> &s, size_t &currline);
    virtual unsigned get_parametric_dimension() const { return dim; }
    // Map a parametric coordinate to the Eulerian position at time level t (t=0 is current).
    virtual void parametric_to_position(const unsigned &t, const std::vector<double> &parametric, std::vector<double> &position) { throw_runtime_error("Empty parametric_to_position called"); }
    // Inverse of parametric_to_position: find the parametric coordinate of a given Eulerian position.
    virtual void position_to_parametric(const unsigned &t, const std::vector<double> &position, std::vector<double> &parametric) { throw_runtime_error("Empty position_to_parametric called"); };
    // Given the parametric coordinates of two points that should be connected along the
    // curve, adjust them (in place) to resolve the periodic wrap-around ambiguity, e.g. for
    // an angle parametrization where +pi and -pi refer to the same point.
    virtual void apply_periodicity(std::vector<std::vector<double>> &parametric){};
    // Serialize the entity's defining geometric data to a string (used for caching/reloading meshes).
    virtual std::string get_information_string() { throw_runtime_error("Please implement get_information_string"); }
  };

  // A circular arc in 2d/3d, defined by its center and two points (start/end)
  // lying on the circle. The parametric coordinate is the polar angle around the center.
  class CurvedEntityCircleArc : public MeshTemplateCurvedEntity
  {
  protected:
    std::vector<double> center, startpt, endpt;
    double radius;

  public:
    CurvedEntityCircleArc(const std::vector<double> &_center, const std::vector<double> &_startpt, const std::vector<double> &_endpt) : MeshTemplateCurvedEntity(1), center(_center), startpt(_startpt), endpt(_endpt)
    {
      radius = 0;
      for (unsigned int i = 0; i < std::min(startpt.size(), center.size()); i++)
        radius += (startpt[i] - center[i]) * (startpt[i] - center[i]);
      radius = sqrt(radius);
    }
    virtual void parametric_to_position(const unsigned &t, const std::vector<double> &parametric, std::vector<double> &position)
    {
      position = center;
      position[0] += radius * cos(parametric[0]);
      position[1] += radius * sin(parametric[0]);
    }
    virtual void position_to_parametric(const unsigned &t, const std::vector<double> &position, std::vector<double> &parametric)
    {
      parametric[0] = atan2(position[1] - center[1], position[0] - center[0]);
    };
    virtual void apply_periodicity(std::vector<std::vector<double>> &parametric)
    {
      if (fabs(parametric[0][0] - parametric[1][0]) > M_PI)
      {
        if (fabs(parametric[0][0]) > fabs(parametric[1][0]))
        {
          if (parametric[0][0] > 0)
          {
            parametric[0][0] = -M_PI + (parametric[0][0] - M_PI);
          }
          else
          {
            parametric[0][0] = M_PI - (parametric[0][0] + M_PI);
          }
        }
        else
        {
          std::ostringstream oss;
          oss << parametric[0][0] / M_PI << "  " << parametric[1][0] / M_PI << std::endl;
          throw_runtime_error("Handle periodic case here: " + oss.str());
        }
      }
    };
    virtual std::string get_information_string()
    {
      std::ostringstream oss;
      oss << radius << std::endl;
      write_vector_information(center, oss);
      write_vector_information(startpt, oss);
      write_vector_information(endpt, oss);
      return oss.str();
    }
  };

  // An arc on the mantle of a cylinder, defined by the cylinder's axis
  // (implicitly, via center/start/end points) and radius. The parametric
  // coordinate is (angle around the axis, position along the axis).
  class CurvedEntityCylinderArc : public MeshTemplateCurvedEntity
  {
  protected:
    std::vector<double> center, startpt, endpt, normal, ds, de, ta, ct;
    double radius;

  public:
    CurvedEntityCylinderArc(const std::vector<double> &_center, const std::vector<double> &_startpt, const std::vector<double> &_endpt) : MeshTemplateCurvedEntity(2), center(_center), startpt(_startpt), endpt(_endpt)
    {
      radius = 0;
      ds.resize(center.size());
      de.resize(center.size());
      for (unsigned int i = 0; i < std::min(startpt.size(), center.size()); i++)
      {
        ds[i] = startpt[i] - center[i];
        de[i] = endpt[i] - center[i];
        radius += ds[i] * ds[i];
      }
      radius = sqrt(radius);
      ta = ds;
      ta[0] /= radius; // Tangent vector (towards the mantle in direction of start-center)
      ta[1] /= radius;
      ta[2] /= radius;
      normal.resize(center.size());
      normal[0] = ds[1] * de[2] - ds[2] * de[1]; // Normal: Along the axis
      normal[1] = ds[2] * de[0] - ds[0] * de[2];
      normal[2] = ds[0] * de[1] - ds[1] * de[0];
      double nl = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
      normal[0] /= nl;
      normal[1] /= nl;
      normal[2] /= nl;
      // Get the cotangent from the cross product
      ct.resize(3);
      ct[0] = normal[1] * ta[2] - normal[2] * ta[1];
      ct[1] = normal[2] * ta[0] - normal[0] * ta[2];
      ct[2] = normal[0] * ta[1] - normal[1] * ta[0];
      nl = sqrt(ct[0] * ct[0] + ct[1] * ct[1] + ct[2] * ct[2]);
      ct[0] /= nl;
      ct[1] /= nl;
      ct[2] /= nl;
      /*
      double check1=0;
      double check2=0;
      double check3=0;
      for (unsigned int i=0;i<3;i++)
      {
       check1+=ct[i]*ta[i];
       check2+=normal[i]*ta[i];
       check3+=normal[i]*ct[i];
      }
      std::cout << "CHECK " << check1 <<"  " << check2 << "  " << check3 << std::endl;
      throw_runtime_error("Bllla");
         std::cout << "NORMAL "  << normal[0] << " , " << normal[1] << " , " << normal[2] << std::endl;
      std::cout << "TANG "  << ta[0] << " , " << ta[1] << " , " << ta[2] << std::endl;
      std::cout << "COT "  << ct[0] << " , " << ct[1] << " , " << ct[2] << std::endl;
         throw_runtime_error("Bllla");
      */
    }
    virtual void parametric_to_position(const unsigned &t, const std::vector<double> &parametric, std::vector<double> &position)
    {
      position = center;
      for (unsigned int i = 0; i < 3; i++)
      {
        position[i] += normal[i] * parametric[1] + radius * (cos(parametric[0]) * ta[i] + sin(parametric[0]) * ct[i]);
      }
      std::cout << " CYL PARAM TO POS " << parametric[0] << "  " << parametric[1] << "  leads to " << position[0] << "  " << position[1] << "  " << position[2] << std::endl;
    }
    virtual void position_to_parametric(const unsigned &t, const std::vector<double> &position, std::vector<double> &parametric)
    {
      parametric[1] = 0.0;
      double x = 0.0, y = 0.0;
      for (unsigned int i = 0; i < 3; i++)
      {
        double delta = position[i] - center[i];
        parametric[1] += delta * normal[i]; // Project on normal for the parametric value here
        x += delta * ta[i];
        y += delta * ct[i];
      }
      parametric[0] = atan2(y, x);
      std::cout << " CYL POS TO PARAM " << position[0] << "  " << position[1] << "  " << position[2] << "  leads to x,y= " << x << " " << y << " parametric " << parametric[0] << " , " << parametric[1] << std::endl;
    };
    virtual void apply_periodicity(std::vector<std::vector<double>> &parametric)
    {
      if (fabs(parametric[0][0] - parametric[1][0]) > M_PI)
      {
        throw_runtime_error("Handle periodic case here");
      }
    };
  };

  // A patch on a sphere (with less than 90 deg opening angle), defined by
  // the sphere's center, a point on the sphere marking the pole/normal
  // direction of the patch, and a tangent direction. The parametric
  // coordinate is (polar angle theta, azimuthal angle phi) in the local
  // tangent/cotangent/normal frame constructed in the constructor.
  class CurvedEntitySpherePart : public MeshTemplateCurvedEntity
  {
  protected:
    std::vector<double> center, normal, tangent, cotangent;
    double radius;

  public:
    CurvedEntitySpherePart(const std::vector<double> &_center, const std::vector<double> &_onsphere_center, const std::vector<double> &_tangent) : MeshTemplateCurvedEntity(2), center(_center), normal(_onsphere_center), cotangent(_tangent)
    {
      radius = 0;
      for (unsigned int i = 0; i < 3; i++)
        radius += (normal[i] - center[i]) * (normal[i] - center[i]);
      radius = sqrt(radius);
      for (unsigned int i = 0; i < 3; i++)
        normal[i] = (normal[i] - center[i]) / radius;
      double tdot = 0.0;
      for (unsigned int i = 0; i < 3; i++)
        tdot += cotangent[i] * cotangent[i];
      tdot = sqrt(tdot);
      for (unsigned int i = 0; i < 3; i++)
        cotangent[i] /= tdot;
      tdot = 0.0;
      for (unsigned int i = 0; i < 3; i++)
        tdot += normal[i] * cotangent[i];
      if (fabs(tdot) > 1 - 1e-7)
      {
        throw_runtime_error("CurvedEntitySpherePart tangent and normal (almost) coinciding... n=[" + std::to_string(normal[0]) + ", " + std::to_string(normal[1]) + "," + std::to_string(normal[2]) + "]  and  t=[" + std::to_string(cotangent[0]) + ", " + std::to_string(cotangent[1]) + "," + std::to_string(cotangent[2]) + "]");
      }
      tangent.resize(3);
      tangent[0] = normal[1] * cotangent[2] - normal[2] * cotangent[1];
      tangent[1] = normal[2] * cotangent[0] - normal[0] * cotangent[2];
      tangent[2] = normal[0] * cotangent[1] - normal[1] * cotangent[0];
      tdot = 0.0;
      for (unsigned int i = 0; i < 3; i++)
        tdot += tangent[i] * tangent[i];
      tdot = sqrt(tdot);
      for (unsigned int i = 0; i < 3; i++)
        tangent[i] /= tdot;
      cotangent[0] = (normal[1] * tangent[2] - normal[2] * tangent[1]);
      cotangent[1] = (normal[2] * tangent[0] - normal[0] * tangent[2]);
      cotangent[2] = (normal[0] * tangent[1] - normal[1] * tangent[0]);
      tdot = 0.0;
      for (unsigned int i = 0; i < 3; i++)
        tdot += cotangent[i] * cotangent[i];
      tdot = sqrt(tdot);
      for (unsigned int i = 0; i < 3; i++)
        cotangent[i] /= tdot;

      std::cout << "NORM TANG COTANG" << std::endl;
      for (unsigned int i = 0; i < 3; i++)
        std::cout << normal[i] << "  " << tangent[i] << "  " << cotangent[i] << std::endl;
    }
    virtual void parametric_to_position(const unsigned &t, const std::vector<double> &parametric, std::vector<double> &position)
    {
      position = center;
      double theta = parametric[0];
      double phi = parametric[1];
      double x = radius * cos(phi) * sin(theta);
      double y = radius * sin(phi) * sin(theta);
      double z = radius * cos(theta);
      for (unsigned int i = 0; i < 3; i++)
      {
        position[i] += x * tangent[i] + y * cotangent[i] + z * normal[i];
      }
    }
    virtual void position_to_parametric(const unsigned &t, const std::vector<double> &position, std::vector<double> &parametric)
    {
      std::vector<double> rel = position;
      for (unsigned int i = 0; i < 3; i++)
        rel[i] -= center[i];
      double dot = 0.0;
      for (unsigned int i = 0; i < 3; i++)
        dot += rel[i] * rel[i];
      dot = sqrt(dot);
      for (unsigned int i = 0; i < 3; i++)
        rel[i] /= dot;
      double z = 0.0;
      double x = 0.0;
      double y = 0.0;
      for (unsigned int i = 0; i < 3; i++)
      {
        x += tangent[i] * rel[i];
        y += cotangent[i] * rel[i];
        z += normal[i] * rel[i];
      }
      parametric[0] = acos(z);
      parametric[1] = atan2(y, x);

      std::vector<double> testpos(3);
      this->parametric_to_position(t, parametric, testpos);
      for (unsigned int i = 0; i < 3; i++)
      {
        std::cout << "TEST FOR pos->par->pos " << i << "  " << position[i] << " vs " << testpos[i] << std::endl;
      }
    };
    virtual void apply_periodicity(std::vector<std::vector<double>> &parametric){
        // TODO:; SHoukld not be required
    };
  };

  // A curve interpolating a sequence of control points `pts` with a
  // Catmull-Rom spline. The parametric coordinate is the spline parameter t
  // (arclength-ish, one segment per pair of consecutive control points).
  // Since Catmull-Rom splines have no closed-form inverse, position_to_parametric
  // works by sampling the spline (see gen_samples()) and searching/refining
  // from the nearest sample.
  class CurvedEntityCatmullRomSpline : public MeshTemplateCurvedEntity
  {
  protected:
    std::vector<std::vector<double>> pts;
    std::vector<double> samples;
    std::vector<std::vector<double>> samplepos;
    unsigned N;
    // Precompute `num` samples of the spline position, used to seed the
    // (numerical) inversion in position_to_parametric.
    void gen_samples(unsigned num);

  public:
    // Evaluate the spline position at parameter t.
    virtual void interpolate(double t, std::vector<double> &pos);
    // Evaluate the spline's tangent (derivative w.r.t. t) at parameter t.
    virtual void dinterpolate(double t, std::vector<double> &dpos);
    CurvedEntityCatmullRomSpline(const std::vector<std::vector<double>> &_pts);
    virtual void parametric_to_position(const unsigned &t, const std::vector<double> &parametric, std::vector<double> &position);
    virtual void position_to_parametric(const unsigned &t, const std::vector<double> &position, std::vector<double> &parametric);
    virtual std::string get_information_string();
  };

  // A facet (edge in 2d, face in 3d) of the mesh template, i.e. the boundary
  // of a bulk element expressed via the indices of its corner (and possibly
  // mid-side) nodes. Facets are the entities that get attached to curved
  // geometry (curved_entity) and to mesh boundaries (on_boundaries), and are
  // deduplicated (see MeshTemplate::facetmap) since the same facet is shared
  // by up to two adjacent bulk elements.
  class MeshTemplateFacet
  {
  public:
    MeshTemplateFacet(const  std::vector<nodeindex_t> &inds, MeshTemplateCurvedEntity *curved, std::vector<MeshTemplateNode *> *nodes);
    std::vector<nodeindex_t> sorted_inds; // For fast finding
    std::vector<nodeindex_t> nodeinds;
    MeshTemplateCurvedEntity *curved_entity;
    std::vector<std::vector<double>> parametrics;
    std::set<unsigned int> on_boundaries;
  };

  class MeshTemplateDomain;
  class MeshTemplateElement;

  // Base class gluing pyoomph's curved-facet description to oomph-lib's
  // MacroElement machinery. An oomph-lib MacroElement maps a reference cube/
  // square/triangle/etc. to a curved physical region by blending its facet
  // parametrizations; this base stores, for each local facet of the element,
  // the corresponding MeshTemplateFacet plus a node permutation (`permutation`)
  // that maps the macro element's canonical facet-node ordering onto the
  // facet's own node ordering (set up in set_facet()/find_permutation()).
  // Concrete subclasses (below) implement macro_element_boundary() for a
  // specific reference-element shape/order by evaluating the appropriate
  // blending function.
  class MeshTemplateMacroElementBase
  {
  protected:
    // Determine how the local node ordering of `new_facet` must be permuted
    // to align with the canonical ordering expected for macro-element facet
    // `ifacet`, using `for_orientation` (the element's own facet) as reference.
    virtual std::vector<unsigned> find_permutation(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation) = 0;

  public:
    // Attach `new_facet` as the macro element's local facet number `ifacet`,
    // computing and storing the required node permutation.
    void set_facet(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation);
    std::vector<MeshTemplateFacet *> facets;
    std::vector<std::vector<unsigned>> permutation;
    std::vector<std::vector<pyoomph::Node *>> default_facet_nodes;
    MeshTemplateMacroElementBase(MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes);
    // Evaluate the macro-element boundary mapping: for local coordinate `s`
    // on the i_direct-th "direction" of the reference element at history
    // time level t, return the corresponding global position f (called by
    // oomph-lib's macro-element blending during mesh construction/refinement).
    virtual void macro_element_boundary(const unsigned &t, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f) = 0;
  };

  // Macro element for 2d, quadrilateral (Q-type), second-order (9-node) elements.
  class MeshTemplateQMacroElement2 : public oomph::QMacroElement<2>, public MeshTemplateMacroElementBase
  {
  protected:
    std::vector<unsigned> find_permutation(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation);

  public:
    MeshTemplateQMacroElement2(MeshTemplateDomain *domain, unsigned index, MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes);
    virtual void macro_element_boundary(const unsigned &t, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f);
  };

  // Macro element for 2d, triangular (T-type), second-order (6-node) elements.
  class MeshTemplateTMacroElement2 : public oomph::TMacroElement<2>, public MeshTemplateMacroElementBase
  {
  protected:
    std::vector<unsigned> find_permutation(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation);

  public:
    MeshTemplateTMacroElement2(MeshTemplateDomain *domain, unsigned index, MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes);
    virtual void macro_element_boundary(const unsigned &t, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f);
  };

  // Macro element for 3d, brick (Q-type), second-order (27-node) elements.
  class MeshTemplateQMacroElement3 : public oomph::QMacroElement<3>, public MeshTemplateMacroElementBase
  {
  protected:
    std::vector<unsigned> find_permutation(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation);

  public:
    MeshTemplateQMacroElement3(MeshTemplateDomain *domain, unsigned index, MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes);
    virtual void macro_element_boundary(const unsigned &t, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f);
  };

  // oomph-lib Domain implementation that simply forwards macro_element_boundary()
  // calls to the individual MeshTemplateMacroElementBase-derived macro elements
  // that were pushed onto Macro_element_pt (one per curved bulk element).
  class MeshTemplateDomain : public oomph::Domain
  {
  public:
    MeshTemplateDomain();
    void push_back_macro_element(oomph::MacroElement *macro) { Macro_element_pt.push_back(macro); }
    void macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f);
  };

  // Abstract base class describing the geometry/topology of a single bulk
  // element in a mesh template (as read from e.g. a GMSH file or built up
  // from Python), independent of which oomph-lib element class it will
  // eventually be turned into. Each concrete subclass corresponds to one
  // geometric element shape/order (point, line, quad, tri, brick, tetra,
  // wedge, pyramid; each in C1 (linear/corner-node-only), C2 (quadratic)
  // and, for triangles/tetrahedra, "TB" (with additional centroid/face
  // "bubble" nodes) variants). Subclasses expose, for each of these nodal
  // spaces, how many nodes it has and which of the element's node_indices
  // belong to that space, so that BulkElementBase::factory_element can build
  // the correct oomph-lib element and node layout.
  class MeshTemplateElement
  {
  protected:
    int geometric_type; // We use here the same as in GMSH
    std::vector<nodeindex_t> node_indices;

  public:
    int get_geometric_type_index() const { return geometric_type; }
    const std::vector<nodeindex_t> &get_node_indices() const { return node_indices; }
    virtual unsigned int get_nnode_C1() const = 0;
    virtual unsigned int get_node_index_C1(const unsigned int &i) const = 0;
    virtual unsigned int get_nnode_C2() const = 0;
    virtual unsigned int get_node_index_C2(const unsigned int &i) const = 0;
    virtual unsigned int get_nnode_C1TB() const {return 0;}
    virtual unsigned int get_node_index_C1TB(const unsigned int &i) const {return node_indices[i];}
    virtual unsigned int get_nnode_C2TB() const {return 0;}
    virtual unsigned int get_node_index_C2TB(const unsigned int &i) const {return node_indices[i];}    
    virtual unsigned int nodal_dimension() const = 0;
    virtual MeshTemplateElement *convert_for_C2_space(MeshTemplate *templ) { return NULL; }
    virtual MeshTemplateElement *convert_for_C1TB_space(MeshTemplate *templ) { return NULL; }    
    MeshTemplateElement(int geomtyp) : geometric_type(geomtyp) {}
    virtual ~MeshTemplateElement() = default;
    virtual unsigned nfacets() { return 0; }
    virtual MeshTemplateFacet *construct_facet(unsigned i)
    {
      throw_runtime_error("Cannot costruct facets for this element");
      return NULL;
    }
    virtual void link_nodes_with_domain(MeshTemplateElementCollection *dom);
  };

  // A 0d point element (single node), used e.g. for point boundaries/probes.
  class MeshTemplateElementPoint : public MeshTemplateElement
  {
  public:
    MeshTemplateElementPoint(const nodeindex_t &n1);
    unsigned int get_nnode_C1() const { return 1; }
    unsigned int get_node_index_C1(const unsigned int &i) const { return 0; }
    unsigned int get_nnode_C2() const { return 1; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return 0; }
    unsigned int nodal_dimension() const { return 0; }
    virtual unsigned nfacets() { return 0; }
  };


  // 1d linear (2-node) line element.
  class MeshTemplateElementLineC1 : public MeshTemplateElement
  {
  public:
    MeshTemplateElementLineC1(const nodeindex_t &n1, const nodeindex_t &n2);
    unsigned int get_nnode_C1() const { return 2; }
    unsigned int get_node_index_C1(const unsigned int &i) const { return i; }
    unsigned int get_nnode_C2() const { return 0; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return -1; }
    unsigned int nodal_dimension() const { return 1; }
    virtual MeshTemplateElement *convert_for_C2_space(MeshTemplate *templ);
    virtual unsigned nfacets() { return 2; }
    virtual MeshTemplateFacet *construct_facet(unsigned i);
  };

  // 1d quadratic (3-node, with a mid-side node) line element.
  class MeshTemplateElementLineC2 : public MeshTemplateElement
  {
  public:
    MeshTemplateElementLineC2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3);
    unsigned int get_nnode_C1() const { return 2; }
    unsigned int get_node_index_C1(const unsigned int &i) const { return (i == 0 ? 0 : 2); }
    unsigned int get_nnode_C2() const { return 3; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return i; }
    unsigned int nodal_dimension() const { return 1; }
    virtual unsigned nfacets() { return 2; }
    virtual MeshTemplateFacet *construct_facet(unsigned i);
    //	virtual MeshTemplateElement * convert_for_C2_space(MeshTemplate *templ);
  };

  // 2d bilinear (4-node, corners only) quadrilateral element.
  class MeshTemplateElementQuadC1 : public MeshTemplateElement
  {
  protected:
  public:
    MeshTemplateElementQuadC1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4);
    unsigned int get_nnode_C1() const { return 4; }
    unsigned int get_node_index_C1(const unsigned int &i) const { return node_indices[i]; }
    unsigned int get_nnode_C2() const { return 0; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return -1; }
    unsigned int nodal_dimension() const { return 2; }
    virtual MeshTemplateElement *convert_for_C2_space(MeshTemplate *templ);
    virtual unsigned nfacets() { return 4; }
    virtual MeshTemplateFacet *construct_facet(unsigned i);
  };

  // 2d biquadratic (9-node: 4 corners, 4 mid-sides, 1 center) quadrilateral element.
  class MeshTemplateElementQuadC2 : public MeshTemplateElement
  {
  protected:
  public:
    MeshTemplateElementQuadC2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4,
                              const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7, const nodeindex_t &n8, const nodeindex_t &n9);
    unsigned int get_nnode_C1() const { return 4; }
    unsigned int get_node_index_C1(const unsigned int &i) const { return node_indices[i]; }
    unsigned int get_nnode_C2() const { return 9; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return node_indices[i]; }
    unsigned int nodal_dimension() const { return 2; }
    virtual unsigned nfacets() { return 4; }
    virtual MeshTemplateFacet *construct_facet(unsigned i);
  };

  // 2d linear (3-node, corners only) triangular element.
  class MeshTemplateElementTriC1 : public MeshTemplateElement
  {
  protected:
  public:
    MeshTemplateElementTriC1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3);
    unsigned int get_nnode_C1() const { return 3; }
    unsigned int get_node_index_C1(const unsigned int &i) const { return node_indices[i]; }
    unsigned int get_nnode_C2() const { return 0; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return -1; }
    unsigned int nodal_dimension() const { return 2; }
    virtual MeshTemplateElement *convert_for_C1TB_space(MeshTemplate *templ);    
    virtual MeshTemplateElement *convert_for_C2_space(MeshTemplate *templ);
    virtual unsigned nfacets() { return 3; }
    virtual MeshTemplateFacet *construct_facet(unsigned i);
  };

  // 2d quadratic (6-node: 3 corners + 3 mid-sides) triangular element.
  class MeshTemplateElementTriC2 : public MeshTemplateElement
  {
  protected:
  public:
    MeshTemplateElementTriC2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5, const nodeindex_t &n6);
    unsigned int get_nnode_C1() const { return 3; }
    unsigned int get_node_index_C1(const unsigned int &i) const { return node_indices[i]; }
    unsigned int get_nnode_C2() const { return 6; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return node_indices[i]; }
    unsigned int nodal_dimension() const { return 2; }
    virtual unsigned nfacets() { return 3; }
    virtual MeshTemplateElement *convert_for_C2TB_space(MeshTemplate *templ);
    virtual MeshTemplateFacet *construct_facet(unsigned i);
  };

  // Linear triangle enriched with a centroid "bubble" node (4 nodes total:
  // 3 corners + 1 center), used for e.g. Taylor-Hood/Crouzeix-Raviart-type
  // discretizations that need an extra internal degree of freedom.
  class MeshTemplateElementTriC1TB : public MeshTemplateElementTriC1
  {
  protected:
  public:
    MeshTemplateElementTriC1TB(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4);
    unsigned int get_nnode_C1TB() const { return 4; }
    unsigned int get_node_index_C1TB(const unsigned int &i) const { return node_indices[i]; }
  };


  // Quadratic triangle enriched with a centroid "bubble" node (7 nodes
  // total: 3 corners + 3 mid-sides + 1 center).
  class MeshTemplateElementTriC2TB : public MeshTemplateElementTriC2
  {
  protected:
  public:
    MeshTemplateElementTriC2TB(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7);
    unsigned int get_nnode_C2TB() const { return 7; }
    unsigned int get_nnode_C1TB() const { return 4; }    
    unsigned int get_node_index_C2TB(const unsigned int &i) const { return node_indices[i]; }
    unsigned int get_node_index_C1TB(const unsigned int &i) const { return (i<3 ? node_indices[i] : node_indices[6]); }    
  };

  // 3d trilinear (8-node, corners only) brick/hexahedron element.
  class MeshTemplateElementBrickC1 : public MeshTemplateElement
  {
  protected:
  public:
    MeshTemplateElementBrickC1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4,
                               const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7, const nodeindex_t &n8);
    unsigned int get_nnode_C1() const { return 8; }
    unsigned int get_node_index_C1(const unsigned int &i) const { return node_indices[i]; }
    unsigned int get_nnode_C2() const { return 0; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return -1; }
    unsigned int nodal_dimension() const { return 3; }
    virtual unsigned nfacets() { return 6; }
    virtual MeshTemplateElement *convert_for_C2_space(MeshTemplate *templ);
    virtual MeshTemplateFacet *construct_facet(unsigned i);
  };

  // 3d triquadratic (27-node: 8 corners + 12 edge mid-points + 6 face
  // centers + 1 body center) brick/hexahedron element.
  class MeshTemplateElementBrickC2 : public MeshTemplateElement
  {
  protected:
  public:
    MeshTemplateElementBrickC2(std::vector<nodeindex_t> ninds);
    unsigned int get_nnode_C1() const { return 8; }
    unsigned int get_node_index_C1(const unsigned int &i) const { return node_indices[i]; }
    unsigned int get_nnode_C2() const { return 27; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return node_indices[i]; }
    unsigned int nodal_dimension() const { return 3; }
    virtual unsigned nfacets() { return 6; }
    virtual MeshTemplateFacet *construct_facet(unsigned i);
  };

  // 3d linear (4-node, corners only) tetrahedral element.
  class MeshTemplateElementTetraC1 : public MeshTemplateElement
  {
  protected:
  public:
    MeshTemplateElementTetraC1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4);
    unsigned int get_nnode_C1() const { return 4; }
    unsigned int get_node_index_C1(const unsigned int &i) const { return node_indices[i]; }
    unsigned int get_nnode_C2() const { return 0; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return -1; }
    unsigned int nodal_dimension() const { return 3; }
    virtual unsigned nfacets() { return 4; }
    virtual MeshTemplateElement *convert_for_C2_space(MeshTemplate *templ);
    virtual MeshTemplateFacet *construct_facet(unsigned i);
    virtual MeshTemplateElement *convert_for_C1TB_space(MeshTemplate *templ);
  };
  
 
  // Linear tetrahedron enriched with a centroid "bubble" node (5 nodes total).
  class MeshTemplateElementTetraC1TB : public MeshTemplateElementTetraC1
  {
  protected:
  public:
    MeshTemplateElementTetraC1TB(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5);
    unsigned int get_nnode_C1TB() const { return 5; }
    unsigned int get_node_index_C1TB(const unsigned int &i) const { return i; }
  };


  // 3d quadratic (10-node: 4 corners + 6 edge mid-points) tetrahedral element.
  class MeshTemplateElementTetraC2 : public MeshTemplateElement
  {
  protected:
  public:
    MeshTemplateElementTetraC2(std::vector<nodeindex_t> ninds);
    unsigned int get_nnode_C1() const { return 4; }
    unsigned int get_node_index_C1(const unsigned int &i) const { return node_indices[i]; }
    unsigned int get_nnode_C2() const { return 10; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return node_indices[i]; }
    unsigned int nodal_dimension() const { return 3; }
    virtual unsigned nfacets() { return 4; }
    virtual MeshTemplateFacet *construct_facet(unsigned i);
    virtual MeshTemplateElement *convert_for_C2TB_space(MeshTemplate *templ);
  };

  // Quadratic tetrahedron enriched with 4 face-centroid "bubble" nodes plus
  // 1 body-centroid node (15 nodes total: 10 + 4 faces + 1 body).
  class MeshTemplateElementTetraC2TB : public MeshTemplateElementTetraC2
  {
  protected:
  public:
    MeshTemplateElementTetraC2TB(std::vector<nodeindex_t> ninds);
    unsigned int get_nnode_C2TB() const { return 15; }
    unsigned int get_node_index_C2TB(const unsigned int &i) const { return node_indices[i]; }
  };

  // 3d linear (6-node) wedge/prism element (triangular cross-section extruded
  // linearly); see also src/wedges_and_pyramids.cpp for its shape functions.
  class MeshTemplateElementWedgeC1 : public MeshTemplateElement
  {
  protected:
  public:    
    MeshTemplateElementWedgeC1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5, const nodeindex_t &n6);
    unsigned int get_nnode_C1() const { return 6; } // It has 6 nodes in total
    unsigned int get_node_index_C1(const unsigned int &i) const { return i; } // First order nodes are the same as the 6 nodes of the C1 element
    unsigned int get_nnode_C2() const { return 0; } // No second order nodes for the C1 element
    unsigned int get_node_index_C2(const unsigned int &i) const { return -1; } // No second order nodes for the C1 element, all indices map to -1
    unsigned int nodal_dimension() const { return 3; } // Wedge is a 3D element
    virtual unsigned nfacets() { return 5; } // How many facets does a wedge have? 2 triangles and 3 quadrilaterals, so 5 in total
    virtual MeshTemplateFacet *construct_facet(unsigned i); // Construct the facet for the i-th facet of the wedge
    virtual MeshTemplateElement *convert_for_C2_space(MeshTemplate *templ); // Convert this C1 wedge element to a C2 wedge element
  };

  // 3d quadratic (18-node) wedge/prism element.
  class MeshTemplateElementWedgeC2 : public MeshTemplateElement
  {
  protected:
  public:
    MeshTemplateElementWedgeC2(std::vector<nodeindex_t> ninds);
    unsigned int get_nnode_C1() const { return 6; }
    unsigned int get_node_index_C1(const unsigned int &i) const { throw_runtime_error("TODO"); return -1; }
    unsigned int get_nnode_C2() const { return 18; }
    unsigned int get_node_index_C2(const unsigned int &i) const { return i; }
    unsigned int nodal_dimension() const { return 3; }
    virtual unsigned nfacets() { return 5; }
    virtual MeshTemplateFacet *construct_facet(unsigned i);
  };

  // 3d linear (5-node) pyramid element (quadrilateral base + apex); see also
  // src/wedges_and_pyramids.cpp for its shape functions.
  class MeshTemplateElementPyramidC1 : public MeshTemplateElement
  {
  protected:
  public:    
    MeshTemplateElementPyramidC1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5);
    unsigned int get_nnode_C1() const { return 5; } // It has 5 nodes in total
    unsigned int get_node_index_C1(const unsigned int &i) const { return i; } // First order nodes are the same as the 5 nodes of the C1 element
    unsigned int get_nnode_C2() const { return 0; } // No second order nodes for the C1 element
    unsigned int get_node_index_C2(const unsigned int &i) const { return -1; } // No second order nodes for the C1 element, all indices map to -1
    unsigned int nodal_dimension() const { return 3; } // Pyramid is a 3D element
    virtual unsigned nfacets() { return 5; } // How many facets does a pyramid have? 
    virtual MeshTemplateFacet *construct_facet(unsigned i); // Construct the facet for the i-th facet of the pyramid
    virtual MeshTemplateElement *convert_for_C2_space(MeshTemplate *templ); // Convert this C1 wedge element to a C2 wedge element. Leave it out for now
  };

  class MeshTemplateElementPyramidC2 : public MeshTemplateElement
  {
  protected:
  public:    
    MeshTemplateElementPyramidC2(std::vector<nodeindex_t> ninds);
    unsigned int get_nnode_C1() const { return 5; }
    unsigned int get_node_index_C1(const unsigned int &i) const { throw_runtime_error("TODO"); return -1; }
    unsigned int get_nnode_C2() const { return 14; } 
    unsigned int get_node_index_C2(const unsigned int &i) const { return i; }
    unsigned int nodal_dimension() const { return 3; }
    virtual unsigned nfacets() { return 5; } 
    virtual MeshTemplateFacet *construct_facet(unsigned i); 
  };

  // A named group ("domain") of MeshTemplateElement objects that all share
  // the same generated oomph-lib element code (code_instance), e.g. "bulk"
  // vs. named interface/boundary domains. This is the level at which the
  // Python side attaches an element implementation (via set_element_code())
  // to a set of geometric elements, and at which the nodal/Lagrangian
  // dimension of the resulting oomph-lib elements is decided.
  class MeshTemplateElementCollection
  {
  protected:
    friend class MeshTemplate;
    MeshTemplate *mesh_template;
    std::string name;
    std::vector<MeshTemplateElement *> elements;
    DynamicBulkElementInstance *code_instance;
    int Nodal_dimension = -1;
    int Lagr_dimension = -1;
    int dim = -1;


  public:
    bool all_nodes_as_boundary_nodes=false;
    // Return a representative reference position (e.g. for setting initial
    // conditions or Dirichlet boundary conditions that depend on position)
    // for the sub-region of this collection selected by `boundindices`.
    virtual std::vector<double> get_reference_position_for_IC_and_DBC(std::set<unsigned int> boundindices);
    virtual int get_element_dimension() const { return dim; }
    virtual int nodal_dimension();
    void set_nodal_dimension(int d) { Nodal_dimension = d; }
    virtual int lagrangian_dimension();
    void set_lagrangian_dimension(int d) { Lagr_dimension = d; }
    MeshTemplateElementCollection(MeshTemplate *t, std::string n) : mesh_template(t), name(n) {}
    void add_point_element(const nodeindex_t &n1);
    void add_line_1d_C1(const nodeindex_t &n1, const nodeindex_t &n2);
    void add_line_1d_C2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3);
    void add_quad_2d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4);
    void add_quad_2d_C2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4,
                                              const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7, const nodeindex_t &n8, const nodeindex_t &n9);
    void add_tri_2d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3);
    void add_SV_tri_2d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3);    
    void add_tri_2d_C2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5, const nodeindex_t &n6);

    void add_brick_3d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4,
                                                const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7, const nodeindex_t &n8);
    void add_brick_3d_C2(const std::vector<nodeindex_t> &inds);
    void add_tetra_3d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4);
    void add_tetra_3d_C2(const std::vector<nodeindex_t> &inds);
    void add_wedge_3d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5, const nodeindex_t &n6);
    void add_wedge_3d_C2(const std::vector<nodeindex_t> &inds);
    void add_pyramid_3d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5);
    void add_pyramid_3d_C2(const std::vector<nodeindex_t> &inds);

    const std::vector<MeshTemplateElement *> &get_elements() const { return elements; }
    std::vector<std::string> get_adjacent_boundary_names();
    void set_element_code(DynamicBulkElementInstance *code_inst);
    //	void set_element_class(BaseFiniteElementCode & cls);
    MeshTemplate *get_template() { return mesh_template; }
    virtual ~MeshTemplateElementCollection();
    void set_all_nodes_as_boundary_nodes() {all_nodes_as_boundary_nodes=true;}

    //  pyoomph::Mesh * get_oomph_mesh() { if (!oomph_mesh) throw_runtime_error("Mesh not yet created. Do a MeshTemplate::finalise_creation() first"); return oomph_mesh;}
  };

  // Records that node `myself` is an intermediate (e.g. mid-side) node
  // created between the (sorted, for order-independent lookup) set of
  // `parent_node_ids`. Used by link_periodic_nodes() to find/match the
  // corresponding intermediate node on the periodic partner side, since
  // such nodes aren't explicitly paired up when periodicity is declared
  // between corner/vertex nodes only.
  class MeshTemplatePeriodicIntermediateNodeInfo
  {
  public:
    nodeindex_t myself;
    std::vector<nodeindex_t> parent_node_ids;
    MeshTemplatePeriodicIntermediateNodeInfo(nodeindex_t m, const std::vector<nodeindex_t> &pids) : myself(m), parent_node_ids(pids) { std::sort(parent_node_ids.begin(), parent_node_ids.end()); }
  };

  // Top-level, oomph-lib-independent description of a mesh: a shared pool of
  // nodes (deduplicated via `nodemap`/`kdtree`), a shared pool of facets
  // (deduplicated via `facetmap`), named mesh boundaries, and one or more
  // MeshTemplateElementCollection "domains" grouping elements that share an
  // element code. This is what gets built up from Python (e.g. by loading a
  // GMSH file or via direct calls) and is later consumed by
  // BulkElementBase::factory_element / MeshTemplateElementCollection to
  // construct the actual oomph-lib Mesh/Node/Element objects (see mesh.cpp).
  class MeshTemplate
  {
  protected:
    friend class MeshTemplateElementCollection;
    Problem *problem;
    int dim;
    std::vector<MeshTemplateNode *> nodes;
    KDTree kdtree;
    std::map<MeshTemplateNode *, nodeindex_t, std::function<bool(const MeshTemplateNode *, const MeshTemplateNode *)>> nodemap; // Required for fast finding unique nodes
    std::vector<MeshTemplateElementCollection *> bulk_element_collections;
    //   std::vector<MeshTemplateElementCollection*> interface_element_collections;
    std::vector<std::string> boundary_names;
    std::vector<MeshTemplateFacet *> facets;
    std::map<MeshTemplateFacet *, unsigned, std::function<bool(const MeshTemplateFacet *, const MeshTemplateFacet *)>> facetmap; // Required for fast finding facets
    MeshTemplateDomain *domain;
    std::vector<MeshTemplatePeriodicIntermediateNodeInfo> inter_nodes_periodic;

  public:
    MeshTemplate();
    // Clear all nodes/facets/collections, e.g. before rebuilding the template from scratch.
    virtual void reset();
    void _set_problem(Problem *p) { problem = p; }
    virtual ~MeshTemplate();
    // Detach/discard the oomph-lib Node objects built from this template
    // (oomph_node pointers), without discarding the geometric description itself.
    void flush_oomph_nodes();
    // Unconditionally append a new node at the given position and return its index.
    nodeindex_t add_node(double x, double y = 0.0, double z = 0.0);
    nodeindex_t add_node_unique(double x, double y = 0.0, double z = 0.0); // Checks if node exists
    // Add (or find an existing) node exactly half-way between nodes n1,n2 (edge mid-point).
    nodeindex_t add_intermediate_node_unique(const nodeindex_t &n1, const nodeindex_t &n2);
    nodeindex_t add_intermediate_node_unique(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, bool boundary_possible); // For tri C2TB
    nodeindex_t add_intermediate_node_unique(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, bool boundary_possible);

    // Create (or reuse, via facetmap) the facet spanned by `vertexindices`
    // and attach it to the given curved geometry entity.
    MeshTemplateFacet * add_facet_to_curve_entity(const std::vector<nodeindex_t> &vertexindices, MeshTemplateCurvedEntity *curved);

    // Declare that node n2 is the periodic image of node n1 (n1 the master).
    void add_periodic_node_pair(const nodeindex_t &n1, const nodeindex_t &n2);

    const std::vector<MeshTemplateNode *> &get_nodes() const { return nodes; }
    std::vector<MeshTemplateNode *> &get_nodes() { return nodes; }
    const std::vector<MeshTemplateFacet *> &get_facets() const { return facets; }
    const std::vector<std::string> &get_boundary_names() const { return boundary_names; }
    std::vector<double> get_node_position(nodeindex_t index) const { return std::vector<double>{nodes[index]->x, nodes[index]->y, nodes[index]->z}; }
    // Find pairs of interface element collections that face each other
    // across a shared boundary (e.g. the two sides of an internal interface
    // used for e.g. two-phase-flow coupling) and wire up their opposite-side
    // connections via _add_opposite_interface_connection.
    void _find_opposite_interface_connections();
    // Determine boundary names whose associated facets intersect (share nodes/edges)
    // with facets of another interface, e.g. to detect contact-line/triple-junction sets.
    std::set<std::string> _find_interface_intersections();

    unsigned int get_boundary_index(const std::string &boundname) const;
    void add_node_to_boundary(const std::string &boundname, const nodeindex_t &ni);
    void add_nodes_to_boundary(const std::string &boundname, const std::vector<nodeindex_t> &ni);
    // Mark all nodes referenced by `ni` (and, via `vertexindices`, the facet
    // itself, optionally attached to curved geometry `curved`) as lying on boundary `boundname`.
    void add_facet_to_boundary(const std::string &boundname, const std::vector<nodeindex_t> &ni,const std::vector<nodeindex_t> &vertexindices, MeshTemplateCurvedEntity *curved=nullptr);

    std::map<MeshTemplateNode *, nodeindex_t, std::function<bool(const MeshTemplateNode *, const MeshTemplateNode *)>> &get_unique_node_map() { return nodemap; }
    // Create a new, initially-empty MeshTemplateElementCollection ("domain") with the given name.
    MeshTemplateElementCollection *new_bulk_element_collection(std::string name);

    MeshTemplateElementCollection *get_collection(std::string name);
    std::vector<MeshTemplateElementCollection *> &get_collections() { return bulk_element_collections; }
    //   void finalise_creation();

    // Build the concrete oomph-lib element (and any nodes it still needs)
    // for the geometric element `el` belonging to collection `coll`, using
    // `coll`'s attached element code to determine which oomph-lib element
    // class/nodal space (C1/C2/C1TB/C2TB) to instantiate.
    BulkElementBase *factory_element(MeshTemplateElement *el, MeshTemplateElementCollection *coll);
    // Resolve all periodic node pairs registered via add_periodic_node_pair
    // (plus their implied intermediate/mid-side nodes, via inter_nodes_periodic)
    // into actual oomph-lib node periodicity links (Node::make_periodic).
    void link_periodic_nodes();

    int get_dimension() const { return dim; }
    Problem *get_problem() { return problem; }

    virtual void _add_opposite_interface_connection(const std::string &sideA, const std::string &sideB) {} // Implemented in Python
  };

}
