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

  class MeshTemplate;

  class BulkElementBase;
  typedef std::size_t nodeindex_t;

  // Mesh templates share a list of nodes
  // They have information of subdomains and added elements

  class MeshTemplateElementCollection;
  class MeshTemplateNode
  {
  public:
    double x, y, z;
    nodeindex_t index;
    Node *oomph_node;
    int periodic_master;
    //	std::vector<nodeindex_t> periodic_master_of;
    bool on_curved_facet;
    std::set<unsigned int> on_boundaries;
    std::set<MeshTemplateElementCollection *> part_of_domain;
    MeshTemplateNode(double _x, double _y, double _z) : x(_x), y(_y), z(_z), oomph_node(NULL), periodic_master(-1), on_curved_facet(false) {}
    MeshTemplateNode(double _x, double _y) : x(_x), y(_y), z(0.0), oomph_node(NULL), periodic_master(-1), on_curved_facet(false) {}
    MeshTemplateNode(double _x) : x(_x), y(0.0), z(0.0), oomph_node(NULL), periodic_master(-1), on_curved_facet(false) {}
  };

  class MeshTemplateCurvedEntity
  {
  protected:
    unsigned dim;
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
    static std::map<int, MeshTemplateCurvedEntity *> load_from_strings(const std::vector<std::string> &s, size_t &currline);
    virtual unsigned get_parametric_dimension() const { return dim; }
    virtual void parametric_to_position(const unsigned &t, const std::vector<double> &parametric, std::vector<double> &position) { throw_runtime_error("Empty parametric_to_position called"); }
    virtual void position_to_parametric(const unsigned &t, const std::vector<double> &position, std::vector<double> &parametric) { throw_runtime_error("Empty position_to_parametric called"); };
    virtual void apply_periodicity(std::vector<std::vector<double>> &parametric){};
    virtual std::string get_information_string() { throw_runtime_error("Please implement get_information_string"); }
  };

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

  // Spherical object with less than 90 deg opening angle
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

  class CurvedEntityCatmullRomSpline : public MeshTemplateCurvedEntity
  {
  protected:
    std::vector<std::vector<double>> pts;
    std::vector<double> samples;
    std::vector<std::vector<double>> samplepos;
    unsigned N;
    void gen_samples(unsigned num);

  public:
    virtual void interpolate(double t, std::vector<double> &pos);
    virtual void dinterpolate(double t, std::vector<double> &dpos);
    CurvedEntityCatmullRomSpline(const std::vector<std::vector<double>> &_pts);
    virtual void parametric_to_position(const unsigned &t, const std::vector<double> &parametric, std::vector<double> &position);
    virtual void position_to_parametric(const unsigned &t, const std::vector<double> &position, std::vector<double> &parametric);
    virtual std::string get_information_string();
  };

  class MeshTemplateFacet
  {
  public:
    MeshTemplateFacet(std::vector<unsigned> &inds, MeshTemplateCurvedEntity *curved, std::vector<MeshTemplateNode *> *nodes);
    std::vector<unsigned> sorted_inds; // For fast finding
    std::vector<unsigned> nodeinds;
    MeshTemplateCurvedEntity *curved_entity;
    std::vector<std::vector<double>> parametrics;
  };

  class MeshTemplateDomain;
  class MeshTemplateElement;

  class MeshTemplateMacroElementBase
  {
  protected:
    virtual std::vector<unsigned> find_permutation(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation) = 0;

  public:
    void set_facet(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation);
    std::vector<MeshTemplateFacet *> facets;
    std::vector<std::vector<unsigned>> permutation;
    std::vector<std::vector<pyoomph::Node *>> default_facet_nodes;
    MeshTemplateMacroElementBase(MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes);
    virtual void macro_element_boundary(const unsigned &t, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f) = 0;
  };

  class MeshTemplateQMacroElement2 : public oomph::QMacroElement<2>, public MeshTemplateMacroElementBase
  {
  protected:
    std::vector<unsigned> find_permutation(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation);

  public:
    MeshTemplateQMacroElement2(MeshTemplateDomain *domain, unsigned index, MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes);
    virtual void macro_element_boundary(const unsigned &t, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f);
  };

  class MeshTemplateTMacroElement2 : public oomph::TMacroElement<2>, public MeshTemplateMacroElementBase
  {
  protected:
    std::vector<unsigned> find_permutation(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation);

  public:
    MeshTemplateTMacroElement2(MeshTemplateDomain *domain, unsigned index, MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes);
    virtual void macro_element_boundary(const unsigned &t, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f);
  };

  class MeshTemplateQMacroElement3 : public oomph::QMacroElement<3>, public MeshTemplateMacroElementBase
  {
  protected:
    std::vector<unsigned> find_permutation(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation);

  public:
    MeshTemplateQMacroElement3(MeshTemplateDomain *domain, unsigned index, MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes);
    virtual void macro_element_boundary(const unsigned &t, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f);
  };

  class MeshTemplateDomain : public oomph::Domain
  {
  public:
    MeshTemplateDomain();
    void push_back_macro_element(oomph::MacroElement *macro) { Macro_element_pt.push_back(macro); }
    void macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f);
  };

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

  class MeshTemplateElementTriC1TB : public MeshTemplateElementTriC1
  {
  protected:
  public:
    MeshTemplateElementTriC1TB(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4);
    unsigned int get_nnode_C1TB() const { return 4; }
    unsigned int get_node_index_C1TB(const unsigned int &i) const { return node_indices[i]; }
  };


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
  };
  
 

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

  class MeshTemplateElementTetraC2TB : public MeshTemplateElementTetraC2
  {
  protected:
  public:
    MeshTemplateElementTetraC2TB(std::vector<nodeindex_t> ninds);
    unsigned int get_nnode_C2TB() const { return 15; }
    unsigned int get_node_index_C2TB(const unsigned int &i) const { return node_indices[i]; }
  };

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
    virtual std::vector<double> get_reference_position_for_IC_and_DBC(std::set<unsigned int> boundindices);
    virtual int get_element_dimension() const { return dim; }
    virtual int nodal_dimension();
    void set_nodal_dimension(int d) { Nodal_dimension = d; }
    virtual int lagrangian_dimension();
    void set_lagrangian_dimension(int d) { Lagr_dimension = d; }
    MeshTemplateElementCollection(MeshTemplate *t, std::string n) : mesh_template(t), name(n) {}
    MeshTemplateElementPoint *add_point_element(const nodeindex_t &n1);
    MeshTemplateElementLineC1 *add_line_1d_C1(const nodeindex_t &n1, const nodeindex_t &n2);
    MeshTemplateElementLineC2 *add_line_1d_C2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3);
    MeshTemplateElementQuadC1 *add_quad_2d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4);
    MeshTemplateElementQuadC2 *add_quad_2d_C2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4,
                                              const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7, const nodeindex_t &n8, const nodeindex_t &n9);
    MeshTemplateElementTriC1 *add_tri_2d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3);
    std::vector<MeshTemplateElementTriC1*> add_SV_tri_2d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3);    
    MeshTemplateElementTriC2 *add_tri_2d_C2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5, const nodeindex_t &n6);

    MeshTemplateElementBrickC1 *add_brick_3d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4,
                                                const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7, const nodeindex_t &n8);
    MeshTemplateElementBrickC2 *add_brick_3d_C2(const std::vector<nodeindex_t> &inds);
    MeshTemplateElementTetraC1 *add_tetra_3d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4);
    MeshTemplateElementTetraC2 *add_tetra_3d_C2(const std::vector<nodeindex_t> &inds);

    const std::vector<MeshTemplateElement *> &get_elements() const { return elements; }
    std::vector<std::string> get_adjacent_boundary_names();
    void set_element_code(DynamicBulkElementInstance *code_inst);
    //	void set_element_class(BaseFiniteElementCode & cls);
    MeshTemplate *get_template() { return mesh_template; }
    virtual ~MeshTemplateElementCollection();
    void set_all_nodes_as_boundary_nodes() {all_nodes_as_boundary_nodes=true;}

    //  pyoomph::Mesh * get_oomph_mesh() { if (!oomph_mesh) throw_runtime_error("Mesh not yet created. Do a MeshTemplate::finalise_creation() first"); return oomph_mesh;}
  };

  class MeshTemplatePeriodicIntermediateNodeInfo
  {
  public:
    nodeindex_t myself;
    std::vector<nodeindex_t> parent_node_ids;
    MeshTemplatePeriodicIntermediateNodeInfo(nodeindex_t m, const std::vector<nodeindex_t> &pids) : myself(m), parent_node_ids(pids) { std::sort(parent_node_ids.begin(), parent_node_ids.end()); }
  };

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
    std::vector<MeshTemplateFacet *> facets;                                                                                     // Only special facets are usually added, i.e. curved ones
    std::map<MeshTemplateFacet *, unsigned, std::function<bool(const MeshTemplateFacet *, const MeshTemplateFacet *)>> facetmap; // Required for fast finding facets
    MeshTemplateDomain *domain;
    std::vector<MeshTemplatePeriodicIntermediateNodeInfo> inter_nodes_periodic;

  public:
    MeshTemplate();
    virtual void reset();
    void _set_problem(Problem *p) { problem = p; }
    virtual ~MeshTemplate();
    void flush_oomph_nodes();
    nodeindex_t add_node(double x, double y = 0.0, double z = 0.0);
    nodeindex_t add_node_unique(double x, double y = 0.0, double z = 0.0); // Checks if node exists
    nodeindex_t add_intermediate_node_unique(const nodeindex_t &n1, const nodeindex_t &n2);
    nodeindex_t add_intermediate_node_unique(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, bool boundary_possible); // For tri C2TB
    nodeindex_t add_intermediate_node_unique(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, bool boundary_possible);

    void add_facet_to_curve_entity(std::vector<unsigned> &facetnodes, MeshTemplateCurvedEntity *curved);

    void add_periodic_node_pair(const nodeindex_t &n1, const nodeindex_t &n2);

    const std::vector<MeshTemplateNode *> &get_nodes() const { return nodes; }
    std::vector<MeshTemplateNode *> &get_nodes() { return nodes; }

    const std::vector<std::string> &get_boundary_names() const { return boundary_names; }
    std::vector<double> get_node_position(nodeindex_t index) const { return std::vector<double>{nodes[index]->x, nodes[index]->y, nodes[index]->z}; }
    void _find_opposite_interface_connections();
    std::set<std::string> _find_interface_intersections();

    unsigned int get_boundary_index(const std::string &boundname) const;
    void add_node_to_boundary(const std::string &boundname, const nodeindex_t &ni);
    void add_nodes_to_boundary(const std::string &boundname, const std::vector<nodeindex_t> &ni);

    std::map<MeshTemplateNode *, nodeindex_t, std::function<bool(const MeshTemplateNode *, const MeshTemplateNode *)>> &get_unique_node_map() { return nodemap; }
    MeshTemplateElementCollection *new_bulk_element_collection(std::string name);

    MeshTemplateElementCollection *get_collection(std::string name);
    std::vector<MeshTemplateElementCollection *> &get_collections() { return bulk_element_collections; }
    //   void finalise_creation();

    BulkElementBase *factory_element(MeshTemplateElement *el, MeshTemplateElementCollection *coll);
    void link_periodic_nodes();

    int get_dimension() const { return dim; }
    Problem *get_problem() { return problem; }

    virtual void _add_opposite_interface_connection(const std::string &sideA, const std::string &sideB) {} // Implemented in Python
  };

}
