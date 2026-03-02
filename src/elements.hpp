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
#include "exception.hpp"
#include "jitbridge.h"

#include "oomph_lib.hpp"

#include "refineable_brick_element.h"

#include "refineable_telements.hpp"

#include "problem.hpp"

#include "mesh_as_geometric_object.h"

// #include "meshtemplate.hpp"

extern "C"
{
  double _pyoomph_get_element_size(void *);
  double _pyoomph_invoke_callback(void *, int, double *, int);
  void _pyoomph_invoke_multi_ret(void *, int, int, double *, double *, double *, int, int); // Index, flag,args,returns,derivative matrix, nargs,nret
  void _pyoomph_fill_shape_buffer_for_point(unsigned, JITFuncSpec_RequiredShapes_FiniteElement_t *, int);
}

namespace pyoomph
{

  // Required for the Hessian nodal derivatives of second order
  class RankSixTensor
  {
  protected:
    unsigned int n1, n2, n3, n4, n5, n6;
    std::vector<double> data;

  public:
    RankSixTensor(unsigned int _n1, unsigned int _n2, unsigned int _n3, unsigned int _n4, unsigned int _n5, unsigned int _n6) : n1(_n1), n2(_n2), n3(_n3), n4(_n4), n5(_n5), n6(_n6), data(_n1 * _n2 * _n3 * _n4 * _n5 * _n6) {}

    inline double &operator()(const unsigned long &i, const unsigned long &j, const unsigned long &k, const unsigned long &l, const unsigned long &m, const unsigned long &n)
    {
      return data[n6 * (n5 * (n4 * (n3 * (n2 * i + j) + k) + l) + m) + n];
    }

    inline double operator()(const unsigned long &i, const unsigned long &j, const unsigned long &k, const unsigned long &l, const unsigned long &m, const unsigned long &n) const
    {
      return data[n6 * (n5 * (n4 * (n3 * (n2 * i + j) + k) + l) + m) + n];
    }
  };

  class BulkElementBase;

  class SinglePassMultiAssembleHessianInfo
  {
  public:
    oomph::Vector<double> &Y;
    oomph::DenseMatrix<double> *M_Hessian;
    oomph::DenseMatrix<double> *J_Hessian;
    bool transposed;
    SinglePassMultiAssembleHessianInfo(oomph::Vector<double> &_Y, oomph::DenseMatrix<double> *J, oomph::DenseMatrix<double> *M, bool _transposed=false) : Y(_Y), M_Hessian(M), J_Hessian(J), transposed(_transposed) {}
  };

  class SinglePassMultiAssembleDParamInfo
  {
  public:
    double *const &parameter;
    oomph::Vector<double> *dRdparam;
    oomph::DenseMatrix<double> *dMdparam;
    oomph::DenseMatrix<double> *dJdparam;
    SinglePassMultiAssembleDParamInfo(double *const &param, oomph::Vector<double> *dres, oomph::DenseMatrix<double> *dJ = NULL, oomph::DenseMatrix<double> *dM = NULL) : parameter(param), dRdparam(dres), dMdparam(dM), dJdparam(dJ) {}
  };

  class SinglePassMultiAssembleInfo
  {
  protected:
    friend class BulkElementBase;
    std::vector<SinglePassMultiAssembleHessianInfo> hessians;
    std::vector<SinglePassMultiAssembleDParamInfo> dparams;

  public:
    int contribution = 0;
    oomph::Vector<double> *residuals = NULL;
    oomph::DenseMatrix<double> *jacobian = NULL;
    oomph::DenseMatrix<double> *mass_matrix = NULL;

    void add_hessian(oomph::Vector<double> &_Y, oomph::DenseMatrix<double> *J, oomph::DenseMatrix<double> *M = NULL,bool transposed=false)
    {
      hessians.push_back(SinglePassMultiAssembleHessianInfo(_Y, J, M,transposed));
    }
    
    SinglePassMultiAssembleHessianInfo & get_hessian(unsigned int i) { return  hessians[i]; }
    
    void add_param_deriv(double *const &param, oomph::Vector<double> *dres, oomph::DenseMatrix<double> *dJ = NULL, oomph::DenseMatrix<double> *dM = NULL)
    {
      dparams.push_back(SinglePassMultiAssembleDParamInfo(param, dres, dJ, dM));
    }
    SinglePassMultiAssembleInfo(int contrib, oomph::Vector<double> *res, oomph::DenseMatrix<double> *J, oomph::DenseMatrix<double> *M = NULL) : contribution(contrib), residuals(res), jacobian(J), mass_matrix(M) {}
  };

  class ElementBase : public virtual oomph::GeneralisedElement
  {
  };

  class FiniteElementBase : public virtual ElementBase, public virtual oomph::RefineableSolidElement, public virtual oomph::ElementWithZ2ErrorEstimator
  {
  public:
  };

  /*Meshio type indices
  0 : vertex
  1 : line
  2 : line3
  3 : triangle
  4 : triangle6
  5 : triangle7
  6 : quad
  7 : quad8 (not intended to be implemented)
  8 : quad9

  */

  class IntegrationSchemeStorage
  {
  protected:
    std::map<unsigned, oomph::Integral *> Q1d;
    std::map<unsigned, oomph::Integral *> T1d;
    std::map<unsigned, oomph::Integral *> Q2d;
    std::map<unsigned, oomph::Integral *> T2d;
    std::map<unsigned, oomph::Integral *> T2dTB;
    std::map<unsigned, oomph::Integral *> Q3d;
    std::map<unsigned, oomph::Integral *> T3d;
    std::map<unsigned, oomph::Integral *> T3dTB;
    std::map<unsigned, oomph::Integral *> &get_integral_order_map(bool tri, unsigned edim, bool bubble);
    void clean_up_map(std::map<unsigned, oomph::Integral *> &map);

  public:
    IntegrationSchemeStorage();
    virtual ~IntegrationSchemeStorage();
    oomph::Integral *get_integration_scheme(bool tris, unsigned edim, unsigned order, bool bubble = false);
  };

  extern IntegrationSchemeStorage integration_scheme_storage;

  class MeshTemplate;
  class MeshTemplateElement;
  class DynamicBulkElementInstance;
  class Problem;
  class BulkElementBase : public virtual FiniteElementBase
  {
  protected:
    DynamicBulkElementInstance *codeinst;

    JITElementInfo_t eleminfo;
    JITShapeInfo_t *shape_info;

    void free_element_info();

    virtual void constrain_bulk_position_space_to_C1();
    virtual void allocate_discontinous_fields();
    virtual void prepare_shape_buffer_for_integration(const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag);
    virtual void fill_shape_info_element_sizes(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, unsigned flag) const;
    virtual double fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds = NULL) const;
    virtual void fill_shape_info_at_s_dNodalPos_helper(JITShapeInfo_t *shape_info, const unsigned &index, const oomph::DenseMatrix<double> &interpolated_t, const oomph::DShape &dpsids_Element, const double det_Eulerian, const oomph::DenseMatrix<double> &aup, bool require_hessian, oomph::RankFourTensor<double> &DXdshape_il_jb, RankSixTensor *D2X2_dshape) const;
    virtual void fill_in_jacobian_from_lagragian_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    virtual void get_dnormal_dcoords_at_s(const oomph::Vector<double> &s, double ***dnormal_dcoord, double *****d2normal_dcoord2) const;
    void update_in_solid_position_fd(const unsigned &i) override; // For FD with element_sizes, we have to update the element size buffer
  public:
    unsigned _numpy_index;
    double initial_cartesian_nondim_size = 0.0;
    double initial_quality_factor = 0.0;
    virtual void fill_dofs_to_dirichlet_set_indices(std::vector<unsigned> &indices);
    virtual void fill_shape_buffer_for_integration_point(unsigned ipt, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag);
    virtual void set_remaining_shapes_appropriately(JITShapeInfo_t *shape_info, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes);
    virtual void fill_element_info(bool without_equations=false);
    virtual void describe_my_dofs(std::ostream &os, const std::string &in) { this->describe_local_dofs(os, in); }
    virtual double J_Lagrangian(const oomph::Vector<double> &s);
    virtual int get_internal_local_eqn(unsigned idindex, unsigned vindex) { return this->internal_local_eqn(idindex, vindex); }
    virtual void get_normal_at_s(const oomph::Vector<double> &s, oomph::Vector<double> &n, double ***dnormal_dcoord, double *****d2normal_dcoord2) const;

    // Discontinuous fields are stored as internal_data, on interfaces possibly also on external_data
    virtual oomph::Data *get_D0_nodal_data(const unsigned &fieldindex);
    virtual oomph::Data *get_DL_nodal_data(const unsigned &fieldindex);
    virtual oomph::Data *get_D1_nodal_data(const unsigned &fieldindex);
    virtual oomph::Data *get_D1TB_nodal_data(const unsigned &fieldindex);
    virtual oomph::Data *get_D2_nodal_data(const unsigned &fieldindex);
    virtual oomph::Data *get_D2TB_nodal_data(const unsigned &fieldindex);

    // Indices to the nodal buffer of the code generation
    virtual unsigned get_C2TB_buffer_index(const unsigned &fieldindex);
    virtual unsigned get_C2_buffer_index(const unsigned &fieldindex);
    virtual unsigned get_C1TB_buffer_index(const unsigned &fieldindex);
    virtual unsigned get_C1_buffer_index(const unsigned &fieldindex);
    virtual unsigned get_D2TB_buffer_index(const unsigned &fieldindex);
    virtual unsigned get_D2_buffer_index(const unsigned &fieldindex);
    virtual unsigned get_D1TB_buffer_index(const unsigned &fieldindex);
    virtual unsigned get_D1_buffer_index(const unsigned &fieldindex);
    virtual unsigned get_DL_buffer_index(const unsigned &fieldindex);
    virtual unsigned get_D0_buffer_index(const unsigned &fieldindex);

    // Parent elements may have more nodal data entries than the interfaces. These functions cast a interface nodal index to the nodal index of the defining element
    virtual unsigned get_D2TB_node_index(const unsigned &fieldindex, const unsigned &nodeindex) const { return nodeindex; }
    virtual unsigned get_D2_node_index(const unsigned &fieldindex, const unsigned &nodeindex) const { return nodeindex; }
    virtual unsigned get_D1TB_node_index(const unsigned &fieldindex, const unsigned &nodeindex) const { return nodeindex; }
    virtual unsigned get_D1_node_index(const unsigned &fieldindex, const unsigned &nodeindex) const { return nodeindex; }

    virtual int get_C2TB_local_equation(const unsigned &fieldindex, const unsigned &nodeindex, const bool &by_elemental_node_index = false);
    virtual int get_C2_local_equation(const unsigned &fieldindex, const unsigned &nodeindex, const bool &by_elemental_node_index = false);
    virtual int get_C1TB_local_equation(const unsigned &fieldindex, const unsigned &nodeindex, const bool &by_elemental_node_index = false);
    virtual int get_C1_local_equation(const unsigned &fieldindex, const unsigned &nodeindex, const bool &by_elemental_node_index = false);
    virtual int get_D2TB_local_equation(const unsigned &fieldindex, const unsigned &nodeindex);
    virtual int get_D2_local_equation(const unsigned &fieldindex, const unsigned &nodeindex);
    virtual int get_D1TB_local_equation(const unsigned &fieldindex, const unsigned &nodeindex);
    virtual int get_D1_local_equation(const unsigned &fieldindex, const unsigned &nodeindex);
    virtual int get_DL_local_equation(const unsigned &fieldindex, const unsigned &nodeindex);
    virtual int get_D0_local_equation(const unsigned &fieldindex);

    virtual void get_D1_fields_at_s(unsigned history_index, const oomph::Vector<double> &s, oomph::Vector<double> &result) const;
    virtual void get_D2_fields_at_s(unsigned history_index, const oomph::Vector<double> &s, oomph::Vector<double> &result) const;
    virtual void get_D1TB_fields_at_s(unsigned history_index, const oomph::Vector<double> &s, oomph::Vector<double> &result) const;
    virtual void get_D2TB_fields_at_s(unsigned history_index, const oomph::Vector<double> &s, oomph::Vector<double> &result) const;
    virtual int nedges() const = 0;
    virtual void add_node_from_finer_neighbor_for_tesselated_numpy(int edge, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) {}
    virtual void inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) {}
    virtual void interpolate_hang_values();
    virtual unsigned num_DG_fields(bool base_bulk_only);
    virtual void interpolate_hang_values_at_interface() {}
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    virtual double fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds = NULL) const;
    virtual unsigned get_meshio_type_index() const = 0;
    virtual void map_nodes_on_macro_element();
    virtual void assemble_hessian_tensor(oomph::DenseMatrix<double> &hbuffer);
    virtual void assemble_hessian_and_mass_hessian(oomph::RankThreeTensor<double> &hbuffer, oomph::RankThreeTensor<double> &mbuffer);
    // Taking the old mesh, map an element with the local coordinates associated to each integration point of the new mesh.
    virtual void prepare_zeta_interpolation(oomph::MeshAsGeomObject *mesh_as_geom);
    // Enable projection
    bool enable_zeta_projection = false;
    // Initialise vector to store.
    std::vector<std::pair<pyoomph::BulkElementBase *, oomph::Vector<double>>> coords_oldmesh;
    // Fill in residuals for projection.
    virtual void residuals_for_zeta_projection(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, const unsigned &do_fill_jacobian);
    // Assign projection time to variable.
    unsigned projection_time = 0;
    const JITElementInfo_t *get_eleminfo() const { return &eleminfo; }
    JITElementInfo_t *get_eleminfo() { return &eleminfo; }
    double get_element_diam() const;
    virtual std::vector<double> get_macro_element_coordinate_at_s(oomph::Vector<double> s);
    DynamicBulkElementInstance *get_code_instance() { return codeinst; }
    const DynamicBulkElementInstance *const get_code_instance() const { return codeinst; }

    static DynamicBulkElementInstance *__CurrentCodeInstance; // Really annoying, but no other way to pass it through the entire mesh stur

    static unsigned zeta_time_history;    // Index in time for zeta. Only Eulerian
    static unsigned zeta_coordinate_type; // 0: Lagrangian, 1: Eulerian -- On interfaces usually boundary coordinate
    static bool use_eigen_error_estimators; 

    double zeta_nodal(const unsigned &n, const unsigned &k, const unsigned &i) const
    {
      if (!zeta_coordinate_type)
        return lagrangian_position_gen(n, k, i);
      else
      {
        return nodal_position_gen(zeta_time_history, n, k, i);
      }
    }

    BulkElementBase();
    static BulkElementBase *create_from_template(MeshTemplate *mt, MeshTemplateElement *el);

    virtual void ensure_external_data();

    virtual void connect_periodic_tree(BulkElementBase *other, const int &mydir, const int &otherdir);

    virtual std::vector<std::string> get_dof_names(bool not_a_root_call = false);
    virtual void debug_analytical_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, double diff_eps);
    virtual void fill_in_generic_residual_contribution_jit(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &mass_matrix, unsigned flag);

    ///\short Compute the derivatives of the
    /// residuals with respect to a parameter
    /// Flag=1 (or 0): do (or don't) compute the Jacobian as well.
    /// Flag=2: Fill in mass matrix too.
    virtual void fill_in_generic_dresidual_contribution_jit(double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam, oomph::DenseMatrix<double> &dmass_matrix_dparam, unsigned flag);
    virtual void get_multi_assembly(std::vector<SinglePassMultiAssembleInfo> &info);

    void fill_in_contribution_to_residuals(oomph::Vector<double> &residuals)
    {
      fill_in_generic_residual_contribution_jit(residuals, oomph::GeneralisedElement::Dummy_matrix, oomph::GeneralisedElement::Dummy_matrix, 0);
    }
    void fill_in_contribution_to_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    void fill_in_contribution_to_jacobian_and_mass_matrix(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &mass_matrix);

    void fill_in_contribution_to_dresiduals_dparameter(double *const &parameter_pt, oomph::Vector<double> &dres_dparam)
    {
      fill_in_generic_dresidual_contribution_jit(parameter_pt, dres_dparam, oomph::GeneralisedElement::Dummy_matrix, oomph::GeneralisedElement::Dummy_matrix, 0);
    }

    void fill_in_contribution_to_djacobian_dparameter(double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam)
    {
      fill_in_generic_dresidual_contribution_jit(parameter_pt, dres_dparam, djac_dparam, oomph::GeneralisedElement::Dummy_matrix, 1);
    }

    void fill_in_contribution_to_djacobian_and_dmass_matrix_dparameter(double *const &parameter_pt, oomph::Vector<double> &dres_dparam, oomph::DenseMatrix<double> &djac_dparam, oomph::DenseMatrix<double> &dmass_matrix_dparam)
    {
      fill_in_generic_dresidual_contribution_jit(parameter_pt, dres_dparam, djac_dparam, dmass_matrix_dparam, 2);
    }

    void fill_in_contribution_to_hessian_vector_products(oomph::Vector<double> const &Y, oomph::DenseMatrix<double> const &C, oomph::DenseMatrix<double> &product);
    void fill_in_generic_hessian(oomph::Vector<double> const &Y, oomph::DenseMatrix<double> &C, oomph::DenseMatrix<double> &product, unsigned flag);

    double eval_integral_expression(unsigned index);
    double eval_local_expression_at_s(unsigned index, const oomph::Vector<double> &s);
    double eval_local_expression_at_node(unsigned index, unsigned node_index);
    double eval_local_expression_at_midpoint(unsigned index);
    double eval_extremum_expression_at_s(unsigned index, const oomph::Vector<double> &s);
    double eval_extremum_expression_at_node(unsigned index, unsigned node_index);
    

    pyoomph::Node * create_interpolated_node(const oomph::Vector<double> & s,bool as_boundary_node);

    bool eval_tracer_advection_in_s_space(unsigned index, double time_frac, const oomph::Vector<double> &s, oomph::Vector<double> &svelo);

    //  void assign_local_eqn_numbers(const bool &store_local_dof_pt);
    void assign_additional_local_eqn_numbers();
    //  virtual void assign_all_generic_local_eqn_numbers(const bool &store_local_dof_pt);

    virtual ~BulkElementBase();

    unsigned ndof_types() const;
    void get_dof_numbers_for_unknowns(std::list<std::pair<unsigned long, unsigned>> &dof_lookup_list) const;

    virtual BulkElementBase *create_son_instance() const = 0;
    unsigned ncont_interpolated_values() const;
    virtual unsigned nadditional_fields_C1();
    virtual unsigned nadditional_fields_C1TB();
    virtual unsigned nadditional_fields_C2();
    virtual unsigned nadditional_fields_C2TB();
    virtual unsigned required_nvalue(const unsigned &n) const;

    virtual void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const = 0;
    virtual void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const = 0;
    virtual void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const = 0;
    virtual void shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    virtual void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    virtual int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const = 0;
    virtual void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const = 0;

    virtual void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const = 0;
    virtual void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const = 0;
    virtual void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const = 0;
    virtual void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    virtual void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    virtual unsigned int get_node_index_C1_to_element(const unsigned int &i) const = 0;
    virtual unsigned int get_node_index_C2_to_element(const unsigned int &i) const = 0;
    virtual unsigned int get_node_index_C2TB_to_element(const unsigned int &i) const { throw_runtime_error("Implement"); }
    virtual unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { throw_runtime_error("Implement for meshio type index "+std::to_string(this->get_meshio_type_index())); }

    // Mapping nnode-> -1 if not defined here or index
    virtual int get_node_index_element_to_C1(const unsigned int &i) const { return i; }
    virtual int get_node_index_element_to_C2(const unsigned int &i) const { return i; }
    virtual int get_node_index_element_to_C2TB(const unsigned int &i) const { return this->get_node_index_element_to_C2(i); }
    virtual int get_node_index_element_to_C1TB(const unsigned int &i) const { return this->get_node_index_element_to_C1(i); }

    virtual oomph::Node *construct_node(const unsigned &n);
    virtual oomph::Node *construct_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt);
    virtual oomph::Node *construct_boundary_node(const unsigned &n);
    virtual oomph::Node *construct_boundary_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt);
    virtual oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index);

    virtual bool is_node_index_part_of_C2(const unsigned &n) { return true; }
    virtual bool is_node_index_part_of_C2TB(const unsigned &n) { return this->is_node_index_part_of_C2(n); }
    virtual bool is_node_index_part_of_C1(const unsigned &n) { return true; }
    virtual bool is_node_index_part_of_C1TB(const unsigned &n) { return this->is_node_index_part_of_C1(n); }

    virtual void get_supporting_C1_nodes_of_C2_node(const unsigned &n, std::vector<oomph::Node *> &support) { throw_runtime_error("Implement"); }

    void get_interpolated_fields_C2TB(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t = 0) const;
    void get_interpolated_fields_C2(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t = 0) const;
    void get_interpolated_fields_C1TB(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t = 0) const;
    void get_interpolated_fields_C1(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t = 0) const;
    void get_interpolated_fields_DL(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t = 0) const;
    void get_interpolated_fields_D0(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t = 0) const;

    virtual oomph::Vector<double> get_midpoint_s();                        // Set s=[0.5*(smin+smax), ... ] (but modified e.g. for tris)
    oomph::Vector<double> get_Eulerian_midpoint_from_local_coordinate();   // Set s=[0.5*(smin+smax), ... ] and evaluate the position
    oomph::Vector<double> get_Lagrangian_midpoint_from_local_coordinate(); // Set s=[0.5*(smin+smax), ... ] and evaluate the position

    void get_interpolated_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values);
    void get_interpolated_values(const oomph::Vector<double> &s, oomph::Vector<double> &values) { get_interpolated_values(0, s, values); }
    void get_interpolated_discontinuous_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values);
    void get_interpolated_discontinuous_values(const oomph::Vector<double> &s, oomph::Vector<double> &values) { get_interpolated_discontinuous_values(0, s, values); }
    void output(std::ostream &outfile, const unsigned &n_plot);

    virtual std::vector<double> get_outline(bool lagrangian) { return std::vector<double>(0); }
    unsigned num_Z2_flux_terms();
    void get_Z2_flux(const oomph::Vector<double> &s, oomph::Vector<double> &flux);
    void rebuild_from_sons(oomph::Mesh *&mesh_pt);
    void further_build();
    virtual void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather) { throw_runtime_error("Implement"); }
    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt);

    unsigned nscalar_paraview() const;
    void scalar_value_paraview(std::ofstream &file_out, const unsigned &i, const unsigned &nplot) const;
    std::string scalar_name_paraview(const unsigned &i) const;
    virtual void further_setup_hanging_nodes();

    virtual int get_nodal_index_by_name(oomph::Node *n, std::string fieldname);
    oomph::Node *node_pt_C1(const unsigned &n_C1) { return this->node_pt(this->get_node_index_C1_to_element(n_C1)); }

    /*
     inline void assign_nodal_local_eqn_numbers(const bool &store_local_dof_pt)
      {
       oomph::RefineableSolidElement::assign_nodal_local_eqn_numbers(store_local_dof_pt);
    //   assign_hanging_local_eqn_numbers(store_local_dof_pt);
    //	 fill_element_info();
      }
    */

    inline void assign_nodal_local_eqn_numbers(const bool &store_local_dof_pt)
    {
      FiniteElement::assign_nodal_local_eqn_numbers(store_local_dof_pt);
      assign_hanging_local_eqn_numbers(store_local_dof_pt);
      //	 fill_element_info();
    }

    virtual void unpin_dummy_values(); // C1 fields on C2 elements have dummy values on only C2 nodes, which needs to be pinned
    virtual void pin_dummy_values();

    void dynamic_split(oomph::Vector<BulkElementBase *> &son_pt) const;

    double geometric_jacobian(const oomph::Vector<double> &x) override;

    void get_debug_jacobian_info(oomph::Vector<double> &R, oomph::DenseMatrix<double> &J, std::vector<std::string> &dofnames);
    double elemental_error_max_override;

    virtual double get_quality_factor();

    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance) { throw_runtime_error("Implement for the specific element"); }

    virtual void set_integration_order(unsigned int order) { throw_runtime_error("Implement"); }

    virtual bool has_bubble() const { return false; } // If not, C2TB is the same space as C2

    virtual void debug_hessian(std::vector<double> Y, std::vector<std::vector<double>> C, double epsilon);
    virtual std::vector<std::pair<oomph::Data *, int>> get_field_data_list(std::string name, bool use_elemental_indices);
  };

  class BulkElementODE0d : public virtual BulkElementBase
  {
  protected:
    //	virtual void fill_element_info(); //TODO simplify this
    oomph::TimeStepper *timestepper;
    static oomph::PointIntegral Default_integration_scheme;

  public:
    int nedges() const { return 0; }
    virtual double fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds = NULL) const;
    virtual unsigned get_meshio_type_index() const { return 0; }
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return 0; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return 0; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return 0; }
    unsigned int get_node_index_C2TB_to_element(const unsigned int &i) const { return 0; }
    unsigned nrecovery_order() { return 0; }
    unsigned nvertex_node() const { return 0; }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return NULL; }
    void further_setup_hanging_nodes() {};
    void to_numpy(double *dest);
    void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const {}
    void build(oomph::Mesh *&, oomph::Vector<oomph::Node *> &, bool &, std::ofstream &) {}
    void check_integrity(double &max_error) { max_error = 0; }
    virtual BulkElementBase *create_son_instance() const { return NULL; }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const {}
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const {}
    void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const {}
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const {}
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 0;
      return 0;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const { isubelem = 0; }

    BulkElementODE0d(DynamicBulkElementInstance *code_inst, oomph::TimeStepper *tstepper);

    static BulkElementODE0d *construct_new(DynamicBulkElementInstance *code_inst, oomph::TimeStepper *tstepper)
    {
      BulkElementBase::__CurrentCodeInstance = code_inst;
      BulkElementODE0d *res = new BulkElementODE0d(code_inst, tstepper);
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual double get_quality_factor() { return 1.0; }

    virtual void set_integration_order(unsigned int order) {}
  };

  // Oomph-libs RefineableSolidQElement<1> needs to be adjusted, since it is marked as broken in the constructor
  class RefineableSolidLineElement : public virtual oomph::RefineableQElement<1>, public virtual oomph::RefineableSolidElement, public virtual oomph::QSolidElementBase
  {
  public:
    RefineableSolidLineElement() : oomph::RefineableQElement<1>(), oomph::RefineableSolidElement()
    {
    }

    /// Broken copy constructor
    RefineableSolidLineElement(const RefineableSolidLineElement &dummy)
    {
      oomph::BrokenCopy::broken_copy("RefineableSolidLineElement");
    }

    virtual ~RefineableSolidLineElement() {}

    void set_macro_elem_pt(oomph::MacroElement *macro_elem_pt)
    {
      oomph::QSolidElementBase::set_macro_elem_pt(macro_elem_pt);
    }

    void set_macro_elem_pt(oomph::MacroElement *macro_elem_pt, oomph::MacroElement *undeformed_macro_elem_pt)
    {
      oomph::QSolidElementBase::set_macro_elem_pt(macro_elem_pt, undeformed_macro_elem_pt);
    }

    void get_jacobian(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian)
    {
      oomph::RefineableSolidElement::get_jacobian(residuals, jacobian);
    }

    void build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt,
               bool &was_already_built,
               std::ofstream &new_nodes_file);
  };

  class BulkElementLine1dC1 : public virtual BulkElementBase,
                              public virtual oomph::QElement<1, 2>,
                              public virtual RefineableSolidLineElement
  {
  protected:
  public:
    int nedges() const { return 2; }
    BulkElementLine1dC1();
    bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    virtual unsigned get_meshio_type_index() const { return 1; }
    void check_integrity(double &max_error) { max_error = 0; } // TODO throw_runtime_error("IMPLEMENT");

    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return 0; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return i; }

    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    unsigned nrecovery_order() { return 1; }
    unsigned nvertex_node() const { return oomph::QElement<1, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<1, 2>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() {};
    virtual std::vector<double> get_outline(bool lagrangian);
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementLine1dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }

    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt)
    {
      BulkElementBase::pre_build(mesh_pt, new_node_pt);
      oomph::RefineableQElement<1>::pre_build(mesh_pt, new_node_pt);
    }

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 2;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 1, order)); }
    virtual void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
  };

  class BulkElementLine1dC2 : public virtual BulkElementBase, public virtual oomph::QElement<1, 3>, public virtual RefineableSolidLineElement
  {
  protected:
    static unsigned int index_C1_to_element[2];
    static int element_index_to_C1[3];
    static bool node_only_C2[3];

  public:
    int nedges() const { return 2; }
    virtual unsigned get_meshio_type_index() const { return 2; }
    BulkElementLine1dC2();
    void interpolate_hang_values();
    void interpolate_hang_values_at_interface();
    bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    void check_integrity(double &max_error) { max_error = 0; } // TODO

    bool is_node_index_part_of_C1(const unsigned &n) override { return !node_only_C2[n]; }

    int get_node_index_element_to_C1(const unsigned int &i) const override { return element_index_to_C1[i]; }
    int get_node_index_element_to_C1TB(const unsigned int &i) const override { return element_index_to_C1[i]; }
    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return index_C1_to_element[i]; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2TB_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return index_C1_to_element[i]; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 3;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 2; }
    unsigned nvertex_node() const { return oomph::QElement<1, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<1, 3>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() {};
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementLine1dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 1, order)); }
    virtual void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
  };

  // TRIANGULAR LINE ELEMENTS

  class BulkTElementLine1dC1 : public virtual BulkElementBase, public virtual oomph::TElement<1, 2>, public virtual oomph::RefineableTElement<1>
  {
  protected:
  public:
    int nedges() const { return 2; }
    BulkTElementLine1dC1();
    bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    virtual unsigned get_meshio_type_index() const { return 1; }
    void check_integrity(double &max_error) { max_error = 0; } // TODO throw_runtime_error("IMPLEMENT");

    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return 0; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return i; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    unsigned nrecovery_order() { return 1; }
    unsigned nvertex_node() const { return oomph::TElement<1, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<1, 2>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() {};
    virtual std::vector<double> get_outline(bool lagrangian);
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkTElementLine1dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }

    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt)
    {
      BulkElementBase::pre_build(mesh_pt, new_node_pt);
      oomph::RefineableTElement<1>::pre_build(mesh_pt, new_node_pt);
    }

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 2;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 1, order)); }
  };

  class BulkTElementLine1dC2 : public virtual BulkElementBase, public virtual oomph::TElement<1, 3>, public virtual oomph::RefineableTElement<1>
  {
  protected:
    static unsigned int index_C1_to_element[2];
    static int element_index_to_C1[3];
    static bool node_only_C2[3];

  public:
    int nedges() const { return 2; }
    virtual unsigned get_meshio_type_index() const { return 2; }
    BulkTElementLine1dC2();
    void interpolate_hang_values();
    void interpolate_hang_values_at_interface();
    bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    void check_integrity(double &max_error) { max_error = 0; } // TODO

    bool is_node_index_part_of_C1(const unsigned &n) override { return !node_only_C2[n]; }
    bool is_node_index_part_of_C1TB(const unsigned &n) override { return !node_only_C2[n]; }
    int get_node_index_element_to_C1(const unsigned int &i) const override { return element_index_to_C1[i]; }
    int get_node_index_element_to_C1TB(const unsigned int &i) const override { return element_index_to_C1[i]; }
    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return index_C1_to_element[i]; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2TB_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return index_C1_to_element[i]; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 3;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 2; }
    unsigned nvertex_node() const { return oomph::TElement<1, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<1, 3>::vertex_node_pt(j); }
    // void further_setup_hanging_nodes() {BulkElementBase::further_setup_hanging_nodes();};
    void further_setup_hanging_nodes() {};
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkTElementLine1dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 1, order)); }
  };

  class BulkElementQuad2dC1 : public virtual BulkElementBase, public virtual oomph::QElement<2, 2>, public virtual oomph::RefineableSolidQElement<2>
  {
  protected:
  public:
    int nedges() const { return 4; }
    BulkElementQuad2dC1();
    virtual void interpolate_hang_values();
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    virtual unsigned get_meshio_type_index() const { return 6; }

    void check_integrity(double &max_error) { max_error = 0; } // TODO

    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return 0; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return i; }

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    void add_node_from_finer_neighbor_for_tesselated_numpy(int edge, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes);
    void inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes);
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index);
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::QElement<2, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<2, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementQuad2dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 2, order)); }
  };

  class BulkElementQuad2dC2 : public virtual BulkElementBase, public virtual oomph::QElement<2, 3>, public virtual oomph::RefineableSolidQElement<2>
  {
  protected:
    static unsigned int index_C1_to_element[4];
    static int element_index_to_C1[9];
    static bool node_only_C2[9];
    virtual void constrain_bulk_position_space_to_C1();

  public:
    virtual void get_supporting_C1_nodes_of_C2_node(const unsigned &n, std::vector<oomph::Node *> &support);
    int nedges() const { return 4; }
    BulkElementQuad2dC2();
    virtual unsigned get_meshio_type_index() const { return 8; }

    void interpolate_hang_values();
    void interpolate_hang_values_at_interface();
    bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);

    void check_integrity(double &max_error) { max_error = 0; } // TODO
    virtual std::vector<double> get_outline(bool lagrangian);
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    virtual oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index);
    bool is_node_index_part_of_C1(const unsigned &n) override { return !node_only_C2[n]; }

    int get_node_index_element_to_C1(const unsigned int &i) const override { return element_index_to_C1[i]; }
    int get_node_index_element_to_C1TB(const unsigned int &i) const override { return element_index_to_C1[i]; }
    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return index_C1_to_element[i]; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2TB_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return index_C1_to_element[i]; }

    void add_node_from_finer_neighbor_for_tesselated_numpy(int edge, oomph::Node *n, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes);
    void inform_coarser_neighbors_for_tesselated_numpy(std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes);
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;

    unsigned nrecovery_order() { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::QElement<2, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<2, 3>::vertex_node_pt(j); }

    void further_setup_hanging_nodes();
    oomph::Node *interpolating_node_pt(const unsigned &n, const int &value_id);
    double local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id);
    oomph::Node *get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id);
    unsigned ninterpolating_node_1d(const int &value_id);
    unsigned ninterpolating_node(const int &value_id);
    void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const;
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementQuad2dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 2, order)); }
  };

  class BulkElementTri2dC1 : public virtual BulkElementBase, public virtual oomph::TElement<2, 2>, public virtual oomph::RefineableTElement<2>
  {
  protected:
  public:
    virtual oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index);
    int nedges() const { return 3; }
    unsigned nnode_on_face() const override { return 2; }
    BulkElementTri2dC1(bool has_bubble = false);
    virtual void interpolate_hang_values();
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    virtual unsigned get_meshio_type_index() const { return 3; }
    void check_integrity(double &max_error) { max_error = 0; } // TODO
    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return 0; }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 3;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::TElement<2, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<2, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementTri2dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order)); }
    oomph::Vector<double> get_midpoint_s() override { return oomph::Vector<double>(this->dim(), 1.0 / 3.0); }
  };

  class BulkElementTri2dC1TB : public virtual BulkElementTri2dC1
  {
  private:
    static oomph::TBubbleEnrichedGauss<2, 3> Default_enriched_integration_scheme; // Don't know which scheme is best here
    //  static const unsigned Central_node_on_face[3];
  public:
    BulkElementTri2dC1TB();
    bool is_node_index_part_of_C1(const unsigned &n) override { return n < 3; }
    bool is_node_index_part_of_C1TB(const unsigned &n) override { return true; }
    virtual void interpolate_hang_values();
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const;

    int get_node_index_element_to_C1(const unsigned int &i) const override { return (i < 4 ? i : -1); }
    int get_node_index_element_to_C1TB(const unsigned int &i) const override { return i; }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return i; }
    inline void d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const { throw_runtime_error("Implement"); }

    void local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const;
    bool has_bubble() const { return true; }
    virtual unsigned get_meshio_type_index() const { return 66; } // Just some otherwise unused value here
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      if (tesselate_tri)
      {
        nsubdiv = 3;
        return 3;
      }
      else
      {
        nsubdiv = 1;
        return 4;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;

    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order, true)); }
  };

  class BulkElementTri2dC2TB;
  class BulkElementTri2dC2 : public virtual BulkElementBase, public virtual oomph::TElement<2, 3>, public virtual oomph::RefineableTElement<2>
  {
  protected:
    virtual void constrain_bulk_position_space_to_C1();

  public:
    void interpolate_hang_values_at_interface();
    virtual void get_supporting_C1_nodes_of_C2_node(const unsigned &n, std::vector<oomph::Node *> &support);
    virtual oomph::Node *boundary_node_pt(const int &face_index, const unsigned int index);
    int nedges() const { return 3; }
    unsigned nnode_on_face() const override { return 3; }
    BulkElementTri2dC2(bool with_bubble = false);
    virtual void interpolate_hang_values();
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    virtual unsigned get_meshio_type_index() const { return 9; }
    bool is_node_index_part_of_C1(const unsigned &n) override { return n < 3; }
    bool is_node_index_part_of_C1TB(const unsigned &n) override { return n < 3 || n == 6; }
    void check_integrity(double &max_error) { max_error = 0; } // TODO
    int get_node_index_element_to_C1(const unsigned int &i) const override { return (i < 3 ? i : -1); }
    int get_node_index_element_to_C1TB(const unsigned int &i) const override { return (i < 3 ? i : (i == 6 ? 3 : -1)); }
    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2TB_to_element(const unsigned int &i) const { return i; }
    //    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return i; }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      if (tesselate_tri)
      {
        nsubdiv = 4;
        return 3;
      }
      else
      {
        nsubdiv = 1;
        return 6;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::TElement<2, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<2, 3>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    virtual BulkElementBase *create_son_instance() const;
    virtual double factor_when_local_coordinate_becomes_invalid(const oomph::Vector<double> &s, const oomph::Vector<double> &ds, oomph::Vector<double> &snormal, double &sdistance);
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order)); }
    oomph::Vector<double> get_midpoint_s() override { return oomph::Vector<double>(this->dim(), 1.0 / 3.0); }
  };

  class BulkElementTri2dC2TB : public virtual BulkElementTri2dC2, public oomph::TBubbleEnrichedElementShape<2, 3>
  {
  private:
    static oomph::TBubbleEnrichedGauss<2, 3> Default_enriched_integration_scheme;
    //  static const unsigned Central_node_on_face[3];
  public:
    BulkElementTri2dC2TB();
    bool is_node_index_part_of_C2(const unsigned &n) override { return n < 6; }
    bool is_node_index_part_of_C2TB(const unsigned &n) override { return true; }
    virtual void interpolate_hang_values();
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    void shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const override;
    void dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const override;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { BulkElementTri2dC2::shape(s, psi); }
    void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const { oomph::TBubbleEnrichedElementShape<2, 3>::shape(s, psi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { BulkElementTri2dC2::dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { oomph::TBubbleEnrichedElementShape<2, 3>::dshape_local(s, psi, dpsi); }
    unsigned int get_node_index_C2TB_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return (i < 3 ? i : 6); }
    inline void d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const { throw_runtime_error("Implement"); }
    inline void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const { oomph::TBubbleEnrichedElementShape<2, 3>::shape(s, psi); }
    inline void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const { oomph::TBubbleEnrichedElementShape<2, 3>::dshape_local(s, psi, dpsids); }
    inline void local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const { oomph::TBubbleEnrichedElementShape<2, 3>::local_coordinate_of_node(j, s); }
    bool has_bubble() const { return true; }
    virtual unsigned get_meshio_type_index() const { return 99; } // Just some otherwise unused value here
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      if (tesselate_tri)
      {
        nsubdiv = 3;
        return 3;
      }
      else
      {
        nsubdiv = 1;
        return 7;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;

    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 2, order, true)); }
  };

  class BulkElementBrick3dC1 : public virtual BulkElementBase, public virtual oomph::QElement<3, 2>, public virtual oomph::RefineableSolidQElement<3>
  {
  protected:
  public:
    int nedges() const { return 8; }
    BulkElementBrick3dC1();
    virtual void interpolate_hang_values();
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    virtual unsigned get_meshio_type_index() const { return 11; }

    void check_integrity(double &max_error) { max_error = 0; } // TODO

    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return 0; }

    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      if (tesselate_tri)
      {
        throw_runtime_error("Tesselation of 3d not possible");
      }
      else
      {
        nsubdiv = 1;
        return 8;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::QElement<3, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<3, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementBrick3dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 3, order)); }
  };

  class BulkElementBrick3dC2 : public virtual BulkElementBase, public virtual oomph::QElement<3, 3>, public virtual oomph::RefineableSolidQElement<3>
  {
  protected:
    static unsigned int index_C1_to_element[8];
    static int element_index_to_C1[27];
    static bool node_only_C2[27];

  public:
    int nedges() const { return 8; }
    BulkElementBrick3dC2();
    virtual unsigned get_meshio_type_index() const { return 14; }

    void interpolate_hang_values();
    bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);

    void check_integrity(double &max_error) { max_error = 0; } // TODO
    virtual std::vector<double> get_outline(bool lagrangian);
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    bool is_node_index_part_of_C1(const unsigned &n) override { return !node_only_C2[n]; }

    int get_node_index_element_to_C1(const unsigned int &i) const override { return element_index_to_C1[i]; }
    int get_node_index_element_to_C1TB(const unsigned int &i) const override { return element_index_to_C1[i]; }
    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return index_C1_to_element[i]; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2TB_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return i; }

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      if (tesselate_tri)
      {
        throw_runtime_error("Tesselation of 3d not possible");
      }
      else
      {
        nsubdiv = 1;
        return 27;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;

    unsigned nrecovery_order() { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::QElement<3, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return QElement<3, 3>::vertex_node_pt(j); }

    void further_setup_hanging_nodes();
    oomph::Node *interpolating_node_pt(const unsigned &n, const int &value_id);
    double local_one_d_fraction_of_interpolating_node(const unsigned &n1d, const unsigned &i, const int &value_id);
    oomph::Node *get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s, const int &value_id);
    unsigned ninterpolating_node_1d(const int &value_id);
    unsigned ninterpolating_node(const int &value_id);
    void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const;
    void get_nodal_s_in_father(const unsigned int &l, oomph::Vector<double> &sfather);
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementBrick3dC2();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(false, 3, order)); }
  };

  class BulkElementTetra3dC1 : public virtual BulkElementBase, public virtual oomph::TElement<3, 2>, public virtual oomph::RefineableTElement<3>
  {
  protected:
  public:
    int nedges() const { return 6; }
    BulkElementTetra3dC1();
    virtual void interpolate_hang_values();
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    virtual unsigned get_meshio_type_index() const { return 4; }

    void check_integrity(double &max_error) { max_error = 0; } // TODO

    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return 0; }
    int get_node_index_element_to_C1(const unsigned int &i) const override { return i; }
    int get_node_index_element_to_C1TB(const unsigned int &i) const override { return i; }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { throw_runtime_error("Makes no sense"); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { throw_runtime_error("Makes no sense"); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      if (tesselate_tri)
      {
        throw_runtime_error("Tesselation of 3d not possible");
      }
      else
      {
        nsubdiv = 1;
        return 4;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    unsigned nrecovery_order() { return 1; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::TElement<3, 2>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<3, 2>::vertex_node_pt(j); }
    void further_setup_hanging_nodes() { BulkElementBase::further_setup_hanging_nodes(); } // There can't be any problem here, since it is all isoparametric
    virtual BulkElementBase *create_son_instance() const
    {
      BulkElementBase::__CurrentCodeInstance = codeinst;
      auto res = new BulkElementTetra3dC1();
      res->codeinst = codeinst;
      BulkElementBase::__CurrentCodeInstance = NULL;
      return res;
    }
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, order)); }
    oomph::Vector<double> get_midpoint_s() override { return oomph::Vector<double>(this->dim(), 1.0 / 3.0); }
  };

  class BulkElementTetra3dC2TB;
  class BulkElementTetra3dC2 : public virtual BulkElementBase, public virtual oomph::TElement<3, 3>, public virtual oomph::RefineableTElement<3>
  {
  protected:
    static unsigned int index_C1_to_element[4];
    static int element_index_to_C1[15];
    static bool node_only_C2[15]; // Including the C2TBs

  public:
    int nedges() const { return 6; }
    BulkElementTetra3dC2(bool has_bubble = false);
    virtual unsigned get_meshio_type_index() const { return 10; }

    void interpolate_hang_values();
    bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);

    void check_integrity(double &max_error) { max_error = 0; } // TODO
    virtual std::vector<double> get_outline(bool lagrangian);
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { this->shape(s, psi); }
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;

    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { this->dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;

    bool is_node_index_part_of_C1(const unsigned &n) override { return !node_only_C2[n]; }

    int get_node_index_element_to_C1(const unsigned int &i) const override { return element_index_to_C1[i]; }
    int get_node_index_element_to_C1TB(const unsigned int &i) const override { throw_runtime_error("TODO"); }
    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return index_C1_to_element[i]; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return index_C1_to_element[i]; }

    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      if (tesselate_tri)
      {
        throw_runtime_error("Tesselation of 3d not possible");
      }
      else
      {
        nsubdiv = 1;
        return 10;
      }
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;

    unsigned nrecovery_order() { return 2; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    unsigned nvertex_node() const { return oomph::TElement<3, 3>::nvertex_node(); }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return TElement<3, 3>::vertex_node_pt(j); }

    void further_setup_hanging_nodes();
    // TODO: For refinement!
    // oomph::Node* interpolating_node_pt(const unsigned &n,const int &value_id);
    // double local_one_d_fraction_of_interpolating_node(const unsigned &n1d,const unsigned &i,const int &value_id);
    // oomph::Node* get_interpolating_node_at_local_coordinate(const oomph::Vector<double> &s,const int &value_id); //TO be done
    // unsigned ninterpolating_node_1d(const int &value_id);
    // unsigned ninterpolating_node(const int &value_id);
    void interpolating_basis(const oomph::Vector<double> &s, oomph::Shape &psi, const int &value_id) const;
    virtual BulkElementBase *create_son_instance() const;
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, order)); }
    oomph::Vector<double> get_midpoint_s() override { return oomph::Vector<double>(this->dim(), 1.0 / 3.0); }
  };

  class BulkElementTetra3dC2TB : public virtual BulkElementTetra3dC2, public oomph::TBubbleEnrichedElementShape<3, 3>
  {
  private:
    static oomph::TBubbleEnrichedGauss<3, 3> Default_enriched_integration_scheme;
    //  static const unsigned Central_node_on_face[3];
  public:
    BulkElementTetra3dC2TB();

    virtual unsigned get_meshio_type_index() const { return 100; } // Just some otherwise unused value here

    virtual void interpolate_hang_values();
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      if (tesselate_tri)
      {
        throw_runtime_error("Tesselation of 3d not possible");
      }
      else
      {
        nsubdiv = 1;
        return 15;
      }
    }
    //   void fill_element_nodal_indices_for_numpy(int *indices,unsigned isubelem,bool tesselate_tri,std::vector<std::vector<std::set<oomph::Node*>>> & add_nodes) const;

    int get_node_index_element_to_C1TB(const unsigned int &i) const override { throw_runtime_error("TODO"); }
    unsigned int get_node_index_C2TB_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return (i < 4 ? i : 14); }
    bool is_node_index_part_of_C2(const unsigned &n) override { return n < 14; }
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const { BulkElementTetra3dC2::shape(s, psi); }
    void shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const { oomph::TBubbleEnrichedElementShape<3, 3>::shape(s, psi); }
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { BulkElementTetra3dC2::dshape_local(s, psi, dpsi); }
    void dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const { oomph::TBubbleEnrichedElementShape<3, 3>::dshape_local(s, psi, dpsi); }
    inline void d2shape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids, oomph::DShape &d2psids) const { throw_runtime_error("Implement"); }
    inline void shape(const oomph::Vector<double> &s, oomph::Shape &psi) const { oomph::TBubbleEnrichedElementShape<3, 3>::shape(s, psi); }
    inline void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const { oomph::TBubbleEnrichedElementShape<3, 3>::dshape_local(s, psi, dpsids); }
    inline void local_coordinate_of_node(const unsigned &j, oomph::Vector<double> &s) const { oomph::TBubbleEnrichedElementShape<3, 3>::local_coordinate_of_node(j, s); }
    bool has_bubble() const { return true; }
    virtual void set_integration_order(unsigned int order) { this->set_integration_scheme(integration_scheme_storage.get_integration_scheme(true, 3, order, true)); }
  };

  class PointElement0d : public virtual BulkElementBase, public virtual oomph::PointElement
  {
  protected:
  public:
    int nedges() const { return 0; }
    PointElement0d();
    virtual unsigned get_meshio_type_index() const { return 0; }
    virtual void dshape_local(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsids) const;
    virtual double invert_jacobian_mapping(const oomph::DenseMatrix<double> &jacobian, oomph::DenseMatrix<double> &inverse_jacobian) const;
    void build(oomph::Mesh *&, oomph::Vector<oomph::Node *> &, bool &, std::ofstream &) {}
    void check_integrity(double &max_error) { max_error = 0; }
    unsigned int get_node_index_C1_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C2TB_to_element(const unsigned int &i) const { return i; }
    unsigned int get_node_index_C1TB_to_element(const unsigned int &i) const { return i; }
    void output(std::ostream &outfile, const unsigned &n_plot) { BulkElementBase::output(outfile, n_plot); }
    void shape_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void shape_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi) const;
    void dshape_local_at_s_C1(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_C2(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    void dshape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const;
    unsigned nrecovery_order() { return 1; }
    unsigned nvertex_node() const { return 1; }
    oomph::Node *vertex_node_pt(const unsigned &j) const { return node_pt(j); }
    void further_setup_hanging_nodes() {};
    virtual BulkElementBase *create_son_instance() const
    {
      throw_runtime_error("Makes no sense");
      return NULL;
    }
    void pre_build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt)
    {
      BulkElementBase::pre_build(mesh_pt, new_node_pt);
    }
    int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nsubdiv, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const
    {
      nsubdiv = 1;
      return 1;
    }
    void fill_element_nodal_indices_for_numpy(int *indices, unsigned isubelem, bool tesselate_tri, std::vector<std::vector<std::set<oomph::Node *>>> &add_nodes) const;
    virtual std::vector<double> get_outline(bool lagrangian);
    bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    virtual double get_quality_factor() { return 1.0; }
    virtual double s_min() const
    {
      return 0.0;
    }

    virtual double s_max() const
    {
      return 0.0;
    }
    virtual void set_integration_order(unsigned int order) {}
  };

  /////////////////////////////

  class InterfaceElementBase : public virtual BulkElementBase, public virtual oomph::SolidFaceElement
  {
  protected:
    InterfaceElementBase *opposite_side;
    bool Is_internal_facet_opposite_dummy;
    std::vector<int> opposite_node_index;
    int opposite_orientation;
    std::vector<int> bulk_eqn_map, opp_interf_eqn_map, opp_bulk_eqn_map, bulk_bulk_eqn_map;
    std::vector<bool> external_data_is_geometric;

    virtual void update_in_external_fd(const unsigned &i);
    virtual bool add_required_ext_data(oomph::Data *data, bool is_geometric);
    virtual void add_required_external_data(JITFuncSpec_RequiredShapes_FiniteElement_t *required, BulkElementBase *from_elem);
    virtual int resolve_local_equation_for_external_contributions(long int globeq, BulkElementBase *from_elem = NULL, std::string *info = NULL);
    virtual void prepare_shape_buffer_for_integration(const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes, unsigned int flag);
    double fill_shape_info_at_s(const oomph::Vector<double> &s, const unsigned int &index, const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, double &JLagr, unsigned int flag, oomph::DenseMatrix<double> *dxds = NULL) const;
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap);
    virtual void ensure_external_data();
    virtual void assign_additional_local_eqn_numbers();
    virtual void fill_in_jacobian_from_lagragian_by_fd(oomph::Vector<double> &residuals, oomph::DenseMatrix<double> &jacobian);
    virtual void add_interface_dofs();
    virtual void fill_element_info_interface_part(bool without_equations=false);
    virtual std::vector<std::string> get_dof_names(bool not_a_root_call = false);
    virtual void get_dnormal_dcoords_at_s(const oomph::Vector<double> &s, double ***dnormal_dcoord, double *****d2normal_dcoord2) const;
    virtual void assign_additional_local_eqn_numbers_from_elem(const JITFuncSpec_RequiredShapes_FiniteElement_t *required, BulkElementBase *from_elem, std::vector<int> &eq_map);
    virtual oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const { throw_runtime_error("Implement"); }
    virtual void fill_opposite_node_indices(JITShapeInfo_t *shape_info)
    {
      for (unsigned int i = 0; i < opposite_node_index.size(); i++)
      {
        shape_info->opposite_node_index[i] = opposite_node_index[i];
      }
    }
    virtual void analyze_opposite_orientation(const std::vector<double> & offset) { throw_runtime_error("Implement"); }
    virtual void add_DG_external_data();
    virtual void interpolate_newly_constructed_additional_dof(const unsigned &lnode, const unsigned &valindex, const std::string &space);

  public:
    InterfaceElementBase() : opposite_side(NULL), Is_internal_facet_opposite_dummy(false) {}

    static bool interpolate_new_interface_dofs;

    virtual void set_remaining_shapes_appropriately(JITShapeInfo_t *shape_info, const JITFuncSpec_RequiredShapes_FiniteElement_t &required_shapes);
    void unpin_dummy_values(); // C1 fields on C2 elements have dummy values on only C2 nodes, which needs to be pinned
    void pin_dummy_values();
    void set_as_internal_facet_opposite_dummy() { Is_internal_facet_opposite_dummy = true; }
    bool is_internal_facet_opposite_dummy() const { return Is_internal_facet_opposite_dummy; }

    std::vector<int> get_attached_element_equation_mapping(const std::string &which);
    void set_opposite_interface_element(BulkElementBase *_opposite_side,std::vector<double>  offset)
    {
      if (_opposite_side && !dynamic_cast<InterfaceElementBase *>(_opposite_side))
      {
        throw_runtime_error("Can only set an Interface Element as the opposite side of and interface element");
      }
      opposite_side = dynamic_cast<InterfaceElementBase *>(_opposite_side);
      const JITFuncSpec_Table_FiniteElement_t *functable = this->codeinst->get_func_table();

      if (functable->merged_required_shapes.opposite_shapes)
      {
        // std::cout << "INTERFACE ELEM MERGED " << functable->merged_required_shapes.opposite_shapes->psi_D0 << std::endl;
        add_required_external_data(functable->merged_required_shapes.opposite_shapes, dynamic_cast<BulkElementBase *>(opposite_side));
        if (functable->merged_required_shapes.opposite_shapes->bulk_shapes)
        {
          //        std::cout << "INTERFACE ELEM MERGED BULK " <<  functable->merged_required_shapes.opposite_shapes->bulk_shapes->psi_D0 << std::endl;
          add_required_external_data(functable->merged_required_shapes.opposite_shapes->bulk_shapes, dynamic_cast<BulkElementBase *>(dynamic_cast<InterfaceElementBase *>(opposite_side)->bulk_element_pt()));
        }
      }

      this->eleminfo.opposite_eleminfo = &(opposite_side->eleminfo);
      std::vector<double> offs=offset;
      for (unsigned int i=offset.size();i<this->nodal_dimension();i++) offs.push_back(0.0);
      this->analyze_opposite_orientation(offs);
    }

    double zeta_nodal(const unsigned &n, const unsigned &k, const unsigned &i) const
    {
      return oomph::FaceElement::zeta_nodal(n, k, i);
    }

    virtual oomph::Vector<double> optimize_s_to_match_x(const oomph::Vector<double> &x);

    virtual pyoomph::Node *opposite_node_pt(unsigned int i)
    {
      if (!opposite_side || opposite_node_index[i] < 0)
        return NULL;
      return dynamic_cast<pyoomph::Node *>(opposite_side->node_pt(opposite_node_index[i]));
    }
    InterfaceElementBase *get_opposite_side() { return opposite_side; }
    const InterfaceElementBase *get_opposite_side() const { return opposite_side; }

    virtual int get_nodal_index_by_name(oomph::Node *n, std::string fieldname);
    virtual double get_interpolated_interface_field(const oomph::Vector<double> &s, const unsigned &ifindex, const std::string &space, const unsigned &t = 0) const;

    unsigned get_C2TB_buffer_index(const unsigned &fieldindex) override;
    unsigned get_C2_buffer_index(const unsigned &fieldindex) override;
    unsigned get_C1TB_buffer_index(const unsigned &fieldindex) override;
    unsigned get_C1_buffer_index(const unsigned &fieldindex) override;
    unsigned get_D2TB_buffer_index(const unsigned &fieldindex) override;
    unsigned get_D2_buffer_index(const unsigned &fieldindex) override;
    unsigned get_D1TB_buffer_index(const unsigned &fieldindex) override;
    unsigned get_D1_buffer_index(const unsigned &fieldindex) override;

    unsigned get_D2TB_node_index(const unsigned &fieldindex, const unsigned &nodeindex) const override;
    unsigned get_D2_node_index(const unsigned &fieldindex, const unsigned &nodeindex) const override;
    unsigned get_D1TB_node_index(const unsigned &fieldindex, const unsigned &nodeindex) const override;
    unsigned get_D1_node_index(const unsigned &fieldindex, const unsigned &nodeindex) const override;
    oomph::Data *get_D1_nodal_data(const unsigned &fieldindex) override;
    oomph::Data *get_D2_nodal_data(const unsigned &fieldindex) override;
    oomph::Data *get_D2TB_nodal_data(const unsigned &fieldindex) override;
    oomph::Data *get_D1TB_nodal_data(const unsigned &fieldindex) override;
    int get_D2TB_local_equation(const unsigned &fieldindex, const unsigned &nodeindex) override;
    int get_D2_local_equation(const unsigned &fieldindex, const unsigned &nodeindex) override;
    int get_D1TB_local_equation(const unsigned &fieldindex, const unsigned &nodeindex) override;
    int get_D1_local_equation(const unsigned &fieldindex, const unsigned &nodeindex) override;
  };

  template <class BASE>
  class InterfaceElement : public virtual BASE, public virtual InterfaceElementBase
  {
  protected:
    virtual bool fill_hang_info_with_equations(const JITFuncSpec_RequiredShapes_FiniteElement_t &required, JITShapeInfo_t *shape_info, int *eqn_remap)
    {
      bool res1 = BASE::fill_hang_info_with_equations(required, shape_info, eqn_remap);
      bool res2 = InterfaceElementBase::fill_hang_info_with_equations(required, shape_info, eqn_remap);
      return res1 || res2;
    }

    virtual void interpolate_hang_values()
    {
      BASE::interpolate_hang_values();
      this->interpolate_hang_values_at_interface();
    }

  public:
    double zeta_nodal(const unsigned &n, const unsigned &k, const unsigned &i) const
    {
      return oomph::FaceElement::zeta_nodal(n, k, i);
    }

    virtual void fill_element_info(bool without_equations=false)
    {
      BASE::fill_element_info(without_equations);
      this->fill_element_info_interface_part(without_equations);
      if (this->nnode())
      {
        oomph::TimeStepper *tstepper = this->node_pt(0)->time_stepper_pt();
        for (unsigned int i = 0; i < this->ninternal_data(); i++)
        {
          this->internal_data_pt(i)->set_time_stepper(tstepper, true);
        }
      }
    }

    InterfaceElement(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index)
    {
      bulk_el_pt->build_face_element(face_index, this);
      this->codeinst = jitcode;
      this->eleminfo.bulk_eleminfo = dynamic_cast<BulkElementBase *>(bulk_el_pt)->get_eleminfo();
      this->add_interface_dofs();
      const JITFuncSpec_Table_FiniteElement_t *functable = this->get_code_instance()->get_func_table();

      if (std::string(functable->dominant_space) == "C2")
      {
        const JITFuncSpec_Table_FiniteElement_t *bfunctable = dynamic_cast<BulkElementBase *>(bulk_el_pt)->get_code_instance()->get_func_table();
        if (std::string(bfunctable->dominant_space) == "C1")
        {
          throw_runtime_error("Cannot attach an interface element with C2 fields to a parent domain with max. C1 space");
        }
      }
      //      std::cout << "ADDING INTERFACE ELEM EXTERNAL DATA " << this->nexternal_data() << std::endl;
      this->flush_external_data();
      //      std::cout << "FLUSING EXTERNAL DATA " << this->nexternal_data() << std::endl;
      this->add_DG_external_data();
      //      std::cout << "DONE ADDING INTERFACE ELEM DG DATA " << this->nexternal_data() << std::endl;

      for (auto &e : this->codeinst->get_linked_external_data().get_required_external_data())
      {
        //        std:: cout << "ADDING ED0 " << std::endl;
        this->add_required_ext_data(e, false);
      }
      //      std::cout << "DONE ADDING INTERFACE ELEM ED0 DATA " << this->nexternal_data() << std::endl;

      if (functable->merged_required_shapes.bulk_shapes)
      {
        //	  std::cout << "ADDING BULK EXT DATA" << std::endl;
        add_required_external_data(functable->merged_required_shapes.bulk_shapes, dynamic_cast<BulkElementBase *>(bulk_el_pt)); // TODO: Also the others? (is it necessary e.g. spatial integration of the stress along interface)
        if (functable->merged_required_shapes.bulk_shapes->bulk_shapes)
        {
          InterfaceElementBase *ip = dynamic_cast<InterfaceElementBase *>(bulk_el_pt);
          add_required_external_data(functable->merged_required_shapes.bulk_shapes->bulk_shapes, dynamic_cast<BulkElementBase *>(ip->bulk_element_pt()));
        }
      }
    }

    virtual void get_normal_at_s(const oomph::Vector<double> &s, oomph::Vector<double> &n, double ***dnormal_dcoord, double *****d2normal_dcoord2) const
    {
      this->outer_unit_normal(s, n);
      if (dnormal_dcoord)
      {
        this->get_dnormal_dcoords_at_s(s, dnormal_dcoord, d2normal_dcoord2);
      }
    }
  };

  class InterfaceElementPoint0d : public virtual InterfaceElement<PointElement0d>
  {
  protected:
  public:
    InterfaceElementPoint0d(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<PointElement0d>(jitcode, bulk_el_pt, face_index)
    {
    }
    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
    {
      return s;
    }
    void analyze_opposite_orientation(const std::vector<double> & offset)
    {
      if (!dynamic_cast<InterfaceElementPoint0d *>(opposite_side))
      {
        throw_runtime_error("Can only connect an InterfaceElementPoint0d to an InterfaceElementPoint0d");
      }
      opposite_orientation = 0; // Does not matter anyhow
      opposite_node_index.resize(1, 0);
    }
  };

  class InterfaceElementLine1dC1 : public InterfaceElement<BulkElementLine1dC1>
  {
  protected:
    bool partial_opposite_internal_facet;
    double partial_opposite_s_at_smin, partial_opposite_s_at_smax;

  public:
    InterfaceElementLine1dC1(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementLine1dC1>(jitcode, bulk_el_pt, face_index), partial_opposite_internal_facet(false)
    {
    }

    virtual pyoomph::Node *opposite_node_pt(unsigned int i)
    {
      if (partial_opposite_internal_facet)
        throw_runtime_error("opposite_node_pt not allowed in internal facets with partial overlap with the opposite side");
      return InterfaceElement<BulkElementLine1dC1>::opposite_node_pt(i);
    }

    void analyze_opposite_orientation(const std::vector<double> & offset)
    {
      if (opposite_side->dim() != 1)
      {
        throw_runtime_error("Can only connect a 1d InterfaceElement to a 1d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }

      double dist0 = 0.0;
      double dist1 = 0.0;
      for (unsigned int i = 0; i < this->nvertex_node(); i++)
      {
        pyoomph::Node *nthis = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(i));
        pyoomph::Node *nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(i));
        for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
          dist0 += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
        nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(1 - i));
        for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
          dist1 += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
      }
      opposite_orientation = (dist0 < dist1 ? 0 : 1);
      /*      if (dynamic_cast<BulkTElementLine1dC1*>(opposite_side))
            {
             std::cout << "FOUND TRI OPPOSITE TO QUAD " << dist0 << "   " << dist1 << std::endl;
            }*/
      if ((dist0 < dist1 ? dist0 : dist1) > 1e-14)
      {
        if (!opposite_side->is_internal_facet_opposite_dummy())
        {
          throw_runtime_error("Vertex nodes are not matching here. This is only allowed for internal facets");
        }
        partial_opposite_internal_facet = true;
        oomph::Vector<double> x_at_smin(this->nodal_dimension(), 0.0), x_at_smax(this->nodal_dimension(), 0.0);
        this->interpolated_x(oomph::Vector<double>(1, this->s_min()), x_at_smin);
        this->interpolated_x(oomph::Vector<double>(1, this->s_max()), x_at_smax);
        partial_opposite_s_at_smin = opposite_side->optimize_s_to_match_x(x_at_smin)[0];
        partial_opposite_s_at_smax = opposite_side->optimize_s_to_match_x(x_at_smax)[0];
      }
      else
      {
        opposite_node_index.resize(2);
        if (opposite_side->nnode() == 2)
        {
          if (!opposite_orientation)
          {
            opposite_node_index[0] = 0;
            opposite_node_index[1] = 1;
          }
          else
          {
            opposite_node_index[0] = 1;
            opposite_node_index[1] = 0;
          }
        }
        else if (opposite_side->nnode() == 3)
        {
          if (!opposite_orientation)
          {
            opposite_node_index[0] = 0;
            opposite_node_index[1] = 2;
          }
          else
          {
            opposite_node_index[0] = 2;
            opposite_node_index[1] = 0;
          }
        }
        else
        {
          throw_runtime_error("Should not happen");
        }
      }
    }

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
    {
      if (partial_opposite_internal_facet)
      {
        double srel = (s[0] - this->s_min()) / (this->s_max() - this->s_min());
        srel = partial_opposite_s_at_smin + (partial_opposite_s_at_smax - partial_opposite_s_at_smin) * srel;
        return oomph::Vector<double>(1, srel);
      }
      else if (dynamic_cast<BulkTElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkTElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = (1 - res[0]) * 0.5;
          //          std::cout << "INFO OPPOSITE " << this->interpolated_x(s,0) << " vs " << opposite_side->interpolated_x(res,0) << "  s  " << s[0] << " vs " << res[0] <<  std::endl;
          return res;
        }
        else
        {
          oomph::Vector<double> res = s;
          res[0] = (res[0] + 1) * 0.5;
          //          std::cout << "INFO NONOPPOSITE " << this->interpolated_x(s,0) << " vs " << opposite_side->interpolated_x(res,0) <<  "  s  " << s[0] << " vs " << res[0] <<std::endl;
          return res;
        }
      }
      else if (dynamic_cast<BulkElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = -res[0];
          return res;
        }
        else
        {
          return s;
        }
      }
      else
      {
        throw_runtime_error("TODO");
      }
    }
  };

  class InterfaceElementLine1dC2 : public InterfaceElement<BulkElementLine1dC2>
  {
  protected:
    bool partial_opposite_internal_facet;
    double partial_opposite_s_at_smin, partial_opposite_s_at_smax;

  public:
    InterfaceElementLine1dC2(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementLine1dC2>(jitcode, bulk_el_pt, face_index), partial_opposite_internal_facet(false)
    {
    }

    /*   inline void assign_nodal_local_eqn_numbers(const bool &store_local_dof_pt)
      {
       oomph::SolidFiniteElement::assign_nodal_local_eqn_numbers(store_local_dof_pt);
    //   assign_hanging_local_eqn_numbers(store_local_dof_pt);
    //	 fill_element_info();
      }*/

    virtual pyoomph::Node *opposite_node_pt(unsigned int i)
    {
      if (partial_opposite_internal_facet)
        throw_runtime_error("opposite_node_pt not allowed in internal facets with partial overlap with the opposite side");
      return InterfaceElement<BulkElementLine1dC2>::opposite_node_pt(i);
    }

    //  void further_setup_hanging_nodes() {} //TODO: REM
    void analyze_opposite_orientation(const std::vector<double> & offset)
    {
      if (opposite_side->dim() != 1)
      {
        throw_runtime_error("Can only connect a 1d InterfaceElement to a 1d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }

      double dist0 = 0.0;
      double dist1 = 0.0;
      pyoomph::Node *nopp0 = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(0));
      pyoomph::Node *nopp1 = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(1));
      pyoomph::Node *nthis0 = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(0));
      pyoomph::Node *nthis1 = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(1));            
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist0 += (nthis0->x(k) - nopp0->x(k)+offset[k]) * (nthis0->x(k) - nopp0->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist0 += (nthis1->x(k) - nopp1->x(k)+offset[k]) * (nthis1->x(k) - nopp1->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist1 += (nthis1->x(k) - nopp0->x(k)+offset[k]) * (nthis1->x(k) - nopp0->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist1 += (nthis0->x(k) - nopp1->x(k)+offset[k]) * (nthis0->x(k) - nopp1->x(k)+offset[k]);
      opposite_orientation = (dist0 < dist1 ? 0 : 1);
      if ((dist0 < dist1 ? dist0 : dist1) > 1e-14)
      {
        if (!opposite_side->is_internal_facet_opposite_dummy())
        {
          throw_runtime_error("Vertex nodes are not matching here. This is only allowed for internal facets");
        }
        partial_opposite_internal_facet = true;
        oomph::Vector<double> x_at_smin(this->nodal_dimension(), 0.0), x_at_smax(this->nodal_dimension(), 0.0);
        this->interpolated_x(oomph::Vector<double>(1, this->s_min()), x_at_smin);
        this->interpolated_x(oomph::Vector<double>(1, this->s_max()), x_at_smax);
        partial_opposite_s_at_smin = opposite_side->optimize_s_to_match_x(x_at_smin)[0];
        partial_opposite_s_at_smax = opposite_side->optimize_s_to_match_x(x_at_smax)[0];
      }
      opposite_node_index.resize(3);
      if (opposite_side->nnode() == 3)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = 1;
          opposite_node_index[2] = 2;
        }
        else
        {
          opposite_node_index[0] = 2;
          opposite_node_index[1] = 1;
          opposite_node_index[2] = 0;
        }
      }
      else if (opposite_side->nnode() == 2)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = -1;
          opposite_node_index[2] = 1;
        }
        else
        {
          opposite_node_index[0] = 1;
          opposite_node_index[1] = -1;
          opposite_node_index[2] = 0;
        }
      }
      else
      {
        throw_runtime_error("Should not happen");
      }
      //    std::cout << "DISTS ARE " << dist0 << "  " << dist1 << " OPP ORIENT " << opposite_orientation << std::endl;
    }

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
    {
      if (partial_opposite_internal_facet)
      {
        double srel = (s[0] - this->s_min()) / (this->s_max() - this->s_min());
        srel = partial_opposite_s_at_smin + (partial_opposite_s_at_smax - partial_opposite_s_at_smin) * srel;
        return oomph::Vector<double>(1, srel);
      }
      else if (dynamic_cast<BulkTElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkTElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = (1 - res[0]) * 0.5;
          return res;
        }
        else
        {
          oomph::Vector<double> res = s;
          res[0] = (res[0] + 1) * 0.5;
          return res;
        }
      }
      else if (dynamic_cast<BulkElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = -res[0];
          return res;
        }
        else
        {
          return s;
        }
      }
      else
      {
        throw_runtime_error("TODO");
      }
    }
  };

  class InterfaceTElementLine1dC1 : public InterfaceElement<BulkTElementLine1dC1>
  {
  protected:
  public:
    InterfaceTElementLine1dC1(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkTElementLine1dC1>(jitcode, bulk_el_pt, face_index)
    {
    }
    void analyze_opposite_orientation(const std::vector<double> & offset)
    {
      if (opposite_side->dim() != 1)
      {
        throw_runtime_error("Can only connect a 1d InterfaceElement to a 1d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }

      double dist0 = 0.0;
      double dist1 = 0.0;
      for (unsigned int i = 0; i < this->nvertex_node(); i++)
      {
        pyoomph::Node *nthis = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(i));
        /*        for (unsigned int j = 0; j < opposite_side->nvertex_node(); j++)
                {*/
        pyoomph::Node *nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(i));
        for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
          dist0 += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
        nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(1 - i));
        for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
          dist1 += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
        //        }
      }
      if ((dist0 < dist1 ? dist0 : dist1) > 1e-14)
      {
        throw_runtime_error("Vertex nodes are not matching here");
      }
      opposite_orientation = (dist0 < dist1 ? 0 : 1);
      //      std::cout << "DISTS " << dist0 << "  " << dist1 << std::endl;
      opposite_node_index.resize(2);

      if (opposite_side->nnode() == 2)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = 1;
        }
        else
        {
          if (dynamic_cast<BulkElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkElementLine1dC2 *>(opposite_side))
          {
            opposite_node_index[0] = 0;
            opposite_node_index[1] = 1;
          }
          else
          {
            opposite_node_index[0] = 1;
            opposite_node_index[1] = 0;
          }
        }
      }
      else if (opposite_side->nnode() == 3)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = 2;
        }
        else
        {
          opposite_node_index[0] = 2;
          opposite_node_index[1] = 0;
        }
      }
      else
      {
        throw_runtime_error("Should not happen");
      }
    }

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
    {
      if (dynamic_cast<BulkTElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkTElementLine1dC2 *>(opposite_side))
      {
        //        std::cout << "LC IN OPP " << s[0] << " : " << opposite_orientation << std::endl;
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = 1 - res[0];
          return res;
        }
        else
        {
          return s;
        }
      }
      else if (dynamic_cast<BulkElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = 2 * (res[0] - 0.5);

          oomph::Vector<double> mycoord(2, 0);
          oomph::Vector<double> ocoord(2, 0);
          this->interpolated_x(s, mycoord);
          opposite_side->interpolated_x(res, ocoord);
          //   std::cout << "S CALC : " << s[0] << " " << res[0] << "  COORDS " << mycoord[0] << " , " << ocoord[0] << "    " << mycoord[1] << " , " << ocoord[1] <<std::endl;

          return res;
        }
        else
        {
          oomph::Vector<double> res = s;
          res[0] = -2 * (res[0] - 0.5);
          return res;
        }
      }
      else
      {
        throw_runtime_error("TODO");
      }
    }
  };

  class InterfaceTElementLine1dC2 : public InterfaceElement<BulkTElementLine1dC2>
  {
  protected:
  public:
    InterfaceTElementLine1dC2(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkTElementLine1dC2>(jitcode, bulk_el_pt, face_index)
    {
    }

    /*   inline void assign_nodal_local_eqn_numbers(const bool &store_local_dof_pt)
      {
       oomph::SolidFiniteElement::assign_nodal_local_eqn_numbers(store_local_dof_pt);
    //   assign_hanging_local_eqn_numbers(store_local_dof_pt);
    //	 fill_element_info();
      }*/

    //  void further_setup_hanging_nodes() {} //TODO: REM
    void analyze_opposite_orientation(const std::vector<double> & offset)
    {
      if (opposite_side->dim() != 1)
      {
        throw_runtime_error("Can only connect a 1d InterfaceElement to a 1d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }

      double dist0 = 0.0;
      double dist1 = 0.0;
      pyoomph::Node *nopp0 = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(0));
      pyoomph::Node *nopp1 = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(1));
      pyoomph::Node *nthis0 = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(0));
      pyoomph::Node *nthis1 = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(1));
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist0 += (nthis0->x(k) - nopp0->x(k)+offset[k]) * (nthis0->x(k) - nopp0->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist0 += (nthis1->x(k) - nopp1->x(k)+offset[k]) * (nthis1->x(k) - nopp1->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist1 += (nthis1->x(k) - nopp0->x(k)+offset[k]) * (nthis1->x(k) - nopp0->x(k)+offset[k]);
      for (unsigned int k = 0; k < std::min(nthis0->ndim(), nopp0->ndim()); k++)
        dist1 += (nthis0->x(k) - nopp1->x(k)+offset[k]) * (nthis0->x(k) - nopp1->x(k)+offset[k]);
      opposite_orientation = (dist0 < dist1 ? 0 : 1);
      if ((dist0 < dist1 ? dist0 : dist1) > 1e-14)
      {
        throw_runtime_error("Vertex nodes are not matching here");
      }
      opposite_node_index.resize(3);
      if (opposite_side->nnode() == 3)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = 1;
          opposite_node_index[2] = 2;
        }
        else
        {
          opposite_node_index[0] = 2;
          opposite_node_index[1] = 1;
          opposite_node_index[2] = 0;
        }
      }
      else if (opposite_side->nnode() == 2)
      {
        if (!opposite_orientation)
        {
          opposite_node_index[0] = 0;
          opposite_node_index[1] = -1;
          opposite_node_index[2] = 1;
        }
        else
        {
          opposite_node_index[0] = 1;
          opposite_node_index[1] = -1;
          opposite_node_index[2] = 0;
        }
      }
      else
      {
        throw_runtime_error("Should not happen");
      }
      //    std::cout << "DISTS ARE " << dist0 << "  " << dist1 << " OPP ORIENT " << opposite_orientation << std::endl;
    }

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
    {
      if (dynamic_cast<BulkTElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkTElementLine1dC2 *>(opposite_side))
      {
        //        std::cout << "LC IN OPP " << s[0] << " : " << opposite_orientation << std::endl;
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = 1 - res[0];
          return res;
        }
        else
        {
          return s;
        }
      }
      else if (dynamic_cast<BulkElementLine1dC1 *>(opposite_side) || dynamic_cast<BulkElementLine1dC2 *>(opposite_side))
      {
        if (opposite_orientation)
        {
          oomph::Vector<double> res = s;
          res[0] = -2 * (res[0] - 0.5);
          return res;
        }
        else
        {
          oomph::Vector<double> res = s;
          res[0] = 2 * (res[0] - 0.5);
          return res;
        }
      }
      else
      {
        throw_runtime_error("TODO");
      }
    }
  };

  class InterfaceElementQuad2dC1 : public InterfaceElement<BulkElementQuad2dC1>
  {
  protected:
  public:
    InterfaceElementQuad2dC1(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementQuad2dC1>(jitcode, bulk_el_pt, face_index)
    {
    }
  };

  class InterfaceElementQuad2dC2 : public InterfaceElement<BulkElementQuad2dC2>
  {
  protected:
  public:
    InterfaceElementQuad2dC2(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementQuad2dC2>(jitcode, bulk_el_pt, face_index)
    {
    }
  };

  class InterfaceElementTri2dC1 : public InterfaceElement<BulkElementTri2dC1>
  {
  protected:
  public:
    InterfaceElementTri2dC1(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementTri2dC1>(jitcode, bulk_el_pt, face_index)
    {
    }

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
    {
      oomph::Vector<double> res = s;
      if (opposite_orientation == 0)
      {
        res[0] = s[0];
        res[1] = s[1];
      }
      else if (opposite_orientation == 1)
      {
        res[0] = s[0];
        res[1] = 1 - s[0] - s[1];
      }
      else if (opposite_orientation == 2)
      {
        res[0] = s[1];
        res[1] = s[0];
      }
      else if (opposite_orientation == 3)
      {
        res[0] = 1 - s[0] - s[1];
        res[1] = s[0];
      }
      else if (opposite_orientation == 4)
      {
        res[0] = s[1];
        res[1] = 1 - s[0] - s[1];
      }
      else
      {
        res[0] = 1 - s[0] - s[1];
        res[1] = s[1];
      }
      return res;
    }

    void analyze_opposite_orientation(const std::vector<double> & offset)
    {
      if (opposite_side->dim() != 2)
      {
        throw_runtime_error("Can only connect a 2d InterfaceElement to a 2d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }
      std::vector<std::vector<int>> perms = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
      std::vector<double> pdists(perms.size(), 0.0);
      for (unsigned int i = 0; i < this->nvertex_node(); i++)
      {
        pyoomph::Node *nthis = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(i));
        for (unsigned int p = 0; p < perms.size(); p++)
        {
          pyoomph::Node *nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(perms[p][i]));
          for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
            pdists[p] += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
        }
      }
      double best_dist = pdists[0];
      opposite_orientation = 0;
      for (unsigned int p = 1; p < perms.size(); p++)
      {
        if (pdists[p] < best_dist)
        {
          best_dist = pdists[p];
          opposite_orientation = p;
        }
      }
      if (best_dist > 1e-14)
      {
        throw_runtime_error("Vertex nodes are not matching here");
      }
      opposite_node_index = perms[opposite_orientation]; // Making use of the fact that also for C2 opposite elements, the vertex nodes are at 0,1,2
    }
  };

  class InterfaceElementTri2dC2 : public InterfaceElement<BulkElementTri2dC2>
  {
  protected:
  public:
    InterfaceElementTri2dC2(DynamicBulkElementInstance *jitcode, FiniteElement *const &bulk_el_pt, const int &face_index) : InterfaceElement<BulkElementTri2dC2>(jitcode, bulk_el_pt, face_index)
    {
    }

    oomph::Vector<double> local_coordinate_in_opposite_side(const oomph::Vector<double> &s) const
    {
      oomph::Vector<double> res = s;
      if (opposite_orientation == 0)
      {
        res[0] = s[0];
        res[1] = s[1];
      }
      else if (opposite_orientation == 1)
      {
        res[0] = s[0];
        res[1] = 1 - s[0] - s[1];
      }
      else if (opposite_orientation == 2)
      {
        res[0] = s[1];
        res[1] = s[0];
      }
      else if (opposite_orientation == 3)
      {
        res[0] = 1 - s[0] - s[1];
        res[1] = s[0];
      }
      else if (opposite_orientation == 4)
      {
        res[0] = s[1];
        res[1] = 1 - s[0] - s[1];
      }
      else
      {
        res[0] = 1 - s[0] - s[1];
        res[1] = s[1];
      }
      return res;
      /*
      oomph::Vector<double> mypos(3,0.0);
      this->interpolated_x(s,mypos);
      oomph::Vector<double> opos(3,0.0);
      opposite_side->interpolated_x(res,opos);

      double diff=0.0;
      for (unsigned int i=0;i<3;i++) diff+=(mypos[i]-opos[i])*(mypos[i]-opos[i]);
      if (diff<1e-14) {
         std::cout << "PERM " <<  opposite_orientation << " SEEMS TO BE GOOD" << std::endl;
        return res;
      }

      std::cout << "DIFF " << diff << " AT PERM " << opposite_orientation << "  SIN " << s[0] << " " << s[1] << " SOUT " << res[0] << "  " << res[1] << std::endl;
      std::cout << "  MY POS " << mypos[0] << "  " << mypos[1] << "  " << mypos[2] << std::endl;
      std::cout << "  OT POS " << opos[0] << "  " << opos[1] << "  " << opos[2] << std::endl;

      oomph::DenseMatrix<double> my_t(2,3,0.0);
      oomph::Shape mypsi(this->nnode());
      oomph::DShape mydpsids(this->nnode(),2);
      this->dshape_local(s,mypsi,mydpsids);
      for(unsigned l=0;l<this->nnode();l++)
     {
       for(unsigned i=0;i<3;i++)
        {
         for(unsigned j=0;j<2;j++)
          {
           my_t(j,i) += this->nodal_position(l,i)*mydpsids(l,j);
          }
        }
     }
      std::cout << "  MY TANG1 " << my_t(0,0) << "  " << my_t(0,1) << "  " << my_t(0,2) << std::endl;
      std::cout << "  MY TANG2 " << my_t(1,0) << "  " << my_t(1,1) << "  " << my_t(1,2) << std::endl;

      oomph::DenseMatrix<double> ot_t(2,3,0.0);
      oomph::Shape otpsi(opposite_side->nnode());
      oomph::DShape otdpsids(opposite_side->nnode(),2);
      opposite_side->dshape_local(s,otpsi,otdpsids);
      for(unsigned l=0;l<opposite_side->nnode();l++)
     {
       for(unsigned i=0;i<3;i++)
        {
         for(unsigned j=0;j<2;j++)
          {
           ot_t(j,i) += opposite_side->nodal_position(l,i)*otdpsids(l,j);
          }
        }
     }
      std::cout << "  OT TANG1 " << ot_t(0,0) << "  " << ot_t(0,1) << "  " << ot_t(0,2) << std::endl;
      std::cout << "  OT TANG2 " << ot_t(1,0) << "  " << ot_t(1,1) << "  " << ot_t(1,2) << std::endl;

      return res;
      */
    }

    void analyze_opposite_orientation(const std::vector<double> & offset)
    {
      if (opposite_side->dim() != 2)
      {
        throw_runtime_error("Can only connect a 2d InterfaceElement to a 2d InterfaceElement");
      }
      if (this->nvertex_node() != opposite_side->nvertex_node())
      {
        throw_runtime_error("Can only connect InterfaceElements with same number of vertex nodes");
      }
      std::vector<std::vector<int>> perms = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
      std::vector<double> pdists(perms.size(), 0.0);
      for (unsigned int i = 0; i < this->nvertex_node(); i++)
      {
        pyoomph::Node *nthis = dynamic_cast<pyoomph::Node *>(this->vertex_node_pt(i));
        for (unsigned int p = 0; p < perms.size(); p++)
        {
          pyoomph::Node *nopp = dynamic_cast<pyoomph::Node *>(opposite_side->vertex_node_pt(perms[p][i]));
          for (unsigned int k = 0; k < std::min(nthis->ndim(), nopp->ndim()); k++)
            pdists[p] += (nthis->x(k) - nopp->x(k)+offset[k]) * (nthis->x(k) - nopp->x(k)+offset[k]);
        }
      }
      double best_dist = pdists[0];
      opposite_orientation = 0;
      for (unsigned int p = 1; p < perms.size(); p++)
      {
        if (pdists[p] < best_dist)
        {
          best_dist = pdists[p];
          opposite_orientation = p;
        }
      }
      if (best_dist > 1e-14)
      {
        throw_runtime_error("Vertex nodes are not matching here");
      }
      opposite_node_index = perms[opposite_orientation];
      opposite_node_index.resize(6, -1);
      if (opposite_side->nnode() > 3)
      {
        if (opposite_orientation == 1)
        { // 3 5 4
          opposite_node_index[3] = 5;
          opposite_node_index[4] = 4;
          opposite_node_index[5] = 3;
        }
        else if (opposite_orientation == 2)
        { // 4 3 5
          opposite_node_index[3] = 3;
          opposite_node_index[4] = 5;
          opposite_node_index[5] = 4;
        }
        else if (opposite_orientation == 5)
        { // 5 4 3, 4 5 3, 3 5 4, 5 3 4,
          opposite_node_index[3] = 4;
          opposite_node_index[4] = 3;
          opposite_node_index[5] = 5;
        }
        else
        {
          for (unsigned int k = 3; k < 6; k++)
          {
            opposite_node_index[k] = opposite_node_index[k - 3] + 3; // Seem to work
          }
        }
      }
      /*
          if (opposite_side->nnode()>3)
          {
           double test_dist=0.0;
           for (unsigned int i=0;i<this->nnode();i++)
           {
                  pyoomph::Node* nthis=dynamic_cast<pyoomph::Node*>(this->node_pt(i));
                  pyoomph::Node* nopp=dynamic_cast<pyoomph::Node*>(opposite_side->node_pt(opposite_node_index[i]));
                    for (unsigned int k=0;k<std::min(nthis->ndim(),nopp->ndim());k++) test_dist+=(nthis->x(k)-nopp->x(k))*(nthis->x(k)-nopp->x(k));
           }
           std::cout << "PERM IS " << opposite_orientation << "  BEST DIST " << best_dist << "  TEST DIST " << test_dist << std::endl;
           std::cout << "    OPPMAP IS "  << opposite_node_index[0] << "  "  << opposite_node_index[1] << "  "  << opposite_node_index[2] << "  "  << opposite_node_index[3] << "  "  << opposite_node_index[4] << "  "  << opposite_node_index[5] << std::endl;
           //std::cout << "    PDISTS ARE "  << pdists[0] << "  "  << pdists[1] << "  "  << pdists[2] << "  "  << pdists[3] << "  "  << pdists[4] << "  "  << pdists[5] << std::endl;
         }
      */
    }
  };

  extern double *__replace_RJM_by_param_deriv;
}
