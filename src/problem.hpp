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

#include "oomph_lib.hpp"
#include <vector>
#include <map>
#include "jitbridge.h"
#include "mesh.hpp"
#include <memory>
#include "hessian_tensor.hpp"

namespace pyoomph
{

  // class MeshTemplate;
  class Problem;
  class DynamicBulkElementInstance;
  class FiniteElementCode;
  class CCompiler;
  class DynamicBulkElementCode
  {
  protected:
    friend class Problem;
    friend class DynamicBulkElementInstance;
    Problem *problem;
    CCompiler *compiler;
    std::string filename;
    JITFuncSpec_Table_FiniteElement_t *functable;
    FiniteElementCode *element_class;
    std::map<std::string, int> integral_function_map,extremum_function_map;
    void *so_handle;

  public:
    const std::string &get_file_name() const { return filename; }
    DynamicBulkElementCode(Problem *prob, CCompiler *ccompiler, std::string fnam, FiniteElementCode *elem);
    virtual ~DynamicBulkElementCode();
    DynamicBulkElementInstance *factory_instance(pyoomph::Mesh *bulkmesh);
    const JITFuncSpec_Table_FiniteElement_t *get_func_table() const { return functable; }
    JITFuncSpec_Table_FiniteElement_t *get_func_table() { return functable; }
    int get_max_dt_order() const { return get_func_table()->max_dt_order; }
    int get_integral_function_index(std::string n);
    int get_extremum_function_index(std::string n);
    unsigned _set_solved_residual(std::string name);
    FiniteElementCode *get_code() { return element_class; }
  };

  // enum FieldSpace {C1=1, C2=2};

  class ExternalDataLink
  {
  public:
    oomph::Data *data;   // External data object
    int value_index;     // Value of this external data to be taken
    int elemental_index; // Index of the external_data_pt inside the element
    ExternalDataLink(oomph::Data *d, int v) : data(d), value_index(v), elemental_index(-1) {}
  };

  class ExternalDataLinkVector : public std::vector<ExternalDataLink>
  {
  protected:
    std::vector<oomph::Data *> elemental_data;

  public:
    ExternalDataLinkVector(unsigned n) : std::vector<ExternalDataLink>(n, ExternalDataLink(NULL, -1)) {}
    void reindex_elemental_data();
    std::vector<oomph::Data *> &get_required_external_data() { return elemental_data; }
  };

  

  // I.e. load one so -> Instantiate for different fields (with different paramsets)
  class DynamicBulkElementInstance
  {
  protected:
    friend class DynamicBulkElementCode;
    friend class Problem;
    friend class FiniteElementCode;
    friend class BulkElementBase;
    DynamicBulkElementCode *dyn;
    DynamicBulkElementInstance(DynamicBulkElementCode *d, pyoomph::Mesh *bm);
    //  std::vector<int> local_field_to_global_field_index_C1;
    //  std::vector<int> local_field_to_global_field_index_C2;
    //  std::vector<int> local_global_parameter_to_global_index;
    ExternalDataLinkVector linked_external_data;
    pyoomph::Mesh *bulkmesh; // Bulk mesh -> required to identify interface field indices
  public:
    std::set<int> nullify_bulk_residuals; // indices to nullify bulk residual contributions of continuous spaces at interface, negative is position dof
    void link_external_data(std::string name, oomph::Data *data, int index,std::string full_source_name);
    Problem *get_problem() { return dyn->problem; }
    pyoomph::Mesh *get_bulk_mesh() { return bulkmesh; }
    void set_bulk_mesh(pyoomph::Mesh *m) { bulkmesh = m; }
    FiniteElementCode *get_element_class() { return dyn->element_class; }
    void sanity_check();
    DynamicBulkElementCode *get_code() { return dyn; }
    const DynamicBulkElementCode *get_code() const { return dyn; }
    bool has_parameter_contribution(const std::string  & param);
    /*void bind_field(const std::string & internal_name,const std::string & global_name);
    void bind_field(unsigned int index,std::string name,FieldSpace space);
    void bind_field_C1(unsigned int index,std::string name) {bind_field(index,name,L1);}
    void bind_field_C2(unsigned int index,std::string name) {bind_field(index,name,L2);}*/
    //  void bind_global_parameter(const std::string & internal_name, const std::string & global_name);
    // unsigned int get_num_fields_C1() const {return dyn->functable->numfields_C1;}
    //  int get_global_field_index_C1(const unsigned int & i) const {return local_field_to_global_field_index_C1[i];}
    // unsigned int get_num_fields_C2() const {return dyn->functable->numfields_C2;}
    bool can_be_time_adaptive() const { return dyn->functable->has_temporal_estimators; }
    //  int get_global_field_index_C2(const unsigned int & i) const {return local_field_to_global_field_index_C2[i];}
    unsigned int get_num_global_parameters() const { return dyn->functable->numglobal_params; }
    //  int get_global_parameter_index(const unsigned int & i) const {return local_global_parameter_to_global_index[i];}
    const JITFuncSpec_Table_FiniteElement_t *get_func_table() const { return dyn->get_func_table(); }
    JITFuncSpec_Table_FiniteElement_t *get_func_table() { return dyn->get_func_table(); }
    int get_max_dt_order() const { return dyn->get_max_dt_order(); }
    int get_nodal_field_index(std::string name);
    std::map<std::string, unsigned> get_nodal_field_indices();
    std::map<std::string, unsigned> get_elemental_field_indices();
    int get_discontinuous_field_index(std::string name);
    int get_integral_function_index(std::string n) { return dyn->get_integral_function_index(n); }
    int get_extremum_function_index(std::string n) { return dyn->get_extremum_function_index(n); }
    unsigned resolve_interface_dof_id(std::string n);
    std::string get_space_of_field(std::string name);
    bool has_moving_nodes() {return dyn->functable->moving_nodes;}
    ExternalDataLinkVector &get_linked_external_data() { return linked_external_data; }
  };

  class Problem;
  /*
  class FieldDescriptor
  {
   protected:
    std::string name;
    FieldSpace space;
    unsigned int global_index;
    Problem * problem;
   public:
    FieldDescriptor(Problem * p,const std::string & n, const FieldSpace & s, const unsigned int & gi) : name(n), space(s), global_index(gi), problem(p) {}
    const std::string & get_name() const {return name;}
    const FieldSpace & get_space() const {return space;}
    const unsigned int & get_global_index() const {return global_index;}
  };
  */

  class GlobalParameterDescriptor
  {
  protected:
    std::string name;
    unsigned int global_index;
    Problem *problem;
    double Value;
    bool positive=false;
  public:
    GlobalParameterDescriptor(Problem *p, const std::string &n, const unsigned int &gi) : name(n), global_index(gi), problem(p), Value(0.0) {}
    const std::string &get_name() const { return name; }
    const unsigned int &get_global_index() const { return global_index; }
    double &value() { return Value; }
    const double &value() const { return Value; }
    void restrict_to_positive_values() {positive=true;}
    bool is_restricted_to_positive_values() const { return positive;}
    void set_analytic_derivative(bool active);
    bool get_analytic_derivative(); 
    Problem *get_problem() { return problem; }
  };
  
  class CustomResJacInformation
  {
   protected:
      friend class Problem;
      bool Require_jacobian;
      std::string dparameter;
      std::vector<double> residuals;
      oomph::Vector<double> Jvals;
      oomph::Vector<int> Jcolumn_index,Jrow_start;      
   public:
      bool require_jacobian() const {return Require_jacobian;}
      void set_custom_residuals(const std::vector<double> & r) {residuals=r;}
      void set_custom_jacobian(const std::vector<double> & Jv, const std::vector<int> & col_index,const std::vector<int> & row_start);      
      std::string get_parameter_name() const {return dparameter;}
      CustomResJacInformation(bool req_J,std::string parameter_name) : Require_jacobian(req_J), dparameter(parameter_name) {}
  };

  class DofAugmentations
  {
    protected:
      friend class pyoomph::Problem;
      Problem * problem;
      std::vector<unsigned> types,split_offsets;
      unsigned total_length;
      std::vector<std::vector<double>> augmented_vectors;
      std::vector<double> augmented_scalars;
      std::vector<std::string> augmented_parameters;
      bool finalized;
    public:
      DofAugmentations(Problem * _problem);
      unsigned add_vector(const std::vector<double> & v);
      unsigned add_scalar(const double & s);
      unsigned add_parameter(std::string);
      std::vector<std::vector<double>> split(unsigned startindex,int endindex);
      
  };

  // Problem class
  class Problem : public oomph::Problem
  {
  protected:
    CCompiler *compiler;
    std::ofstream * logfile;
    bool _is_quiet;
    friend class DynamicBulkElementInstance;
    //	 MeshTemplate *meshtemplate;
    std::vector<DynamicBulkElementCode *> bulk_element_codes;
    //   std::map<std::string,FieldDescriptor*> fields_by_name;
    //   std::vector<FieldDescriptor*> fields_by_index;
    std::map<std::string, GlobalParameterDescriptor *> global_params_by_name;
    std::vector<GlobalParameterDescriptor *> global_params_by_index;
    //   const FieldDescriptor * assert_field(const std::string & name,const FieldSpace & space );

    oomph::CRDoubleMatrix *eigen_JacobianMatrixPt, *eigen_MassMatrixPt;
    double global_temporal_error_norm();
    std::string bifurcation_tracking_mode = "";
    std::string _solved_residual = "";
    bool symmetric_hessian_assembly=true;
    
    void actions_after_change_in_global_parameter(double *const &parameter_pt) override;
    void actions_after_parameter_increase(double *const &parameter_pt) override;    

    double lambda_tracking_real = 0.0; // Real(lambda) for tracking of eigenbranches
    virtual void sparse_assemble_row_or_column_compressed_for_periodic_orbit(oomph::Vector<int*>& column_or_row_index,oomph::Vector<int*>& row_or_column_start,oomph::Vector<double*>& value,oomph::Vector<unsigned>& nnz,oomph::Vector<double*>& residuals,bool compressed_row_flag);
    void sparse_assemble_row_or_column_compressed(oomph::Vector<int*>& column_or_row_index,oomph::Vector<int*>& row_or_column_start,oomph::Vector<double*>& value,oomph::Vector<unsigned>& nnz,oomph::Vector<double*>& residual,bool compressed_row_flag) override;
    unsigned n_unaugmented_dofs=0;
    std::vector<std::string> defined_fields;
    std::vector<std::string> residual_names;
    std::vector<std::vector<bool>> residual_contributing_fields; // [residual][defined_field] -> whether the defined field contributes to the residual (i.e. whether there is a least a test function contributing to it)
    std::vector<std::vector<std::vector<bool>>> jacobian_contributing_fields; // [residual][defined_field][defined_field] -> whether the jacobian has a contribution from the derivative of the residual w.r.t. the defined field (i.e. whether there is at least a test function and trial function contributing to it)
    std::vector<std::vector<std::set<DynamicBulkElementCode*>>> residual_contributing_codes; // [residual][defined_field] -> Contributions to the residual from the different codes (e.g. bulk, interface, different domains)
    std::vector<std::vector<std::vector<std::set<DynamicBulkElementCode*>>>> jacobian_contributing_codes; // [residual][defined_field][defined_field] -> Contributions to the jacobian from the different codes (e.g. bulk, interface, different domains)
    std::vector<std::vector<bool>> pin_due_to_empty_jacobian_row; // [residual][defined_field] -> whether this field has to be pinned due to an empty jacobian row (i.e. no contributions on the current residual which leads to a non-invertible jacobian)
    std::vector<bool> removed_fields_due_to_missing_jacobian_row; // [defined_field] -> whether this field has been removed from the dofs due to missing jacobian row (i.e. no contributions on any residual, which leads to a non-invertible jacobian)
  public:
    void assemble_defined_field_list();
    std::string get_jacobian_information_string();
    void sparse_assemble_row_or_column_compressed_base_problem(oomph::Vector<int*>& column_or_row_index,oomph::Vector<int*>& row_or_column_start,oomph::Vector<double*>& value,oomph::Vector<unsigned>& nnz,oomph::Vector<double*>& residuals,bool compressed_row_flag);
    unsigned get_n_unaugmented_dofs() const {return n_unaugmented_dofs;}
    bool use_custom_residual_jacobian=false;
    bool improved_pitchfork_tracking_on_unstructured_meshes=false;
    bool is_field_removed_from_dofs_due_to_missing_jacobian_row(int global_field_index);

    double * get_lambda_tracking_real() {return &lambda_tracking_real;}

    void set_sparse_assembly_method(const std::string & method);
    std::string get_sparse_assembly_method();
    
    std::vector<DynamicBulkElementCode *> &get_bulk_element_codes() { return bulk_element_codes; }
    std::string get_bifurcation_tracking_mode() const { return bifurcation_tracking_mode; }
    std::vector<std::complex<double>> get_bifurcation_eigenvector();
    double get_bifurcation_omega();
    std::vector<double> get_arclength_dof_derivative_vector();
    std::vector<double> get_arclength_dof_current_vector();
    void update_dof_vectors_for_continuation(const std::vector<double> & ddof, const std::vector<double> & curr);
    void update_param_info_for_continuation(double dp,double p0);
    void set_dof_direction_arclength(std::vector<double> ddir);
    void get_dofs(oomph::DoubleVector& dofs) const  override {oomph::Problem::get_dofs(dofs);}
    void get_dofs(const unsigned& t, oomph::DoubleVector& dofs) const override;
    void set_dofs(const oomph::DoubleVector& dofs) override {oomph::Problem::set_dofs(dofs);}
    void set_dofs(const unsigned& t, oomph::DoubleVector& dof_pt) override;
    void set_dofs(const unsigned& t, oomph::Vector<double*>& dof_pt) override;
    void adapt(unsigned &n_refined, unsigned &n_unrefined) override
    {
      std::pair<unsigned, unsigned> res = this->_adapt();
      n_refined = res.first;
      n_unrefined = res.second;
    }
    std::vector<double> get_last_residual_convergence()
    {
      std::vector<double> res(Max_res.size());
      for (unsigned int i = 0; i < Max_res.size(); i++)
        res[i] = Max_res[i];
      return res;
    }
    virtual std::pair<unsigned, unsigned> _adapt()
    {
      unsigned n_refined = 0;
      unsigned n_unrefined = 0;
      //    std::cout <<"FAKE ADAPT" << std::endl;
      oomph::Problem::adapt(n_refined, n_unrefined);
      return std::make_pair(n_refined, n_unrefined);
    }
    //   MeshTemplate* get_mesh_template();
    DynamicBulkElementCode *load_dynamic_bulk_element_code(std::string dynamic_lib, FiniteElementCode *element_class);
    //   const FieldDescriptor * get_field(const std::string & name)  {return fields_by_name[name];}
    //   const FieldDescriptor * get_field(const unsigned int & index)  {return fields_by_index[index];}

    GlobalParameterDescriptor *assert_global_parameter(const std::string &name);
    GlobalParameterDescriptor *get_global_parameter(const std::string &name) { return global_params_by_name[name]; }
    GlobalParameterDescriptor *get_global_parameter(const unsigned int &index) { return global_params_by_index[index]; }
    int resolve_parameter_value_ptr(double *ptr);

    // const bool has_field(const std::string & name)  {return fields_by_name.count(name)>0;}
    const bool has_global_parameter(const std::string &name) { return global_params_by_name.count(name) > 0; }
    std::set<std::string> get_global_parameter_names()
    {
      std::set<std::string> res;
      for (auto const &pkeys : global_params_by_name)
        res.insert(pkeys.first);
      return res;
    }

    void ensure_dummy_values_to_be_dummy();
    virtual void actions_after_adapt();
    virtual void setup_pinning() {}
    virtual void set_initial_condition();
    virtual std::tuple<std::vector<double>, std::vector<bool>> get_current_dofs();
    virtual std::vector<double> get_history_dofs(unsigned t);
    virtual std::vector<double> get_current_pinned_values(bool with_pos);
    
    virtual void set_current_dofs(const std::vector<double> &inp);
    virtual void set_history_dofs(unsigned t, const std::vector<double> &inp);
    virtual void set_current_pinned_values(const std::vector<double> &inp,bool with_pos,unsigned t=0);
    virtual bool &always_take_one_newton_step() { return Always_take_one_newton_step; }
    virtual bool get_Keep_temporal_error_below_tolerance() { return Keep_temporal_error_below_tolerance; }
    virtual void set_Keep_temporal_error_below_tolerance(bool s) { Keep_temporal_error_below_tolerance = s; }
    /* 	 pyoomph::Mesh * mesh_pt()
       {
        return dynamic_cast<pyoomph::Mesh*>(oomph::Problem::mesh_pt());
       }*/

    double &newton_relaxation_factor() { return Relaxation_factor; }
    double &DTSF_max_increase_factor() { return DTSF_max_increase; }
    double &DTSF_min_decrease_factor() { return DTSF_min_decrease; }
    double &minimum_ds() { return Minimum_ds; }

    void set_mesh_pt(pyoomph::Mesh *mesh)
    {
      std::cout << "Setting mesh " << mesh << "  " << dynamic_cast<pyoomph::Mesh *>(mesh) << std::endl;
      oomph::Problem::mesh_pt() = mesh;
      std::cout << "Getting mesh " << oomph::Problem::mesh_pt() << "  " << dynamic_cast<pyoomph::Mesh *>(oomph::Problem::mesh_pt()) << "   " << mesh_pt() << std::endl;
    }

    // Expose these to be public
    virtual void actions_after_change_in_global_parameter(const std::string &  paramname) {}
    virtual void actions_after_parameter_increase(const  std::string &  paramname) {}
    virtual void actions_after_change_in_bifurcation_parameter() {}
    virtual void actions_before_newton_convergence_check() {}
    void reset_assembly_handler_to_default() override;
    virtual std::vector<double> get_parameter_derivative(const std::string param);
    void activate_my_fold_tracking(double *const &parameter_pt, const oomph::DoubleVector &eigenvector, const bool &block_solve);
    void activate_my_fold_tracking(double *const &parameter_pt, const bool &block_solve);
    void activate_my_pitchfork_tracking(double *const &parameter_pt, const oomph::DoubleVector &eigenvector, const bool &block_solve);
    void activate_my_hopf_tracking(double *const &parameter_pt, const double &omega, const oomph::DoubleVector &null_real, const oomph::DoubleVector &null_imag, const bool &block_solve);
    void activate_my_azimuthal_tracking(double *const &parameter_pt, const double &omega, const oomph::DoubleVector &null_real, const oomph::DoubleVector &null_imag, std::map<std::string, std::string> special_residual_forms);
    void set_arclength_parameter(std::string nam, double val);
    double arc_length_step(const std::string param, const double &ds, unsigned max_adapt);
    double get_arc_length_parameter_derivative() { return Parameter_derivative; }
    void set_arc_length_parameter_derivative(double dp) { Parameter_derivative=dp; }    
    void set_arc_length_theta_sqr(double thetasqr) {Theta_squared=thetasqr;}
    double get_arc_length_theta_sqr() {return Theta_squared;}    
    void start_bifurcation_tracking(const std::string param, const std::string typus, const bool &blocksolve, const std::vector<double> &eigenv1, const std::vector<double> &eigenv2, const double &omega, std::map<std::string, std::string> special_residual_forms);
    //void start_custom_augmented_system(oomph::AssemblyHandler *handler);
    void reset_augmented_dof_vector_to_nonaugmented();
    void add_augmented_dofs(DofAugmentations &aug);
    DofAugmentations * create_dof_augmentation() {return new DofAugmentations(this);}
    void start_orbit_tracking(const std::vector<std::vector<double>> &history, const double &T,int bspline_order,int gl_order,std::vector<double> knots,unsigned T_constraint_mode);
    void after_bifurcation_tracking_step();
    double &global_parameter(const std::string &n);

    // Required for custom bifurcation trackers
    oomph::Vector<double *> &GetDofPtr() { return this->Dof_pt; }
    oomph::LinearAlgebraDistribution *GetDofDistributionPt() { return this->Dof_distribution_pt; }
    oomph::Vector<oomph::Vector<unsigned>> &GetSparcseAssembleWithArraysPA() { return this->Sparse_assemble_with_arrays_previous_allocation; }
    
    virtual void quiet(bool _quiet);
    virtual bool _set_solved_residual(std::string name, bool raise_error=true,bool remove_dofs_without_jacobian_row=true);
    bool has_empty_jacobian_rows_marked() const;
    virtual void _replace_RJM_by_param_deriv(std::string name,bool active);
    virtual std::string _get_solved_residual() { return _solved_residual; }
    virtual bool is_quiet() const { return _is_quiet; }
    // void set_diagonal_zero_entries(bool yesno) {KeepZeroDiagonal=yesno;} //Requires a patched oomph-lib problem class
    Problem();
    virtual void unload_all_dlls();
    virtual unsigned get_max_dt_order() const;
    virtual ~Problem();
    virtual CCompiler *get_ccompiler() { return compiler; }
    virtual void set_ccompiler(CCompiler *comp) { compiler = comp; }
    virtual void assemble_eigenproblem_matrices(oomph::CRDoubleMatrix *&M, oomph::CRDoubleMatrix *&J, double sigma_r);
    
    virtual void  get_custom_residuals_jacobian(CustomResJacInformation * info)
    {
     throw_runtime_error("Must be implemented");
    }
    
    virtual void get_jacobian_by_elemental_assembly(oomph::DoubleVector &residuals,oomph::CRDoubleMatrix &jacobian) 
    {      
        oomph::Problem::get_jacobian(residuals,jacobian);
    }    
    virtual void get_residuals_by_elemental_assembly(oomph::DoubleVector &residuals)
    {
     oomph::Problem::get_residuals(residuals);
    }
    virtual void get_derivative_wrt_global_parameter_elemental_assembly(double* const& parameter_pt,oomph::DoubleVector &result)
    {
     oomph::Problem::get_derivative_wrt_global_parameter(parameter_pt,result);
    }
    virtual void get_residuals(oomph::DoubleVector &residuals);
    virtual void get_jacobian(oomph::DoubleVector &residuals,oomph::CRDoubleMatrix &jacobian);
    virtual void get_derivative_wrt_global_parameter(double* const& parameter_pt,oomph::DoubleVector& result);
    
    virtual SparseRank3Tensor assemble_hessian_tensor(bool symmetric);
    virtual std::vector<double> get_second_order_directional_derivative(std::vector<double> dir);
    
    virtual void set_symmetric_hessian_assembly(bool active) {symmetric_hessian_assembly=active;}
    virtual bool get_symmetric_hessian_assembly() const {return symmetric_hessian_assembly;}

    #ifndef OOMPH_HAS_MPI
    virtual void actions_before_distribute() {}
    virtual void actions_after_distribute() {}
    #endif
    
    void set_FD_step_used_in_get_hessian_vector_products(double step) {FD_step_used_in_get_hessian_vector_products=step;}
    
    //	 virtual bool _globally_convergent_newton_method() {return Use_globally_convergent_newton_method;} // Damn this private decl
    void open_log_file(const std::string & fname,const bool & activate_logging=true);

    void assemble_multiassembly(std::vector<std::string> what,std::vector<std::string> contributions,std::vector<std::string> params,std::vector<std::vector<double>> & hessian_vectors,std::vector<unsigned> & hessian_vector_indices,std::vector<std::vector<double>> & data,std::vector<std::vector<int>> &csrdata,unsigned & ndof,std::vector<int> & return_indices);
    
  };

  void RequiredShapes_merge(JITFuncSpec_RequiredShapes_FiniteElement_t *src, JITFuncSpec_RequiredShapes_FiniteElement_t *dest);
  void RequiredShapes_free(JITFuncSpec_RequiredShapes_FiniteElement_t *p);
}
