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
#include "nodes.hpp"
#include "exception.hpp"
#include "ginac.hpp"
#include "lagr_error_estimator.hpp"
#include "kdtree.hpp"

namespace pyoomph
{

	class MeshTemplate;
	class MeshTemplateElementCollection;
	class Problem;
	class BulkElementBase;
	class DynamicBulkElementInstance;
	class MeshKDTree;

	class Mesh;
	

	

	class Mesh : public virtual oomph::RefineableMeshBase, public virtual oomph::Mesh
	{
	protected:
		Problem *problem;
		std::string domainname;
		std::vector<std::string> boundary_names;
		std::map<std::string, GiNaC::ex> initial_conditions;
		std::map<std::string, double> output_scales;
		std::map<std::string, unsigned> interface_dof_ids;
		std::vector<bool> dirichlet_active;
		std::map<pyoomph::Node *, pyoomph::Node *> copied_masters;
		MeshKDTree *lagrangian_kdtree;
		DynamicBulkElementInstance *codeinst = NULL;

	public:
		bool interpolated_lagrangian_coordinates_at_remeshing=false;
		virtual pyoomph::Node *resolve_copy_master(pyoomph::Node *cpy);
		std::vector<pyoomph::Node*> add_interpolated_nodes_at(const std::vector<std::vector<double> > & coords,bool all_as_boundary_nodes);
		virtual void store_copy_master(pyoomph::Node *cpy, pyoomph::Node *mst);
		virtual void _setup_information_from_old_mesh(Mesh *old);
		virtual void _save_state(std::vector<double> &meshdata);
		virtual void _load_state(const std::vector<double> &meshdata);
		virtual void pin_all_my_dofs(std::set<std::string> only_dofs, std::set<std::string> ignore_dofs, std::set<unsigned> ignore_continuous_at_interfaces);
		virtual void describe_global_dofs(std::vector<int> &doftype, std::vector<std::string> &typnames);
		virtual void describe_my_dofs(std::ostream &os, const std::string &in) { this->describe_local_dofs(os, in); }
		virtual void set_lagrangian_nodal_coordinates();
		// Function to activate the debugging.
		bool duarte_debug = false;
		virtual void activate_duarte_debug();
		// From the old mesh, map each element with the local coordinates associated to each integration point of the new mesh.
		virtual void prepare_zeta_interpolation(pyoomph::Mesh *oldmesh);
		virtual void set_time_level_for_projection(unsigned time_level);
		virtual void prepare_interpolation();
		virtual void nodal_interpolate_from(Mesh *from, int boundary_index);
		virtual void nodal_interpolate_along_boundary(Mesh *from, int bind, int oldbind, Mesh *imesh, Mesh *oldimesh, double boundary_max_dist);
		virtual void _set_problem(Problem *p, DynamicBulkElementInstance *code);
		std::vector<std::vector<double>> get_values_at_zetas(const std::vector<std::vector<double>> &zetas, std::vector<bool> &masked_lines, bool with_scales);
		virtual void fill_dof_types(int *typarr);
		virtual void ensure_halos_for_periodic_boundaries();
		virtual GiNaC::ex evaluate_integral_function(std::string name);
		virtual GiNaC::ex evaluate_extremum(std::string name,int sign,BulkElementBase *& extreme_element,oomph::Vector<double> &extreme_local_coords,unsigned flags);
		virtual std::vector<std::string> list_integral_functions();
		virtual std::vector<std::string> list_local_expressions();
		virtual void fill_internal_facet_buffers(std::vector<BulkElementBase *> &internal_elements, std::vector<int> &internal_face_dir, std::vector<BulkElementBase *> &opposite_elements, std::vector<int> &opposite_face_dir, std::vector<int> &opposite_already_at_index) { throw_runtime_error("Please specify this function for each dimension"); }
		virtual void generate_interface_elements(std::string intername, Mesh *imesh, DynamicBulkElementInstance *jitcode);
		virtual void ensure_external_data();
		virtual double get_temporal_error_norm_contribution();
		void set_output_scale(std::string fname, GiNaC::ex s, DynamicBulkElementInstance *_code);
		double get_output_scale(std::string fname);
		int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nelem, bool discontinuous); // Gets the number of required elemental indices
		void to_numpy(double *xbuffer, int *eleminds, unsigned elemstride, int *elemtypes, bool tesselate_tri, bool nondimensional, double *D0_data, double *DL_data, unsigned history_index, bool discontinuous);
		std::vector<double> evaluate_local_expression_at_nodes(unsigned index, bool nondimensional, bool discontinuous = false);
		void set_initial_condition(std::string fieldname, GiNaC::ex expression);
		virtual void setup_initial_conditions(bool resetting_first_step, std::string ic_name);
		virtual void pin_noncontributing_dofs();
		virtual void setup_Dirichlet_conditions(bool only_update_vals);
		virtual void set_dirichlet_active(std::string name, bool active);
		virtual bool get_dirichlet_active(std::string name);
		virtual void boundary_coordinates_bool(unsigned boundary_index);
		virtual bool is_boundary_coordinate_defined(unsigned boundary_index);
		void set_spatial_error_estimator_pt(pyoomph::LagrZ2ErrorEstimator *errest) { this->spatial_error_estimator_pt() = errest; }
		//  Mesh(Problem * p,MeshTemplate *templ, std::string domain);
		//	BulkNodeIterator  nodes() { return BulkNodeIterator(this);} //Iterate over all nodes
		//	NodalIteratorAccess  boundary_nodes(const  std::string & bn );  //Iterate over a boundary
		//	NodalIteratorAccess  boundary_nodes(const std::vector<std::string> & bn );  //Iterate over boundaries
		Problem *get_problem() { return problem; }
		Mesh() : problem(NULL),  lagrangian_kdtree(NULL) {}
		unsigned get_boundary_index(const std::string &n)
		{
			for (unsigned int i = 0; i < boundary_names.size(); i++)
				if (n == boundary_names[i])
					return i;
			std::ostringstream errmsg;
			errmsg << "Boundary '" << n << "' not in mesh. Available boundaries: " << std::endl;
			for (unsigned int i = 0; i < boundary_names.size(); i++)
				errmsg << "  " << boundary_names[i] << std::endl;
			throw_runtime_error(errmsg.str());
		}
		std::vector<std::string> get_boundary_names()
		{
			return boundary_names;
		}
		virtual int has_interface_dof_id(std::string n);		  //-1 if not present
		virtual unsigned resolve_interface_dof_id(std::string n); // add it if not present
		virtual unsigned count_nnode(bool discontinuous = false);
		virtual Node *get_some_node() { return (this->nnode() ? dynamic_cast<Node *>(this->node_pt(0)) : NULL); }
		virtual void fill_node_map(std::map<oomph::Node *, unsigned> &nodemap);
		virtual std::vector<oomph::Node *> fill_reversed_node_map(bool discontinuous = false);
		virtual void enlarge_elemental_error_max_override_to_only_nodal_connected_elems(unsigned bind);
		virtual unsigned get_nodal_dimension();
		virtual int get_element_dimension();
		virtual void invalidate_lagrangian_kdtree();
		virtual MeshKDTree *get_lagrangian_kdtree();
		virtual std::map<std::string, std::string> get_field_information(); // first: names, second: list of spaces (C2,C1,DL,D0), but also (../C2 etc for elements defined on bulk domains)
		virtual ~Mesh();
		virtual void check_integrity();
	};

	class DummyErrorEstimator : public oomph::Z2ErrorEstimator // Only be used to make sure that the error_estimator_pt is not NULL, which causes problems if PARANOID
	{
	};

	class InterfaceMesh : public Mesh
	{
	protected:
		DynamicBulkElementInstance *code;
		std::string interfacename;
		Mesh *bulkmesh;
		virtual void setup_boundary_information1d(pyoomph::Mesh *parent, const std::set<unsigned> &possible_bounds);
		virtual void setup_boundary_information2d(pyoomph::Mesh *parent, const std::set<unsigned> &possible_bounds);
		std::vector<double> opposite_offset_vector,reversed_opposite_offset_vector;
	public:
		InterfaceMesh();
		virtual ~InterfaceMesh();
		virtual void update_zeta_in_buffer();
		virtual void set_opposite_interface_offset_vector(const std::vector<double> & offset);
		virtual std::vector<double>  get_opposite_interface_offset_vector() {return opposite_offset_vector;}
		virtual void fill_internal_facet_buffers(std::vector<BulkElementBase *> &internal_elements, std::vector<int> &internal_face_dir, std::vector<BulkElementBase *> &opposite_elements, std::vector<int> &opposite_face_dir, std::vector<int> &opposite_already_at_index);
		std::vector<oomph::FiniteElement *> opposite_interior_facets;
		virtual double get_temporal_error_norm_contribution();
		virtual void adapt(const oomph::Vector<double> &elemental_error) {}
		virtual void refine_uniformly(oomph::DocInfo &doc_info) {}
		virtual unsigned unrefine_uniformly() { return 0; }
		virtual void clear_before_adapt();
		virtual void rebuild_after_adapt();
		virtual void nullify_selected_bulk_dofs();
		virtual void set_rebuild_information(Mesh *_bulkmesh, std::string intername, DynamicBulkElementInstance *jitcode);
		virtual Mesh *get_bulk_mesh() { return bulkmesh; }
		virtual unsigned count_nnode(bool discontinuous = false); // Interface meshes don't have their own nodes...
		virtual Node *get_some_node() { return (this->nelement() ? dynamic_cast<Node *>(dynamic_cast<oomph::FiniteElement *>(this->element_pt(0))->node_pt(0)) : NULL); }
		virtual void fill_node_map(std::map<oomph::Node *, unsigned> &nodemap);
		virtual std::vector<oomph::Node *> fill_reversed_node_map(bool discontinuous = false);
		virtual int has_interface_dof_id(std::string n) { return bulkmesh->has_interface_dof_id(n); }
		virtual unsigned resolve_interface_dof_id(std::string n) { return bulkmesh->resolve_interface_dof_id(n); }
		virtual void setup_boundary_information(pyoomph::Mesh *parent);
		virtual void connect_interface_elements_by_kdtree(InterfaceMesh *other);
		virtual unsigned get_nodal_dimension();
		virtual int get_element_dimension();
	};

	class ODEStorageMesh : public Mesh
	{
	protected:
		std::map<std::string, unsigned> name_to_index;

	public:
		ODEStorageMesh() : Mesh()
		{
			this->disable_adaptation();
			this->spatial_error_estimator_pt() = new DummyErrorEstimator();
		}
		virtual ~ODEStorageMesh()
		{
			this->Element_pt.clear(); // Keep the ODEs alive, they are killed by python
		}
		virtual double get_temporal_error_norm_contribution();
		virtual void adapt(const oomph::Vector<double> &elemental_error) {}
		virtual void refine_uniformly(oomph::DocInfo &doc_info) {}
		virtual unsigned unrefine_uniformly() { return 0; }
		virtual void setup_initial_conditions(bool resetting_first_step, std::string ic_name);
		virtual void setup_Dirichlet_conditions(bool only_update_vals);
		virtual unsigned add_ODE(std::string name, oomph::GeneralisedElement *ode);
		virtual oomph::GeneralisedElement *get_ODE(std::string name);
		virtual unsigned get_nodal_dimension() { return 0; }
		virtual int get_element_dimension() { return 0; }
	};

	class DynamicTree : public virtual oomph::Tree
	{
	protected:
	public:
		DynamicTree(oomph::RefineableElement *const &object_pt) : oomph::Tree(object_pt) {}
		DynamicTree(oomph::RefineableElement *const &object_pt, Tree *const &father_pt, const int &son_type) : oomph::Tree(object_pt, father_pt, son_type) { Level = father_pt->level() + 1; }

		typedef void (DynamicTree::*DynamicVoidMemberFctPt)();

		void dynamic_split_if_required();

		void dynamic_traverse_leaves(DynamicTree::DynamicVoidMemberFctPt member_function)
		{
			unsigned numsons = Son_pt.size();
			if (numsons > 0)
			{
				for (unsigned i = 0; i < numsons; i++)
				{
					dynamic_cast<pyoomph::DynamicTree *>(Son_pt[i])->dynamic_traverse_leaves(member_function);
				}
			}
			else
			{
				(this->*member_function)();
			}
		}
	};

	class DynamicTreeRoot : public virtual DynamicTree, public virtual oomph::TreeRoot
	{
	public:
		/// Broken copy constructor
		DynamicTreeRoot(const DynamicTreeRoot &dummy) : DynamicTree(NULL), oomph::TreeRoot(NULL)
		{
			oomph::BrokenCopy::broken_copy("DynamicTreeRoot");
		}

		/// Broken assignment operator
		void operator=(const DynamicTreeRoot &)
		{
			oomph::BrokenCopy::broken_assign("DynamicTreeRoot");
		}

		DynamicTreeRoot(oomph::RefineableElement *const &object_pt) : DynamicTree(object_pt), oomph::TreeRoot(object_pt)
		{
			Root_pt = this;
		}
	};

	// The basis class for all templated meshes // TODO Move elsewhere
	class TemplatedMeshBase : public virtual oomph::TreeBasedRefineableMeshBase, public virtual pyoomph::Mesh
	{

	protected:
		//  std::string domainname;
		//  std::vector<std::string> boundary_names;

		unsigned add_new_element(pyoomph::BulkElementBase *new_el, std::vector<pyoomph::Node *> nodes);

		void split_elements_if_required()
		{
			// Find the number of trees in the forest
			if (!this->Forest_pt)
			{
				throw_runtime_error("Trying to adapt a mesh with an unset tree forest");
			}
			unsigned n_tree = this->Forest_pt->ntree();
			// Loop over all "active" elements in the forest and split them
			// if required
			for (unsigned long e = 0; e < n_tree; e++)
			{
				dynamic_cast<pyoomph::DynamicTree *>(this->Forest_pt->tree_pt(e))->dynamic_traverse_leaves(&pyoomph::DynamicTree::dynamic_split_if_required);
			}
		}

		/// \short p-refine all the elements if required. Overload the template-free
		/// interface so that any temporary copies of the element that are created
		/// will be of the correct type.
		void p_refine_elements_if_required()
		{
			std::cerr << "Cannot p refine" << std::endl;
		}

	protected:
#ifdef OOMPH_HAS_MPI

		/// Additional setup of shared node scheme
		/// This is Required for reconcilliation of hanging nodes acrross processor
		/// boundaries when using elements with nonuniformly spaced nodes.
		/// ELEMENT template parameter is required so that
		/// MacroElementNodeUpdateNodes which are added as external halo master nodes
		/// can be made fully functional
		void additional_synchronise_hanging_nodes(
			const unsigned &ncont_interpolated_values);

#endif

	public:
		//	void set_spatial_error_estimator_pt(oomph::Z2ErrorEstimator * errest) {this->spatial_error_estimator_pt()=errest;}
		TemplatedMeshBase() : pyoomph::Mesh() {}
		//	Problem * get_problem() {return problem;}

		/// Broken copy constructor
		TemplatedMeshBase(const TemplatedMeshBase &dummy) : pyoomph::Mesh()
		{
			oomph::BrokenCopy::broken_copy("TemplatedMeshBase");
		}

		virtual void setup_interior_boundary_elements(unsigned bindex) {} // Tri meshes must add internal boundary elements by hand

		/// Broken assignment operator
		void operator=(const TemplatedMeshBase &)
		{
			oomph::BrokenCopy::broken_assign("TemplatedMeshBase");
		}

		virtual void generate_from_template(MeshTemplateElementCollection *coll) = 0;

		virtual std::vector<double> update_elemental_errors(std::vector<double> &errors)
		{
			return errors;
		}

		void adapt(const oomph::Vector<double> &elemental_error)
		{
			// For python, we need to convert it to a std::vector...
			std::vector<double> errors(elemental_error.size());
			for (unsigned int i = 0; i < elemental_error.size(); i++)
				errors[i] = elemental_error[i];
			errors = update_elemental_errors(errors);
			if (errors.size() != elemental_error.size())
			{
				throw_runtime_error("Mesh.update_elemental_errors may not change the size of the error vector");
			}
			oomph::Vector<double> updated_errors(elemental_error.size());
			for (unsigned int i = 0; i < elemental_error.size(); i++)
				updated_errors[i] = errors[i];
			TreeBasedRefineableMeshBase::adapt(updated_errors);
		}

		void prune_dead_nodes_without_respecting_boundaries()
		{
			oomph::Vector<oomph::Node*> new_node_pt;
    		unsigned long n_node = this->nnode();
    		for (unsigned long n = 0; n < n_node; n++)
			{	
				if (!(this->Node_pt[n]->is_obsolete()))
				{
					new_node_pt.push_back(this->Node_pt[n]);
				}
				else
				{
					delete this->Node_pt[n];
					this->Node_pt[n]=NULL;
				}
			}
			this->Node_pt = new_node_pt;
		}		
	};

	class MeshKDTree
	{
	protected:
		bool lagrangian;
		unsigned tindex;
		std::vector<pyoomph::Node *> nodes_by_index;
		std::map<pyoomph::Node *, std::set<pyoomph::BulkElementBase *>> nodes_to_elem;
		KDTree *tree;
		unsigned find_index(const std::vector<double> &coord, double *distreturn = NULL);
		double max_search_radius;

	public:
		MeshKDTree(pyoomph::Mesh *mesh, bool use_lagrangian, unsigned time_index);
		virtual ~MeshKDTree()
		{
			if (tree)
				delete tree;
		}
		pyoomph::Node *find_node(const oomph::Vector<double> &coord, double *distreturn = NULL);
		pyoomph::BulkElementBase *find_element(oomph::Vector<double> zeta, oomph::Vector<double> &sreturn);
	};

}
