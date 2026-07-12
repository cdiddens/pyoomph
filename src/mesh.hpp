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




	// Base class for all pyoomph meshes. Wraps an oomph-lib (Refineable)Mesh and adds
	// everything pyoomph needs on top: named boundaries, per-field initial/Dirichlet
	// conditions, output scaling, interface-dof bookkeeping, JIT-compiled element code
	// binding (codeinst) and helpers for interpolation/projection between meshes.
	class Mesh : public virtual oomph::RefineableMeshBase, public virtual oomph::Mesh
	{
	protected:
		Problem *problem; // Owning Problem (non-owning pointer)
		std::string domainname; // Name of the domain this mesh represents, e.g. as used in the Python interface
		std::vector<std::string> boundary_names; // Names of the boundaries, indexed like oomph-lib's boundary indices
		std::map<std::string, GiNaC::ex> initial_conditions; // Symbolic initial condition expressions, keyed by field/IC name
		std::map<std::string, double> output_scales; // Nondimensionalization output scales, keyed by field name
		std::map<std::string, unsigned> interface_dof_ids; // Names of interface-only degrees of freedom mapped to their global index
		std::vector<bool> dirichlet_active; // Whether each Dirichlet condition (by index) is currently active
		std::map<pyoomph::Node *, pyoomph::Node *> copied_masters; // Maps a copied node to its master node (see resolve_copy_master)
		MeshKDTree *lagrangian_kdtree; // Lazily-built KD-tree over Lagrangian node coordinates, used for spatial lookups
		DynamicBulkElementInstance *codeinst = NULL; // JIT-compiled element code instance backing the elements of this mesh

	public:
		bool interpolated_lagrangian_coordinates_at_remeshing=false;
		// Given a node that is a "copy" (e.g. periodic/interface copy) of another node, return its master.
		// Follows copied_masters; typically returns cpy itself if it is not a copy.
		virtual pyoomph::Node *resolve_copy_master(pyoomph::Node *cpy);
		// Create new nodes at the given (Eulerian or Lagrangian, depending on caller) coordinates by
		// interpolating from the elements they fall into; used e.g. for probes/interpolation meshes.
		// If all_as_boundary_nodes is set, all created nodes are added as BoundaryNodes (needed if they may later be put on a boundary).
		std::vector<pyoomph::Node*> add_interpolated_nodes_at(const std::vector<std::vector<double> > & coords,bool all_as_boundary_nodes);
		// Register mst as the master node for the copy cpy (see copied_masters / resolve_copy_master).
		virtual void store_copy_master(pyoomph::Node *cpy, pyoomph::Node *mst);
		// Transfer mesh-level bookkeeping (boundary names, ICs, output scales, ...) from an old mesh,
		// e.g. after remeshing/adaptation has created a fresh mesh object.
		virtual void _setup_information_from_old_mesh(Mesh *old);
		// Serialize all data required to restore this mesh's nodal state (positions, values, history) into meshdata.
		virtual void _save_state(std::vector<double> &meshdata);
		// Restore nodal state previously written by _save_state.
		virtual void _load_state(const std::vector<double> &meshdata);
		// Pin all dofs of this mesh's elements, optionally restricted to only_dofs / excluding ignore_dofs.
		// ignore_continuous_at_interfaces lists dof indices that must stay unpinned where continuous across an interface.
		virtual void pin_all_my_dofs(std::set<std::string> only_dofs, std::set<std::string> ignore_dofs, std::set<unsigned> ignore_continuous_at_interfaces);
		// Fill doftype/typnames with a description of each global dof (used for debugging/introspection).
		virtual void describe_global_dofs(std::vector<int> &doftype, std::vector<std::string> &typnames);
		virtual void describe_my_dofs(std::ostream &os, const std::string &in) { this->describe_local_dofs(os, in); }
		// Copy the current nodal Eulerian coordinates into the Lagrangian coordinates of all nodes (i.e. "freeze" the current shape as reference configuration).
		virtual void set_lagrangian_nodal_coordinates();
		// Function to activate the debugging.
		bool duarte_debug = false;
		virtual void activate_duarte_debug();
		// Fill a lookup buffer mapping each local dof to its global field index, used to assemble field-wise data.
		virtual void fill_dof_to_global_field_index_buffer(std::vector<int> &dofs_to_global_field_index);
		// From the old mesh, map each element with the local coordinates associated to each integration point of the new mesh.
		virtual void prepare_zeta_interpolation(pyoomph::Mesh *oldmesh);
		virtual void set_time_level_for_projection(unsigned time_level);
		// Prepare internal caches (e.g. KD-tree) required before interpolation calls; must be called before nodal_interpolate_*.
		virtual void prepare_interpolation();
		// Interpolate nodal values of this mesh from the mesh "from". If boundary_index>=0, only nodes on that
		// boundary are interpolated (used when remeshing only a boundary/interface region).
		virtual void nodal_interpolate_from(Mesh *from, int boundary_index);
		// Interpolate nodal values along a boundary from an old mesh, using the arclength-like boundary coordinate
		// to find the closest correspondence; boundary_max_dist limits how far a match may be to still be accepted.
		virtual void nodal_interpolate_along_boundary(Mesh *from, int bind, int oldbind, Mesh *imesh, Mesh *oldimesh, double boundary_max_dist);
		// Bind this mesh to its owning Problem and the JIT-compiled element code instance used to create its elements.
		virtual void _set_problem(Problem *p, DynamicBulkElementInstance *code);
		// Evaluate all fields at a list of local coordinates ("zetas") per element; masked_lines flags entries to
		// skip. with_scales selects whether output_scales are applied (dimensional vs. nondimensional output).
		std::vector<std::vector<double>> get_values_at_zetas(const std::vector<std::vector<double>> &zetas, std::vector<bool> &masked_lines, bool with_scales);
		virtual void fill_dof_types(int *typarr);
		// Make sure the halo layer (MPI-distributed meshes) is wide enough to represent periodic boundary partners.
		virtual void ensure_halos_for_periodic_boundaries();
		// Evaluate a named user-defined integral (as set up in the JIT code) over this mesh.
		virtual GiNaC::ex evaluate_integral_function(std::string name);
		// Find the extremum (sign>0: max, sign<0: min) of a named local expression over the mesh; returns the
		// value and, via out-params, the element and local coordinate where it is attained.
		virtual GiNaC::ex evaluate_extremum(std::string name,int sign,BulkElementBase *& extreme_element,oomph::Vector<double> &extreme_local_coords,unsigned flags);
		virtual std::vector<std::string> list_integral_functions();
		virtual std::vector<std::string> list_local_expressions();
		// Fill buffers describing internal facets (element faces shared between two bulk elements) and, for
		// periodic/interface meshes, their opposite-side counterparts. Dimension-specific; base class throws.
		virtual void fill_internal_facet_buffers(std::vector<BulkElementBase *> &internal_elements, std::vector<int> &internal_face_dir, std::vector<BulkElementBase *> &opposite_elements, std::vector<int> &opposite_face_dir, std::vector<int> &opposite_already_at_index) { throw_runtime_error("Please specify this function for each dimension"); }
		// Build an InterfaceMesh of elements attached to the boundary/interface named intername, using the
		// JIT-compiled interface element code jitcode; imesh is the interface mesh to populate.
		virtual void generate_interface_elements(std::string intername, Mesh *imesh, DynamicBulkElementInstance *jitcode);
		virtual void ensure_external_data();
		virtual double get_temporal_error_norm_contribution();
		// Store the output scale factor s (a symbolic expression, evaluated via _code) used to nondimensionalize field fname on output.
		void set_output_scale(std::string fname, GiNaC::ex s, DynamicBulkElementInstance *_code);
		double get_output_scale(std::string fname);
		int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nelem, bool discontinuous); // Gets the number of required elemental indices
		// Fill flat buffers with node coordinates and per-element connectivity/type info for numpy-based plotting/export.
		// tesselate_tri splits quads/general elements into triangles; discontinuous keeps per-element (DG-style) node copies.
		void to_numpy(double *xbuffer, int *eleminds, unsigned elemstride, int *elemtypes, bool tesselate_tri, bool nondimensional, double *D0_data, double *DL_data, unsigned history_index, bool discontinuous);
		std::vector<double> evaluate_local_expression_at_nodes(unsigned index, bool nondimensional, bool discontinuous = false);
		// Register a symbolic initial-condition expression for field fieldname (evaluated later in setup_initial_conditions).
		void set_initial_condition(std::string fieldname, GiNaC::ex expression);
		// Evaluate and assign the registered initial conditions to all nodes/elements of this mesh.
		// resetting_first_step distinguishes a true (re)start from a mid-simulation IC re-application (e.g. after remeshing);
		// ic_name selects a particular named IC set (multiple ICs can be registered under different names).
		virtual void setup_initial_conditions(bool resetting_first_step, std::string ic_name);
		// Pin dofs that do not actually contribute to any equation (e.g. because they were disabled by requirements),
		// so the linear system doesn't end up singular.
		virtual void pin_noncontributing_dofs();
		// Evaluate and (re-)apply all Dirichlet conditions on this mesh. If only_update_vals is set, only the pinned
		// values are refreshed (e.g. for time-dependent BCs), without changing which dofs are pinned.
		virtual void setup_Dirichlet_conditions(bool only_update_vals);
		virtual void set_dirichlet_active(std::string name, bool active);
		virtual bool get_dirichlet_active(std::string name);
		// Ensure the intrinsic boundary coordinate (arclength/zeta along boundary_index) has been set up on all its nodes.
		virtual void boundary_coordinates_bool(unsigned boundary_index);
		virtual bool is_boundary_coordinate_defined(unsigned boundary_index);
		void set_spatial_error_estimator_pt(pyoomph::LagrZ2ErrorEstimator *errest) { this->spatial_error_estimator_pt() = errest; }
		//  Mesh(Problem * p,MeshTemplate *templ, std::string domain);
		//	BulkNodeIterator  nodes() { return BulkNodeIterator(this);} //Iterate over all nodes
		//	NodalIteratorAccess  boundary_nodes(const  std::string & bn );  //Iterate over a boundary
		//	NodalIteratorAccess  boundary_nodes(const std::vector<std::string> & bn );  //Iterate over boundaries
		Problem *get_problem() { return problem; }
		Mesh() : problem(NULL),  lagrangian_kdtree(NULL) {}
		// Look up the oomph-lib boundary index for a named boundary; throws with a helpful message (listing
		// all available boundary names) if n is not found.
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
		// Discard the cached lagrangian_kdtree (e.g. because nodes moved); it will be rebuilt lazily on next use.
		virtual void invalidate_lagrangian_kdtree();
		// Lazily build (if necessary) and return the KD-tree over Lagrangian node coordinates.
		virtual MeshKDTree *get_lagrangian_kdtree();
		virtual std::map<std::string, std::string> get_field_information(); // first: names, second: list of spaces (C2,C1,DL,D0), but also (../C2 etc for elements defined on bulk domains)
		virtual ~Mesh();
		virtual void check_integrity();
	};

	class DummyErrorEstimator : public oomph::Z2ErrorEstimator // Only be used to make sure that the error_estimator_pt is not NULL, which causes problems if PARANOID
	{
	};

	// A mesh of interface/facet elements attached to a boundary of a bulk mesh (e.g. free surface,
	// contact line, or coupling between two bulk domains). Has no nodes of its own beyond those it
	// shares with the underlying bulk mesh(es); its elements wrap bulk element faces.
	class InterfaceMesh : public Mesh
	{
	protected:
		DynamicBulkElementInstance *code; // JIT-compiled code for the interface elements
		std::string interfacename;
		Mesh *bulkmesh; // The bulk mesh this interface is attached to
		// Build boundary information (which interface elements/nodes lie on which sub-boundary) for a 1d interface
		// (i.e. attached to a 2d bulk mesh), restricted to boundary indices in possible_bounds.
		virtual void setup_boundary_information1d(pyoomph::Mesh *parent, const std::set<unsigned> &possible_bounds);
		// Same as setup_boundary_information1d but for a 2d interface (attached to a 3d bulk mesh).
		virtual void setup_boundary_information2d(pyoomph::Mesh *parent, const std::set<unsigned> &possible_bounds);
		std::vector<double> opposite_offset_vector,reversed_opposite_offset_vector; // Constant offset (e.g. for periodic/translated interfaces) to the opposite side and its reverse
	public:
		InterfaceMesh();
		virtual ~InterfaceMesh();
		virtual void update_zeta_in_buffer();
		virtual void update_equation_remapping();
		// Set the offset vector used to relate this interface's coordinates to the geometrically opposite interface
		// (e.g. across a periodic domain); also updates the cached reversed_opposite_offset_vector.
		virtual void set_opposite_interface_offset_vector(const std::vector<double> & offset);
		virtual std::vector<double>  get_opposite_interface_offset_vector() {return opposite_offset_vector;}
		virtual void fill_internal_facet_buffers(std::vector<BulkElementBase *> &internal_elements, std::vector<int> &internal_face_dir, std::vector<BulkElementBase *> &opposite_elements, std::vector<int> &opposite_face_dir, std::vector<int> &opposite_already_at_index);
		std::vector<oomph::FiniteElement *> opposite_interior_facets; // Facets on the geometrically opposite side (e.g. periodic partner), matched to this mesh's facets by index
		virtual double get_temporal_error_norm_contribution();
		virtual void adapt(const oomph::Vector<double> &elemental_error) {}
		virtual void refine_uniformly(oomph::DocInfo &doc_info) {}
		virtual unsigned unrefine_uniformly() { return 0; }
		// Undo prior interface-element rebuild state before the bulk mesh is adapted (interface elements are
		// rebuilt afterwards from the new bulk mesh via rebuild_after_adapt).
		virtual void clear_before_adapt();
		// Regenerate this interface's elements from the (possibly refined/coarsened) bulk mesh after adaptation.
		virtual void rebuild_after_adapt();
		// Unpin/null out dofs on this interface that must not be treated as independent (e.g. because they are
		// algebraically slaved to the bulk mesh across the interface).
		virtual void nullify_selected_bulk_dofs();
		// Store the information (bulk mesh, interface name, JIT code) needed to rebuild this interface mesh later,
		// e.g. after adaptation via rebuild_after_adapt.
		virtual void set_rebuild_information(Mesh *_bulkmesh, std::string intername, DynamicBulkElementInstance *jitcode);
		virtual Mesh *get_bulk_mesh() { return bulkmesh; }
		virtual unsigned count_nnode(bool discontinuous = false); // Interface meshes don't have their own nodes...
		virtual Node *get_some_node() { return (this->nelement() ? dynamic_cast<Node *>(dynamic_cast<oomph::FiniteElement *>(this->element_pt(0))->node_pt(0)) : NULL); }
		virtual void fill_node_map(std::map<oomph::Node *, unsigned> &nodemap);
		virtual std::vector<oomph::Node *> fill_reversed_node_map(bool discontinuous = false);
		virtual int has_interface_dof_id(std::string n) { return bulkmesh->has_interface_dof_id(n); }
		virtual unsigned resolve_interface_dof_id(std::string n) { return bulkmesh->resolve_interface_dof_id(n); }
		virtual void setup_boundary_information(pyoomph::Mesh *parent);
		// Match this interface mesh's elements/nodes to those of another interface mesh (e.g. the opposite side of
		// a periodic domain) by nearest-neighbor lookup via a KD-tree, populating opposite_interior_facets etc.
		virtual void connect_interface_elements_by_kdtree(InterfaceMesh *other);
		virtual unsigned get_nodal_dimension();
		virtual int get_element_dimension();
	};

	// A "mesh" that does not discretize a spatial domain but instead stores ODE (0-dimensional)
	// elements, i.e. degrees of freedom that are not associated with any node/geometry (global ODEs).
	class ODEStorageMesh : public Mesh
	{
	protected:
		std::map<std::string, unsigned> name_to_index; // Maps ODE name to its index in Element_pt

	public:
		ODEStorageMesh();
		virtual ~ODEStorageMesh();
		virtual double get_temporal_error_norm_contribution();
		virtual void adapt(const oomph::Vector<double> &elemental_error) {}
		virtual void refine_uniformly(oomph::DocInfo &doc_info) {}
		virtual unsigned unrefine_uniformly() { return 0; }
		virtual void setup_initial_conditions(bool resetting_first_step, std::string ic_name);
		virtual void setup_Dirichlet_conditions(bool only_update_vals);
		// Register a new named ODE (GeneralisedElement) in this storage mesh; returns its index.
		virtual unsigned add_ODE(std::string name, oomph::GeneralisedElement *ode);
		// Look up a previously added ODE element by name.
		virtual oomph::GeneralisedElement *get_ODE(std::string name);
		virtual unsigned get_nodal_dimension() { return 0; }
		virtual int get_element_dimension() { return 0; }
		virtual oomph::GeneralisedElement *_create_ode_element(oomph::TimeStepper *ts);
	};

	// pyoomph's replacement for oomph-lib's Tree that supports "dynamic" (JIT-compiled) elements:
	// tree traversal needs to go through pyoomph's element/mesh types rather than the statically
	// templated oomph-lib element hierarchy.
	class DynamicTree : public virtual oomph::Tree
	{
	protected:
	public:
		DynamicTree(oomph::RefineableElement *const &object_pt) : oomph::Tree(object_pt) {}
		DynamicTree(oomph::RefineableElement *const &object_pt, Tree *const &father_pt, const int &son_type) : oomph::Tree(object_pt, father_pt, son_type) { Level = father_pt->level() + 1; }

		typedef void (DynamicTree::*DynamicVoidMemberFctPt)();

		// Split (refine) this tree's associated element if oomph-lib's refinement flags request it.
		void dynamic_split_if_required();

		// Recursively call member_function on every leaf (element without sons) of the subtree rooted here.
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

	// Root node of a DynamicTree (one per top-level element in the forest).
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
	// Common base for meshes that are generated from a MeshTemplate (i.e. all "real" spatial meshes,
	// as opposed to InterfaceMesh/ODEStorageMesh). Adds the tree-based refinement machinery (via
	// oomph::TreeBasedRefineableMeshBase) and facet bookkeeping used to (re)derive boundary information.
	class TemplatedMeshBase : public virtual oomph::TreeBasedRefineableMeshBase, public virtual pyoomph::Mesh
	{

	protected:
		//  std::string domainname;
		//  std::vector<std::string> boundary_names;
		std::map<std::set<pyoomph::Node *> ,std::vector<unsigned>> facets; // Map from facets (vertex node sets) to boundary indices
		// Create and register a new element of the same JIT-compiled type as new_el's prototype, built on the
		// given node list; used when subdividing/regenerating elements (e.g. triangle refinement).
		unsigned add_new_element(pyoomph::BulkElementBase *new_el, std::vector<pyoomph::Node *> nodes);
		// (Re)build the facets map from the MeshTemplate's boundary information, using bound_map to translate
		// template boundary indices to this mesh's boundary indices.
        virtual void setup_facets_from_template(MeshTemplate *templ,const std::vector<int> & bound_map);
		// Traverse the tree forest and p/h-refine any leaf element that oomph-lib has flagged for splitting.
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
	    bool identication_of_boundary_elements_by_facets=true;
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


		virtual void setup_boundary_element_info(std::ostream &outfile) ;
		// Populate this mesh's elements/nodes/boundaries from an already-built MeshTemplateElementCollection
		// (the dimension-specific subclasses implement this: TemplatedMeshBase1d/2d/3d).
		virtual void generate_from_template(MeshTemplateElementCollection *coll) = 0;

		// Hook allowing Python code to post-process elemental error estimates before adapt() uses them
		// to decide refinement/coarsening; identity by default.
		virtual std::vector<double> update_elemental_errors(std::vector<double> &errors)
		{
			return errors;
		}

		// Can we refine the mesh? By defult, no: only the dimension-specific subclasses implement refinement.
		virtual bool refinement_possible() {return false;} 

		// Wraps oomph-lib's TreeBasedRefineableMeshBase::adapt: lets the Python-side
		// update_elemental_errors hook post-process the per-element error estimates (e.g. to bias
		// refinement) before handing them to the actual oomph-lib refinement/coarsening logic.
		void adapt(const oomph::Vector<double> &elemental_error)
		{
			if (!this->refinement_possible())
			{
				return; // No-op if the mesh type does not support refinement
			}
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

		// Remove nodes flagged as obsolete (is_obsolete()) from Node_pt, deleting them. Unlike oomph-lib's
		// usual node-pruning, this does not update boundary node lists, so it must only be used when the
		// caller already knows no obsolete node is registered on any boundary.
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

	// Spatial index over a mesh's nodes, used for nearest-node and containing-element lookups
	// (e.g. interpolation between meshes, connecting periodic/interface meshes). Can be built over
	// either Eulerian or Lagrangian coordinates, at a given history/time index.
	class MeshKDTree
	{
	protected:
		bool lagrangian; // Whether the tree indexes Lagrangian (true) or Eulerian (false) coordinates
		unsigned tindex; // Time-history index of the coordinates used to build the tree
		std::vector<pyoomph::Node *> nodes_by_index; // Node pointers, indexed the same way as the underlying KDTree
		std::map<pyoomph::Node *, std::set<pyoomph::BulkElementBase *>> nodes_to_elem; // Elements adjacent to each node, used to search from a node to a containing element
		KDTree *tree;
		// Return the index (into nodes_by_index/tree) of the node nearest to coord; optionally returns the distance.
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
		// Locate the element containing the point zeta (in the same coordinate system the tree was built with),
		// starting the search from the nearest node; sreturn receives the local coordinates within that element.
		pyoomph::BulkElementBase *find_element(oomph::Vector<double> zeta, oomph::Vector<double> &sreturn);
	};

}
