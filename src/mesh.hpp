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
#include "refineable_telements.hpp" // for RefineableTElement<2>::clear_shared_edge_node_registry()
#include "wedges_and_pyramids.hpp"  // for RefineableWedgeElement::clear_shared_node_registry()

namespace pyoomph
{

	class MeshTemplate;
	class MeshTemplateElementCollection;
	class Problem;
	class BulkElementBase;
	class DynamicJITCode;

	class Mesh;


	// Cross-check that all processes agree about the elements they share, and report or throw if they do
	// not. For every pair of processes, each haloed element is compared against the halo copy the other
	// process holds of it: their positions, their refinement levels, their pending refinement/unrefinement
	// flags, and -- if `errors` is given -- the error estimate that is about to decide their fate.
	//
	// Comparing the POSITIONS is what makes the output usable. Mesh::halo_element_pt() and
	// haloed_element_pt() build their vectors by walking the leaves of the same root elements' trees, so
	// the two lists correspond entry by entry only for as long as the trees agree. Once they diverge, every
	// later comparison (and every real halo exchange, including the error estimator's) is silently
	// comparing unrelated elements. Carrying an independent identity tells "these two processes disagree
	// about this element" apart from "these two lists no longer describe the same elements at all".
	//
	// Returns the number of inconsistencies found ACROSS ALL PROCESSES (every process gets the same
	// number, so a caller cannot act on a verdict its peers do not share); 0 means they agree. Costs a
	// handful of
	// point-to-point messages, so it is meant for diagnosis, not for every adapt of a production run --
	// see PYOOMPH_CHECK_HALO_CONSISTENCY in TemplatedMeshBase::adapt(). No-op unless the mesh is
	// distributed over more than one process.
	unsigned check_halo_element_consistency(oomph::Mesh *mesh_pt, const std::string &stage,
											const oomph::Vector<double> *errors, bool throw_on_mismatch);


	// Base class for all pyoomph meshes. Wraps an oomph-lib (Refineable)Mesh and adds
	// everything pyoomph needs on top: named boundaries, per-field initial/Dirichlet
	// conditions, output scaling, interface-dof bookkeeping, JIT-compiled element code
	// binding (jitcode) and helpers for interpolation/projection between meshes.
	class Mesh : public virtual oomph::RefineableMeshBase, public virtual oomph::Mesh
	{
	protected:
		Problem *problem; // Owning Problem (non-owning pointer)
		std::string domainname; // Name of the domain this mesh represents, e.g. as used in the Python interface
		std::vector<std::string> boundary_names; // Names of the boundaries, indexed like oomph-lib's boundary indices
		std::map<std::string, GiNaC::ex> initial_conditions; // Symbolic initial condition expressions, keyed by field/IC name
		std::map<std::string, double> output_scales; // Nondimensionalization output scales, keyed by field name
		std::map<unsigned, double> boundary_zeta_periods; // boundary index -> period, absent/0 meaning not periodic
		std::map<std::string, unsigned> interface_dof_ids; // Names of interface-only degrees of freedom mapped to their global index
		std::vector<bool> dirichlet_active; // Whether each Dirichlet condition (by index) is currently active
		std::map<pyoomph::Node *, pyoomph::Node *> copied_masters; // Maps a copied node to its master node (see resolve_copy_master)
		unsigned long topology_generation = 0; // Bumped whenever cached element pointers/coordinates go stale, see bump_topology_generation()
		DynamicJITCode *jitcode = NULL; // JIT-compiled element code backing the elements of this mesh

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
		// Fill a lookup buffer mapping each local dof to its global field index, used to assemble field-wise data.
		virtual void fill_dof_to_global_field_index_buffer(std::vector<int> &dofs_to_global_field_index);
		// From the old mesh, map each element with the local coordinates associated to each integration point of the new mesh.
		virtual void prepare_zeta_interpolation(pyoomph::Mesh *oldmesh);
		// Locate a list of points in this mesh and report what happened, without transferring any
		// values. Exists because the locator is otherwise only reachable through routines that do a
		// great deal else (and, in add_interpolated_nodes_at's case, cannot be pointed at an
		// interface mesh at all - it builds bulk nodes from element_pt(0)). Returns one row per
		// point: {found, offset, s...}, where offset is the perpendicular distance to the element in
		// codimension-1 (projection) mode and 0 when the map was inverted exactly.
		std::vector<std::vector<double>> locate_points(const std::vector<std::vector<double>> &coords, bool lagrangian);
		// Locate points and evaluate this mesh's fields there, one row per point:
		// [found, <continuous fields>, <DL fields>, <D0 fields>, <position if requested>], all at
		// time level time_level. The continuous block is in the mesh's own nodal field index order.
		// Unlocated points come back as a single 0.0.
		std::vector<std::vector<double>> evaluate_at_points(const std::vector<std::vector<double>> &coords, bool lagrangian, bool with_position, unsigned time_level);
		// Print how long the point-location phase of nodal_interpolate_from took, and how the
		// candidate sets were arrived at. Off by default; interpolation is called often enough that
		// unconditional timing output would be noise.
		//
		// Deliberately still process-wide, unlike interpolate_new_interface_dofs and
		// use_eigen_error_estimators, which moved onto Problem because switching them for one Problem
		// silently switched them for every other one. This one only prints: the worst a second
		// Problem suffers is diagnostics it did not ask for. It is also switched on BEFORE the
		// Problem that will report exists (see tests/test_mesh_point_locator.py), which a per-Problem
		// setting could not express.
		static bool report_interpolation_timing;
		virtual void set_time_level_for_projection(unsigned time_level);
		// Turn the zeta-projection residual on or off for every element of this mesh. It has to stay
		// on across several assemblies (a Newton solve assembles more than once), so it is switched
		// explicitly rather than consumed by the first assembly.
		void set_zeta_projection_enabled(bool yesno);
		// Whether prepare_zeta_interpolation() has run and left usable integration-point mappings.
		bool has_zeta_projection_prepared() const;
		// Prepare internal caches (e.g. KD-tree) required before interpolation calls; must be called before nodal_interpolate_*.
		virtual void prepare_interpolation();
		// Write each hanging node's constrained value (interpolated from its masters via the hanging scheme)
		// into its own RAW storage. Node::value() computes the hanging value on the fly from the masters'
		// RAW values, but the raw storage of the hanging node itself is what Data::value() (and hence direct
		// node value access / output) returns. On a distributed mesh the raw storage is left stale after the
		// first post-adapt solve (the masters -- halo nodes -- are only value-synchronised at solve end), so
		// this pass, called after that synchronisation, makes the raw values consistent with the constraint.
		void collapse_hanging_node_values()
		{
			for (unsigned long in = 0; in < this->nnode(); in++)
			{
				oomph::Node *n = this->node_pt(in);
				if (!n) continue;
				const unsigned nt = n->ntstorage();
				const unsigned nv = n->nvalue();
				for (unsigned i = 0; i < nv; i++)
					if (n->is_hanging(i))
						for (unsigned t = 0; t < nt; t++)
							n->set_value(t, i, n->value(t, i));
			}
		}

		// Push every hanging node's master-interpolated POSITION and value into its own raw storage, by
		// running each element's interpolate_hang_values(). Unlike collapse_hanging_node_values above,
		// this also covers the nodal positions, and it goes through the elements because that is where
		// the hang flattening (constraints composed with genuine hangs) lives.
		//
		// A hanging node's raw storage is a cache of its masters; only an assembly/output pass refreshes
		// it. Anything that writes the dof vector from outside the Newton solver therefore leaves it
		// stale, and a stale hanging POSITION is not merely cosmetic: the triangle/tetrahedron
		// node-sharing in RefineableTElement<2>/<3>::build matches a new son node against a snapshot of
		// the existing node positions, so a hanging node sitting in the wrong place makes the son
		// duplicate it instead of reusing it -- the mesh tears and the node count no longer matches what
		// the stored refinement pattern reproduces on reload.
		void interpolate_hanging_values();

		// Diagnostic: cross-check that all processes agree about the elements they share (positions,
		// refinement levels, pending refinement flags). Returns the number of inconsistencies found, 0 if
		// the processes agree; no-op and 0 unless the mesh is distributed over more than one process.
		// Collective -- every process must call it. See check_halo_element_consistency.
		unsigned check_halo_consistency(bool throw_on_mismatch = false)
		{
			return check_halo_element_consistency(this, "manual check", NULL, throw_on_mismatch);
		}

		virtual void clear_additional_dof_constraints(); // Clear any additional dof constraints that have been applied to this mesh's nodes
		virtual void apply_additional_dof_constraints(); // Apply any additional dof constraints that have been registered on this mesh's nodes
		// Interpolate nodal values of this mesh from the mesh "from". If boundary_index>=0, only nodes on that
		// boundary are interpolated (used when remeshing only a boundary/interface region).
		// Transfer nodal values from `from`. boundary_index >= 0 restricts to one boundary of an
		// interface mesh. use_boundary_coordinate selects HOW the matching point is found there: through
		// the intrinsic boundary coordinate (zeta) when one is defined, or - when it is not - by
		// projecting each node onto the old interface geometry, which needs no chart and is therefore
		// the only option for a 2d interface in 3d. See dev_docs/mesh_point_locator.md.
		virtual void nodal_interpolate_from(Mesh *from, int boundary_index, bool use_boundary_coordinate = true);
		// Whether nodal_interpolate_from(from) is a transfer OUT of a distributed mesh INTO a replicated
		// one, which is what remeshing a distributed problem does: the new mesh is rebuilt in full on
		// every rank (see Problem._redistribute_after_remeshing), while the old one is still partitioned.
		// Each rank can then only place the nodes that fall into its own share of the old mesh, so the
		// transfer is only complete once the ranks have pooled what each of them found.
		bool interpolation_is_shared_across_ranks(Mesh *from) const;
		// Pool per-node transferred data across the ranks: everything a mesh-to-mesh transfer writes on a
		// node (values at every time level, the position history, the Lagrangian coordinates when those
		// are interpolated). `nodes` must be the same list in the same order on every rank, and `weights`
		// says which of them this rank filled - each datum comes out as the sum over the ranks that had
		// it divided by their number. Returns the summed weights, i.e. per node how many ranks had it.
		// Collective on the Problem's communicator; only call it where every rank arrives.
		std::vector<double> pool_node_values_across_ranks(const std::vector<oomph::Node *> &nodes,
														  std::vector<double> weights);
		// Pool that transfer. Every rank holds the same nodes and elements in the same order here, so the
		// values of whatever each rank could place are summed across the ranks and divided by how many
		// placed it. Nodes rescued from another rank are added to completed_nodes and removed from
		// missing_nodes, so that the nearest-node fallback - and the count it reports - are left with the
		// nodes that are genuinely outside the old mesh everywhere. Returns how many were rescued.
		// See dev_docs/distributed_remeshing.md, stage 3.
		unsigned share_interpolation_across_ranks(Mesh *from, int boundary_index, bool interface_case,
												  const std::vector<bool> &completed_elements,
												  std::set<oomph::Node *> &completed_nodes,
												  std::set<oomph::Node *> &missing_nodes);
		// Interpolate nodal values along a boundary from an old mesh, using the arclength-like boundary coordinate
		// to find the closest correspondence; boundary_max_dist limits how far a match may be to still be accepted.
		virtual void nodal_interpolate_along_boundary(Mesh *from, int bind, int oldbind, Mesh *imesh, Mesh *oldimesh, double boundary_max_dist);
		// Settle the boundary matching above across the ranks. Every rank produces a match for every
		// destination node here - the nearest of ITS old nodes, however far away - so unlike the
		// point-located transfer this is not "who found it" but "who found it closest": one MPI_MINLOC
		// over the match distances picks the owner (ties by the lower rank, which the nodes on a
		// partition boundary need), and only its blend is kept. Also reports what no rank could match,
		// which is the only place where that count means anything. See dev_docs/distributed_remeshing.md.
		void pool_boundary_interpolation_across_ranks(const std::vector<oomph::Node *> &newnodes,
													  const std::vector<double> &local_dist,
													  Mesh *oldimesh, const std::string &where);
		// Bind this mesh to its owning Problem and the JIT-compiled element code used to create its elements.
		virtual void _set_problem(Problem *p, DynamicJITCode *code);
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
		virtual void fill_internal_facet_buffers(std::vector<BulkElementBase *> &, std::vector<int> &, std::vector<BulkElementBase *> &, std::vector<int> &, std::vector<int> &) { throw_runtime_error("Please specify this function for each dimension"); }
		// Build an InterfaceMesh of elements attached to the boundary/interface named intername, using the
		// JIT-compiled interface element code interface_jitcode; imesh is the interface mesh to populate.
		virtual void generate_interface_elements(std::string intername, Mesh *imesh, DynamicJITCode *interface_jitcode);
		virtual void ensure_external_data();
		virtual double get_temporal_error_norm_contribution();
		// Store the output scale factor s (a symbolic expression, evaluated via _code) used to nondimensionalize field fname on output.
		void set_output_scale(std::string fname, GiNaC::ex s, DynamicJITCode *_code);
		double get_output_scale(std::string fname);
		int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nelem, bool discontinuous); // Gets the number of required elemental indices
		// Same, but also fills source_element_indices (if non-NULL) with the mesh element index behind each output row
		int get_num_numpy_elemental_indices(bool tesselate_tri, unsigned &nelem, bool discontinuous, std::vector<int> *source_element_indices);
		// Mesh element index behind each element row produced by to_numpy with the same arguments. The rows are
		// sub-elements (tesselate_tri splits a quad into triangles), so they cannot be zipped with element_pt().
		std::vector<int> get_numpy_element_source_indices(bool tesselate_tri, bool discontinuous);
		// Row indices, in to_numpy's node ordering, of the nodes shared with process p, index-matched to process p's
		// own list for this rank (empty without MPI). Used to merge the per-rank meshes into one global mesh.
		virtual std::vector<int> get_shared_node_numpy_indices(unsigned p);

		// --- Partition-independent addressing, for state files (dev_docs/distributed_state_files.md) ---
		void assign_global_base_element_indices();            // number the roots; must run before distribute()
		std::vector<long> get_element_structural_keys();      // (root index, packed tree path) per element
		// Order two elements the same way on every rank: by root index, then by the refinement path
		// read root->leaf. <0 if a comes first, >0 if b does, 0 if it cannot be decided (the base
		// indices have not been assigned yet, i.e. before the initial distribution). Callers then fall
		// back to local element order, which is the same order while the mesh is still whole.
		static int compare_structural_order(oomph::GeneralisedElement *a, oomph::GeneralisedElement *b);
		// (root global_base_index, packed refinement path) of a single element, as
		// get_element_structural_keys() reports them. False, leaving the outputs untouched, when the base
		// indices have not been assigned. Packed rather than digit-wise because callers of this one need
		// EQUALITY across ranks, not the preorder that compare_structural_order() has to reproduce.
		static bool element_structural_key(oomph::GeneralisedElement *e, long &root_index, long &path);
		// Refinement trees of all roots held here, ascending: preorder son counts (0 = leaf), concatenated
		void get_all_refinement_signatures(std::vector<long> &roots, std::vector<int> &lengths, std::vector<int> &data);
		std::vector<int> get_element_node_indices(unsigned &stride); // nelement x stride, -1 padded
		void save_nodal_state(std::vector<double> &data, std::vector<int> &lengths);      // node_pt order
		void load_nodal_state(const std::vector<double> &data, const std::vector<int> &lengths);
		void save_elemental_state(std::vector<double> &data, std::vector<int> &lengths);  // element order
		void load_elemental_state(const std::vector<double> &data, const std::vector<int> &lengths);
		void refine_selected_elements_by_index(const std::vector<unsigned> &indices);
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
		// Whole-vector snapshot/restore of the activation flags, see mesh.cpp for why it exists.
		std::vector<bool> get_dirichlet_active_flags() const;
		void set_dirichlet_active_flags(const std::vector<bool> &flags);
		// Ensure the intrinsic boundary coordinate (arclength/zeta along boundary_index) has been set up on all its nodes.
		virtual void boundary_coordinates_bool(unsigned boundary_index);
		virtual bool is_boundary_coordinate_defined(unsigned boundary_index);
		// Period of the intrinsic boundary coordinate along a boundary, or 0 for a non-periodic one.
		//
		// A closed interface loop has no single-valued zeta: the element closing the loop runs from the
		// last node's zeta back to the first's, so it spans the whole range and matches essentially any
		// query. Recording the period instead makes that element well behaved - it is read as running
		// from z_last to z_first + period - and is what lets a closed loop be parameterised at all.
		// See dev_docs/mesh_point_locator.md.
		void set_boundary_zeta_period(unsigned boundary_index, double period);
		double get_boundary_zeta_period(unsigned boundary_index) const;
		void set_spatial_error_estimator_pt(pyoomph::LagrZ2ErrorEstimator *errest) { this->spatial_error_estimator_pt() = errest; }
		//  Mesh(Problem * p,MeshTemplate *templ, std::string domain);
		//	BulkNodeIterator  nodes() { return BulkNodeIterator(this);} //Iterate over all nodes
		//	NodalIteratorAccess  boundary_nodes(const  std::string & bn );  //Iterate over a boundary
		//	NodalIteratorAccess  boundary_nodes(const std::vector<std::string> & bn );  //Iterate over boundaries
		Problem *get_problem() { return problem; }
		Mesh() : problem(NULL) {}
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
		// Human-readable path of this mesh, e.g. "droplet" or "droplet/interface". Diagnostics that
		// name only a boundary INDEX and the JIT domain name are hard to act on - the index is an
		// internal number and the domain name alone does not say which boundary of it is meant.
		virtual std::string get_full_domain_path();
		// Name of one boundary of this mesh, or "boundary <index>" if it has none.
		std::string get_boundary_name_or_index(unsigned boundary_index);
		std::vector<std::string> get_boundary_names()
		{
			return boundary_names;
		}
		virtual int has_interface_dof_id(std::string n);		  //-1 if not present
		virtual unsigned resolve_interface_dof_id(std::string n); // add it if not present
		virtual unsigned count_nnode(bool discontinuous = false);
		virtual Node *get_some_node() { return (this->nnode() ? static_cast<Node *>(this->node_pt(0)) : NULL); }
		virtual void fill_node_map(std::map<oomph::Node *, unsigned> &nodemap);
		virtual std::vector<oomph::Node *> fill_reversed_node_map(bool discontinuous = false);
		virtual void enlarge_elemental_error_max_override_to_only_nodal_connected_elems(unsigned bind);

		// --- Refinement directives -------------------------------------------------------------------
		// The declarative refinement criteria (RefineToLevel, RefineMaxElementSize) evaluated in C++
		// rather than by a Python loop over the elements.
		//
		// The point is not speed. These criteria are pure functions of an element's own level or size, so
		// evaluating them here means they are evaluated for EVERY element this process holds, halo copies
		// included -- and a halo copy therefore reaches the same verdict as the element it is a copy of,
		// by construction, with nothing to synchronise afterwards. The Python versions ran over
		// mesh.elements() and produced exactly the same values, but they did so as one more rank-local
		// override that the halo exchange then had to repair.
		//
		// Registered once, when the equations are compiled; the override VALUES are recomputed on every
		// adapt (they are reset each time), the directives themselves persist.
		struct RefinementDirective
		{
			enum Kind
			{
				ToLevel,      // refine until refinement_level() >= level; level < 0 means "to the maximum"
				MaxElementSize // refine while the element is larger than max_size
			};
			Kind kind;
			int level;
			double max_size;
		};
		void add_refinement_directive_to_level(int level)
		{
			RefinementDirective d;
			d.kind = RefinementDirective::ToLevel;
			d.level = level;
			d.max_size = 0.0;
			refinement_directives.push_back(d);
		}
		void add_refinement_directive_max_element_size(double s)
		{
			RefinementDirective d;
			d.kind = RefinementDirective::MaxElementSize;
			d.level = 0;
			d.max_size = s;
			refinement_directives.push_back(d);
		}
		void clear_refinement_directives() { refinement_directives.clear(); }
		unsigned nrefinement_directives() const { return refinement_directives.size(); }
		// Evaluate every registered directive over this mesh's elements, raising their
		// elemental_error_max_override where the directive asks for refinement (or merely asks that an
		// element not be unrefined away again). Only ever raises, so the order of the directives -- and
		// of this call relative to the other override sources -- does not matter.
		virtual void apply_refinement_directives();
		std::vector<RefinementDirective> refinement_directives;
		virtual unsigned get_nodal_dimension();
		virtual int get_element_dimension();
		// Announce that anything caching this mesh's elements or their coordinates must rebuild:
		// adaptation, remeshing, interface-element regeneration, a reset of the Lagrangian
		// coordinates. Consumers compare get_topology_generation() against the value they last
		// built at.
		//
		// A counter rather than a cached-object pointer on purpose. The tracers used to detect
		// staleness by comparing against the address of a cached kd-tree that the invalidation
		// deleted - so the allocator handing the same address back a moment later made the change
		// invisible, and the stored pointer was dangling either way.
		virtual void bump_topology_generation();
		unsigned long get_topology_generation() const { return topology_generation; }
		virtual std::map<std::string, std::string> get_field_information(); // first: names, second: list of spaces (C2,C1,DL,D0), but also (../C2 etc for elements defined on bulk domains)
		~Mesh() override;
		virtual void check_integrity();
		void map_nodes_on_macro_elements(); // Re-derive nodal positions from the macro element of every element that has one
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
		// JIT code defining the interface elements. Deliberately kept separate from the inherited
		// Mesh::jitcode: set_rebuild_information() fills this one, _set_problem() the other, and
		// the two are set at different points of interface construction.
		DynamicJITCode *code;
		std::string interfacename;
		Mesh *bulkmesh; // The bulk mesh this interface is attached to
		// Build boundary information (which interface elements/nodes lie on which sub-boundary) for a 1d interface
		// (i.e. attached to a 2d bulk mesh), restricted to boundary indices in possible_bounds.
		virtual void setup_boundary_information1d(pyoomph::Mesh *parent, const std::set<unsigned> &possible_bounds);
		// Same as setup_boundary_information1d but for a 2d interface (attached to a 3d bulk mesh).
		virtual void setup_boundary_information2d(pyoomph::Mesh *parent, const std::set<unsigned> &possible_bounds);
		std::vector<double> opposite_offset_vector,reversed_opposite_offset_vector; // Constant offset (e.g. for periodic/translated interfaces) to the opposite side and its reverse
		bool warned_about_discontinuous_reset = false; // So rebuild_after_adapt's transfer warning is printed once per interface, not once per adaptation

		// Interface-owned discontinuous values as a cloud of sampled values, which is how they are
		// carried across an adaptation (and, with the values pulled from the old mesh instead, across a
		// remeshing). The elements themselves cannot
		// be carried: clear_before_adapt() deletes every one of them and rebuild_after_adapt() makes
		// new ones from the adapted bulk mesh, so there is no father/son relation to exploit the way
		// the bulk path does. What survives is the field as a function of position, sampled at
		// scattered points and fitted back onto whatever elements come out the other side.
		//
		// Every discontinuous space goes through this, DL/D0 and the NODAL DG spaces (D1/D2/D1TB/D2TB)
		// alike: what is sampled is always one scalar per field per point, and what differs is only the
		// basis the fit reconstructs coefficients in (DL's modal one, a nodal Lagrange one, or the
		// single constant of D0) and which internal Data those coefficients are written to.
		struct DiscontinuousSnapshot
		{
			unsigned space_dim = 0;     // coordinate components per sample
			unsigned nDL = 0, nD0 = 0;  // field counts at snapshot time, checked again on restore
			unsigned nDL_modes = 0;     // eleminfo.nnode_DL, i.e. how many coefficients a DL field has
			unsigned ntstorage = 0;     // time levels stored per sample - history is the whole point
			// One entry per PRESENT nodal DG space, in the func table's order. Only the fields the
			// interface declares itself are carried (numfields_new); the ones inherited from the bulk
			// belong to the bulk element and travel with it. Part of the compatibility check, because a
			// snapshot taken against a different set of spaces cannot be read back.
			std::vector<unsigned> dg_space_index, dg_numfields_new, dg_nmodes;
			std::vector<double> coords; // space_dim per sample
			std::vector<double> values; // ntstorage*(sum(dg_numfields_new)+nDL+nD0) per sample
			// Scalars stored per sample and time level, in the internal-Data order [DG][DL][D0].
			unsigned nvalues_per_time() const
			{
				unsigned n = nDL + nD0;
				for (unsigned s = 0; s < dg_numfields_new.size(); s++)
					n += dg_numfields_new[s];
				return n;
			}
			bool same_fields_as(const DiscontinuousSnapshot &o) const
			{
				return nDL == o.nDL && nD0 == o.nD0 && dg_space_index == o.dg_space_index &&
					   dg_numfields_new == o.dg_numfields_new && dg_nmodes == o.dg_nmodes;
			}
			bool empty() const { return coords.empty(); }
			void clear()
			{
				coords.clear();
				values.clear();
				dg_space_index.clear();
				dg_numfields_new.clear();
				dg_nmodes.clear();
				space_dim = nDL = nD0 = nDL_modes = ntstorage = 0;
			}
		};
		DiscontinuousSnapshot discontinuous_snapshot;
		// Which samples of a value cloud belong to which of this mesh's elements: element -> list of
		// (sample index, local coordinate inside that element).
		typedef std::map<BulkElementBase *, std::vector<std::pair<unsigned, std::vector<double>>>> DiscontinuousSampleMap;
		// Fits the values of `snap` onto this mesh's elements using the assignment in `per_elem`, and
		// fills the elements that got nothing from their recovery expressions (or leaves them at zero
		// and reports them). Returns how many were left at zero. Shared by the two transfer paths,
		// which differ only in where the values and the assignment come from.
		unsigned fit_discontinuous_data(const DiscontinuousSnapshot &snap, const DiscontinuousSampleMap &per_elem);
		// Elements the last restore_discontinuous_data() could not fill from the snapshot and could not
		// recover either, i.e. the ones left at zero. Indices into this mesh's element list, valid until
		// the next rebuild. Exposed so a test (or a Python-side post-adapt fixup) can find them.
		std::vector<unsigned> discontinuous_unrestored_elements;
	public:
		InterfaceMesh();
		~InterfaceMesh() override;
		virtual void update_zeta_in_buffer();
		virtual void update_equation_remapping();
		// Set the offset vector used to relate this interface's coordinates to the geometrically opposite interface
		// (e.g. across a periodic domain); also updates the cached reversed_opposite_offset_vector.
		virtual void set_opposite_interface_offset_vector(const std::vector<double> & offset);
		virtual std::vector<double>  get_opposite_interface_offset_vector() {return opposite_offset_vector;}
		void fill_internal_facet_buffers(std::vector<BulkElementBase *> &internal_elements, std::vector<int> &internal_face_dir, std::vector<BulkElementBase *> &opposite_elements, std::vector<int> &opposite_face_dir, std::vector<int> &opposite_already_at_index) override;
		std::vector<oomph::FiniteElement *> opposite_interior_facets; // Facets on the geometrically opposite side (e.g. periodic partner), matched to this mesh's facets by index
		double get_temporal_error_norm_contribution() override;
		void adapt(const oomph::Vector<double> &) override {}
		void refine_uniformly(oomph::DocInfo &) override {}
		unsigned unrefine_uniformly() override { return 0; }
		// Undo prior interface-element rebuild state before the bulk mesh is adapted (interface elements are
		// rebuilt afterwards from the new bulk mesh via rebuild_after_adapt).
		virtual void clear_before_adapt();
		// Regenerate this interface's elements from the (possibly refined/coarsened) bulk mesh after adaptation.
		virtual void rebuild_after_adapt();
		// The nodal discontinuous (D1/D2/...) fields this interface declares itself ("name (space)").
		// A query, not a guard: all three of them are gone. Empty when there are none.
		std::vector<std::string> get_own_nodal_dg_fields() const;
		// Sample this interface's own discontinuous fields - DL, D0 and the nodal DG spaces - into
		// discontinuous_snapshot before the elements holding them are destroyed, and fit them back onto
		// the rebuilt elements afterwards. Called by clear_before_adapt/rebuild_after_adapt; no-ops when
		// the interface owns no discontinuous field at all.
		virtual void snapshot_discontinuous_data();
		virtual void restore_discontinuous_data();
		// Carry those same fields over from the corresponding interface mesh of a mesh that has been
		// REPLACED (Problem.force_remesh) rather than adapted. Unlike the adaptation path this cannot
		// assume that anything stayed where it was, so it disambiguates the facet soup topologically -
		// see the implementation.
		virtual void interpolate_discontinuous_data_from(InterfaceMesh *old);
		// Partition-independent address of every element of this interface, three longs each:
		// (root global_base_index, packed refinement path) of the bulk element it is attached to, plus
		// the packed chain of face indices leading from that element to this one. A face element has no
		// refinement tree of its own, so it cannot use Mesh::get_element_structural_keys(); but the bulk
		// element it hangs off does, and the face indices pick it out uniquely from among that element's
		// faces and their faces. For an ordinary interface the chain is a single face index, i.e. the
		// same key the interior-facet halo scheme pairs facets across ranks with
		// (Problem::setup_interior_facet_halo_scheme), which is what makes a state file written on one
		// partition readable on another.
		// -1,-1,-1 for an element whose bulk key is not available (before the base indices are assigned).
		std::vector<long> get_interface_element_structural_keys();
		const std::vector<unsigned> &get_discontinuous_unrestored_elements() const { return discontinuous_unrestored_elements; }
		// Unpin/null out dofs on this interface that must not be treated as independent (e.g. because they are
		// algebraically slaved to the bulk mesh across the interface).
		virtual void nullify_selected_bulk_dofs();
		// Store the information (bulk mesh, interface name, JIT code) needed to rebuild this interface mesh later,
		// e.g. after adaptation via rebuild_after_adapt.
		virtual void set_rebuild_information(Mesh *_bulkmesh, std::string intername, DynamicJITCode *interface_jitcode);
		virtual Mesh *get_bulk_mesh() { return bulkmesh; }
		const std::string &get_interface_name() const { return interfacename; }
		// Drop the halo/haloed element lists. They point at elements this mesh is about to delete, and
		// flush_element_and_node_storage() does not touch them - a stale entry is a dangling read inside
		// Problem::copy_haloed_eqn_numbers_helper, i.e. a segfault far from here.
		void clear_halo_element_scheme()
		{
			// The lists themselves only exist in an MPI build (oomph::Mesh declares them inside its own
			// OOMPH_HAS_MPI guard), and without MPI there is nothing to go stale.
#ifdef OOMPH_HAS_MPI
			Root_halo_element_pt.clear();
			Root_haloed_element_pt.clear();
#endif
		}
		std::string get_full_domain_path() override;
		unsigned count_nnode(bool discontinuous = false) override; // Interface meshes don't have their own nodes...
		Node *get_some_node() override { return (this->nelement() ? static_cast<Node *>(dynamic_cast<oomph::FiniteElement *>(this->element_pt(0))->node_pt(0)) : NULL); }
		void fill_node_map(std::map<oomph::Node *, unsigned> &nodemap) override;
		std::vector<oomph::Node *> fill_reversed_node_map(bool discontinuous = false) override;
		std::vector<int> get_shared_node_numpy_indices(unsigned p) override; // Delegates to the bulk mesh's scheme
		int has_interface_dof_id(std::string n) override { return bulkmesh->has_interface_dof_id(n); }
		unsigned resolve_interface_dof_id(std::string n) override { return bulkmesh->resolve_interface_dof_id(n); }
		virtual void setup_boundary_information(pyoomph::Mesh *parent);
		// Match this interface mesh's elements/nodes to those of another interface mesh (e.g. the opposite side of
		// a periodic domain) by nearest-neighbor lookup via a KD-tree, populating opposite_interior_facets etc.
		virtual void connect_interface_elements_by_kdtree(InterfaceMesh *other);
		unsigned get_nodal_dimension() override;
		int get_element_dimension() override;
	};

	// A "mesh" that does not discretize a spatial domain but instead stores ODE (0-dimensional)
	// elements, i.e. degrees of freedom that are not associated with any node/geometry (global ODEs).
	class ODEStorageMesh : public Mesh
	{
	protected:
		std::map<std::string, unsigned> name_to_index; // Maps ODE name to its index in Element_pt

	public:
		ODEStorageMesh();
		~ODEStorageMesh() override;
		double get_temporal_error_norm_contribution() override;
		void adapt(const oomph::Vector<double> &) override {}
		void refine_uniformly(oomph::DocInfo &) override {}
		unsigned unrefine_uniformly() override { return 0; }
		void setup_initial_conditions(bool resetting_first_step, std::string ic_name) override;
		void setup_Dirichlet_conditions(bool only_update_vals) override;
		// Register a new named ODE (GeneralisedElement) in this storage mesh; returns its index.
		virtual unsigned add_ODE(std::string name, oomph::GeneralisedElement *ode);
		// Look up a previously added ODE element by name.
		virtual oomph::GeneralisedElement *get_ODE(std::string name);
		unsigned get_nodal_dimension() override { return 0; }
		int get_element_dimension() override { return 0; }
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
		DynamicTreeRoot(const DynamicTreeRoot &) : oomph::Tree(), DynamicTree(NULL), oomph::TreeRoot(NULL)
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
		void split_elements_if_required() override
		{
			// Start of a refinement round: clear the triangle/tet shared-node registries so they
			// only hold nodes created during this round (keyed by father node pairs). See
			// RefineableTElement<2>/<3>'s Shared_edge_node_registry.
			oomph::RefineableTElement<2>::clear_shared_edge_node_registry();
			oomph::RefineableTElement<3>::clear_shared_edge_node_registry();
			oomph::RefineableWedgeElement::clear_shared_node_registry();
			oomph::RefineablePyramidElement::clear_shared_node_registry();
			// Detect a MIXED 3d forest (>=2 of {brick,tet,wedge,pyramid} present among the current leaves). When
			// set, the brick/tet/wedge/pyramid builds share one weight-augmented registry so cross-shape face
			// nodes are shared, not torn -- and a brick refines through build_as_brick_son rather than oomph's
			// native octree path. A pure single-family mesh leaves it false (each keeps its own path).
			{
				bool has_brick = false, has_tet = false, has_wedge = false, has_pyr = false;
				for (unsigned long e = 0; e < this->nelement(); e++)
				{
					oomph::GeneralisedElement *g = this->element_pt(e);
					if (dynamic_cast<oomph::BrickElementBase *>(g)) has_brick = true;
					else if (dynamic_cast<oomph::RefineableWedgeElement *>(g)) has_wedge = true;
					else if (dynamic_cast<oomph::RefineablePyramidElement *>(g)) has_pyr = true;
					else if (dynamic_cast<oomph::RefineableTElement<3> *>(g)) has_tet = true;
				}
				oomph::RefineablePyramidElement::Mixed_forest_active =
					((has_brick ? 1 : 0) + (has_tet ? 1 : 0) + (has_wedge ? 1 : 0) + (has_pyr ? 1 : 0)) >= 2;
			}
			// Snapshot the live nodes by the TOPOLOGICAL key they were born with, so a build() in this round
			// can reuse a node an earlier round created (e.g. by a neighbour that was refined before this
			// one) instead of duplicating it -- which would tear a moving mesh apart at a refine/coarsen
			// interface. Used by the mixed-3d builds (wedge/pyramid/brick-as-son, and tets inside a mixed
			// forest); the 2d triangle and pure-tet builds resolve the same question by walking the tree.
			// Rebuilding it here rather than keeping a registry alive across rounds is what makes the
			// pointers safe: every node in the list is alive, and so are its generating nodes, because
			// unrefinement removes the finer nodes before the coarser ones they were built from.
			oomph::RefineablePyramidElement::clear_shared_node_snapshot();
			for (unsigned long in = 0; in < this->nnode(); in++)
				oomph::RefineablePyramidElement::register_node_in_snapshot(this->node_pt(in));
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
		void p_refine_elements_if_required() override
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
			const unsigned &ncont_interpolated_values) override;

#endif

	public:
		// --- Facet-based adjacency (generic, shape/scheme-neutral neighbour lookup) ---
		// Groups the current active (leaf) bulk elements by the vertex-node set of each of their
		// facets (edges in 2d, faces in 3d), yielding for every facet the list of (element, local
		// face index) pairs incident on it. This is the shape- and split-scheme-neutral
		// neighbour-finding primitive for the generic refinement engine (see
		// dev_docs/adaptive_refinement.md): unlike oomph's compass-based QuadTree/OcTree neighbour
		// tables it works uniformly for triangles/tets/wedges/pyramids and mixed meshes. It is built
		// from get_possible_face_indices() / get_vertex_nodes_of_face(), the same primitives the
		// boundary detection uses.
		typedef std::map<std::set<pyoomph::Node *>, std::vector<std::pair<pyoomph::BulkElementBase *, int>>> FacetAdjacencyMap;
		FacetAdjacencyMap build_facet_adjacency() const;

		// Consistency summary of build_facet_adjacency() as {n_facets, n_boundary_facets,
		// n_interior_facets, max_incidence}. On a conforming mesh every facet is incident on exactly
		// 1 (boundary) or 2 (interior) element faces, so max_incidence == 2 and n_boundary_facets
		// equals the number of genuine boundary faces. A max_incidence > 2 signals a
		// non-manifold/hanging configuration (expected on a non-conformingly refined mesh). Used to
		// validate the adjacency primitive.
		std::vector<unsigned> facet_adjacency_summary() const;

		// --- Per-face boundary tags (the unified boundary-element identification) ---
		// Seeds BulkElementBase::face_boundaries on every current element from the `facets` map, i.e.
		// from the MeshTemplate's explicit facet->boundary records. Called once right after
		// generate_from_template/setup_facets_from_template; from then on the tags are carried by the
		// elements themselves and propagated to the sons at every split (BulkElementBase::dynamic_split),
		// so they stay exact under non-uniform refinement and survive re-rooting of the tree forest.
		// Sets face_boundary_tags_valid if the template supplied any facet information at all.
		void seed_face_boundaries_from_facets();

		// The single, shape- and dimension-neutral boundary-element identification: fills
		// Boundary_element_pt / Face_index_at_boundary purely from the elements' face tags. Replaces
		// the per-shape reconstructions from nodal boundary membership (quads/tris in 2d,
		// bricks/tets in 3d), which cannot tell a genuine boundary face from an interior face whose
		// vertices all happen to lie on one and the same boundary.
		void setup_boundary_element_info_from_face_tags();

		// True once seed_face_boundaries_from_facets() has run on a mesh whose template actually
		// carried facet information. Meshes that build their elements outside the template +
		// refinement path (e.g. by calling add_tri_C1 and friends directly) leave this false and keep
		// using the legacy node-membership reconstruction.
		bool face_boundary_tags_valid = false;
		// Drops all face tags and the validity flag; used when the element set is replaced wholesale
		// by something that cannot supply facet information.
		void invalidate_face_boundary_tags();

		// --- Nodal boundary membership, reconciled against the same face tags ---
		// The refinement rules give a new node the boundaries shared by all its generating nodes, which
		// is a superset of the truth whenever an element has two or more faces on one and the same
		// boundary. The tags do not have that weakness, so they are used to correct the node labels
		// after every adapt. See dev_docs/boundary_node_membership.md.
		void collect_face_tag_node_sets(std::vector<std::set<oomph::Node *>> &truth, std::set<oomph::Node *> &decidable) const;
		// Diagnostic: (spurious, missing) without changing anything. `missing` must be zero.
		std::pair<unsigned, unsigned> check_boundary_node_membership_against_face_tags() const;
		// Removes the spurious memberships; returns how many. Fills Pending_boundary_membership_removals.
		unsigned repair_boundary_node_membership_from_face_tags();
		void detach_pending_boundary_memberships();
		// What the last local repair removed, replayed onto the other ranks' halo copies by
		// reconcile_boundary_node_membership_across_processes().
		std::vector<std::pair<oomph::Node *, unsigned>> Pending_boundary_membership_removals;
		// Escape hatch, not an opt-in: on every mesh where an element meets a boundary in a single face
		// the repair provably removes nothing, so it is on by default.
		bool repair_boundary_node_membership = true;

		// Hooks called by oomph's adapt_mesh(); see the //FOR PYOOMPH comment on oomph::Mesh for why
		// there are two of them and why their placement is load-bearing.
		void reconcile_boundary_node_membership_locally() override { repair_boundary_node_membership_from_face_tags(); }
		void reconcile_boundary_node_membership_across_processes() override;

		//	void set_spatial_error_estimator_pt(oomph::Z2ErrorEstimator * errest) {this->spatial_error_estimator_pt()=errest;}
		TemplatedMeshBase() : pyoomph::Mesh() {}
		//	Problem * get_problem() {return problem;}

		/// Broken copy constructor
		TemplatedMeshBase(const TemplatedMeshBase &) : oomph::Mesh(), pyoomph::Mesh()
		{
			oomph::BrokenCopy::broken_copy("TemplatedMeshBase");
		}

		virtual void setup_interior_boundary_elements(unsigned ) {} // Tri meshes must add internal boundary elements by hand

		// Explicitly destroys the tree forest (if any) right now, instead of waiting for this mesh's own
		// destructor to do so. Tree::~Tree() deletes the "father" (non-leaf, already-refined-away) elements
		// it still owns - these are not reachable via the mesh's own element_pt() array any more (they were
		// removed from it when they got refined into their sons), so a generic "delete every element_pt(j)"
		// pass (e.g. Problem::unload_all_dlls()) never sees them. Calling this first, while the compiled
		// element code (and its DLL) is still loaded, lets those father elements' destructors run safely;
		// leaf elements (nsons==0) are explicitly left untouched by Tree::~Tree(), so this cannot conflict
		// with a subsequent explicit deletion of this mesh's own (leaf) element_pt()/node_pt() arrays.
		void _kill_tree_forest_now()
		{
			if (this->Forest_pt)
			{
				delete this->Forest_pt;
				this->Forest_pt = 0;
			}
		}

		/// Broken assignment operator
		void operator=(const TemplatedMeshBase &)
		{
			oomph::BrokenCopy::broken_assign("TemplatedMeshBase");
		}


		void setup_boundary_element_info(std::ostream &outfile) override ;
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
		void adapt(const oomph::Vector<double> &elemental_error) override
		{
			adapt_select(elemental_error);
			adapt_execute();
			adapt_finalise();
		}

		// adapt() in three separately callable pieces. The point of the split is the gap between the
		// first and the second: two meshes that share a coupled interface are adapted individually, and
		// the only place their decisions can be reconciled EXACTLY is after both have decided and before
		// either has acted. Reconciling the ERRORS beforehand cannot do it -- oomph unrefines a father
		// only if all its sons agree, and a veto from a son that does not touch the interface is
		// invisible to any interface-level comparison of errors
		// (dev_docs/interface_refinement_coupling.md §6). Only the flags carry that information.
		//
		// Used in the split form only when there is something to reconcile; everything else keeps calling
		// adapt(), which is these three back to back and behaves exactly as it did before the split.

		// Decide, without acting: translate the errors (after the Python hook and the halo
		// synchronisation) into per-element refine/unrefine flags.
		void adapt_select(const oomph::Vector<double> &elemental_error)
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
			// The errors reaching this point are not necessarily the pure Z2 estimates: pyoomph lets
			// equations override them per element (RefineToLevel, RefineMaxElementSize, interface-driven
			// overrides, user hooks...). Those overrides are computed rank-locally, so a halo copy can
			// end up with a different error than the element it is a copy of, and oomph-lib then refines
			// the two inconsistently. Reduce over all copies before adapting.
			this->synchronise_elemental_errors(updated_errors);
			// Opt-in diagnostic, off by default. Checked AFTER the synchronisation above, not before:
			// disagreeing overrides are expected there and are exactly what it just repaired, so what is
			// worth reporting is any disagreement that SURVIVED the repair.
			const int halochk = this->halo_consistency_check_mode();
			if (halochk) check_halo_element_consistency(this, "after error synchronisation", &updated_errors, halochk > 1);
			pending_n_refine = 0;
			pending_n_unrefine = 0;
			this->select_elements_for_refinement_and_unrefinement(updated_errors, pending_n_refine, pending_n_unrefine);
		}

		// Act on the flags. Recounts them first: anything that reconciled the flags in the gap (see
		// harmonise_adapt_selection) has invalidated the counts select produced, and those counts drive
		// the collective "is this worth doing at all" gate inside execute_selected_adaptation -- a stale
		// zero there would skip the adaptation entirely.
		void adapt_execute()
		{
			if (!this->refinement_possible()) return;
			this->recount_pending_adaptation();
			this->execute_selected_adaptation(pending_n_refine, pending_n_unrefine);
			pending_n_refine = 0;
			pending_n_unrefine = 0;
			const int halochk = this->halo_consistency_check_mode();
			if (halochk) check_halo_element_consistency(this, "after refinement", NULL, halochk > 1);
		}

		// Everything that has to happen after the elements have actually been split/merged. Deferred to
		// its own call because the 2:1 balancing below refines FURTHER elements of its own accord, some
		// of them at a coupled interface -- so on a coupled problem it has to converge jointly with the
		// cross-mesh conformity repair, not before it.
		void adapt_finalise()
		{
			if (!this->refinement_possible()) return;
			// Enforce 2:1 refinement balancing where the shape needs it (tetrahedra, whose geometric
			// hang scheme has no tree-level balancing): refine any element that is >1 level coarser
			// than a neighbour, iterating to a fixed point, so all hanging is single-level. No-op for
			// meshes handled by oomph-lib's own tree (quad/hex) or that tolerate multi-level hangs.
			this->enforce_refinement_balance();
			// After refinement, install any shape-specific hanging nodes that oomph-lib's per-element
			// setup did not (triangles/tets use a geometric hang scheme rather than the box coordinate
			// descent). Runs before equation numbering, so the hangs are picked up by
			// assign_(hanging_)local_eqn_numbers. No-op for quad/hex meshes.
			this->post_adapt_setup_hanging_nodes();
			const int halochk = this->halo_consistency_check_mode();
			if (halochk) check_halo_element_consistency(this, "after 2:1 balancing", NULL, halochk > 1);
		}

		// Recompute the pending refine/unrefine counts from the flags as they stand now. oomph's adapt()
		// counts them as it sets them, which is fine as long as nothing touches the flags in between --
		// exactly what the split above allows.
		void recount_pending_adaptation();

		// What adapt_select() (and any reconciliation after it) decided, without acting on it. The point
		// of asking is that an adaptation deciding to do nothing is common -- every mesh sitting at
		// max_refinement_level with errors still above the refinement tolerance decides it on every
		// single solve -- and the teardown/rebuild the caller wraps the adaptation in costs the same
		// whether or not an element moves. See Problem._adapt_with_interfacial_errors.
		std::pair<unsigned, unsigned> adapt_pending_counts()
		{
			if (!this->refinement_possible()) return std::make_pair(0u, 0u);
			this->recount_pending_adaptation();
			return std::make_pair(pending_n_refine, pending_n_unrefine);
		}

		// Drop a selection that is not going to be executed. Only the statistics really need it -- a
		// selection is abandoned exactly when nothing is flagged -- but nrefined()/nunrefined() would
		// otherwise keep reporting the numbers of the last adaptation that did run.
		void adapt_abandon()
		{
			pending_n_refine = 0;
			pending_n_unrefine = 0;
			this->Nrefined = 0;
			this->Nunrefined = 0;
		}

		// Put the node vector into the canonical order -- the order the elements walk the nodes in --
		// and report whether that changed anything.
		//
		// This is the one thing an abandoned adaptation still has to do. oomph-lib does it in the
		// branch of execute_selected_adaptation() that decides the adaptation is not worth carrying
		// out, and says why in its own comment: "to establish a standard ordering regardless of the
		// sequence of mesh refinements -- this is required to allow dump/restart on refined meshes".
		// Skipping the adaptation skipped this with it, so a run that never refined kept the order the
		// mesh generator built, while load_state(), a real refinement and a distribution rebuild all
		// produced the canonical one -- and states written by one could no longer be compared with
		// states of the other.
		//
		// Separated from adapt_execute() because the two have very different costs. The reordering is
		// one sweep over the elements and is idempotent, so it changes something only the first time it
		// is reached; the caller renumbers only in that case and keeps its Jacobian sparsity pattern on
		// every later no-op adaptation, which is the whole point of abandoning one.
		bool reorder_nodes_if_needed()
		{
			if (!this->refinement_possible()) return false;
			oomph::Vector<oomph::Node *> reordering;
			this->get_node_reordering(reordering, true);
			const unsigned n = this->nnode();
			bool changed = false;
			for (unsigned i = 0; i < n; i++)
			{
				if (this->node_pt(i) != reordering[i])
				{
					changed = true;
					break;
				}
			}
			if (changed)
			{
				for (unsigned i = 0; i < n; i++) this->node_pt(i) = reordering[i];
			}
			return changed;
		}

	private:
		// Carried from adapt_select() to adapt_execute(). oomph's own adapt() keeps these on the stack;
		// the split needs them to survive the gap.
		unsigned pending_n_refine = 0, pending_n_unrefine = 0;

	public:

		// Make every process agree on the error of each shared element, by MAX-reducing over all copies
		// (halo -> owner -> halo again) rather than by letting the owner's value win. oomph-lib's own Z2
		// estimator synchronises the values IT computes, but pyoomph's per-element error overrides are
		// applied afterwards and rank-locally, which would otherwise reintroduce the disagreement -- and
		// an override is not always computed on the rank that owns the element it applies to (see the
		// implementation). A max is the right reduction: the quantity is an upper-bound override.
		// No-op in serial or on a non-distributed mesh.
		void synchronise_elemental_errors(oomph::Vector<double> &errs);

		// How thoroughly adapt() should cross-check that the processes still agree about the elements they
		// share: 0 off, 1 report to stdout, 2 throw. Read from PYOOMPH_CHECK_HALO_CONSISTENCY
		// ("1"/"warn"/"report", or "2"/"throw"), then agreed across all processes -- the check itself is
		// collective, so a variable set on only some ranks would otherwise deadlock rather than diagnose.
		int halo_consistency_check_mode();

		// Hook to enforce 2:1 refinement balancing after adapt; overridden by dimension-specific
		// subclasses that need it (currently 3d tetrahedra). No-op by default.
		virtual void enforce_refinement_balance() {}

		// Hook to (re)build shape-specific hanging nodes after refinement; overridden by the
		// dimension-specific subclasses that need it (currently 2d triangles). No-op by default.
		virtual void post_adapt_setup_hanging_nodes() {}

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

}
