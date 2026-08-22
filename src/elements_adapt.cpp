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


// h-adaptivity: pre_build/further_build/rebuild_from_sons, the son-to-father local coordinate maps,
// face_index_in_father, the refinement patterns, and the mixed-refinement son builders that let a
// brick or pyramid father produce sons of a different element type.

#include "macroelements.hpp"
#include "elements.hpp"
#include "elements_concrete.hpp"
#include "exception.hpp"
#include "problem.hpp"
#include "nodes.hpp"
#include "meshtemplate.hpp"
#include "expressions.hpp"
#include "timestepper.hpp"

namespace pyoomph
{

	// Links this element's refinement tree root to "other"'s along direction mydir/otherdir as a
	// periodic neighbor, by dispatching to the appropriate oomph-lib tree type (QuadTree for 2d,
	// BinaryTree for 1d, OcTree for 3d) and translating pyoomph's signed-integer direction
	// convention (-1/1: W/E, -2/2: S/N, -3/3: D/U) into that tree type's named direction constants.
	// This makes oomph-lib's h-refinement treat the two boundaries as topologically adjacent for
	// hanging-node/neighbor-finding purposes, implementing periodic boundary conditions on refined meshes.
	void BulkElementBase::connect_periodic_tree(BulkElementBase *other, const int &mydir, const int &otherdir)
	{
		oomph::QuadTree *my_qt = dynamic_cast<oomph::QuadTree *>(Tree_pt);
		oomph::BinaryTree *my_bt = dynamic_cast<oomph::BinaryTree *>(Tree_pt);
		oomph::OcTree *my_ot = dynamic_cast<oomph::OcTree *>(Tree_pt);
		oomph::TreeRoot * myroot=NULL;
		oomph::TreeRoot * otherroot=NULL;
		int my_root_dir,other_root_dir;
		if (my_qt)
		{
			using namespace oomph::QuadTreeNames;
			oomph::QuadTree *other_qt = dynamic_cast<oomph::QuadTree *>(other->tree_pt());
			if (!other_qt) throw_runtime_error("Cannot connect a QuadTree with a non-QuadTree for a periodic boundary");
			myroot=my_qt->root_pt(); otherroot=other_qt->root_pt();
			if (mydir==-1) my_root_dir=W;
			else if (mydir==1) my_root_dir=E;
			else if (mydir==-2) my_root_dir=S;
			else if (mydir==2) my_root_dir=N;
			else throw_runtime_error("Invalid direction");
			if (otherdir==-1) other_root_dir=W;
			else if (otherdir==1) other_root_dir=E;
			else if (otherdir==-2) other_root_dir=S;
			else if (otherdir==2) other_root_dir=N;
			else throw_runtime_error("Invalid direction");						
		}
		else if (my_ot)
		{
			using namespace oomph::OcTreeNames;
			oomph::OcTree *other_ot = dynamic_cast<oomph::OcTree *>(other->tree_pt());			
			if (!other_ot) throw_runtime_error("Cannot connect a OcTree with a non-OcTree for a periodic boundary");			
			myroot=my_ot->root_pt(); otherroot=other_ot->root_pt();						
			if (mydir==-1) my_root_dir=L;
			else if (mydir==1) my_root_dir=R;
			else if (mydir==-2) my_root_dir=D;
			else if (mydir==2) my_root_dir=U;			
			else if (mydir==-3) my_root_dir=B;			
			else if (mydir==3) my_root_dir=F;			
			else throw_runtime_error("Invalid direction");
			if (otherdir==-1) other_root_dir=L;
			else if (otherdir==1) other_root_dir=R;
			else if (otherdir==-2) other_root_dir=D;
			else if (otherdir==2) other_root_dir=U;			
			else if (otherdir==-3) other_root_dir=B;			
			else if (otherdir==3) other_root_dir=F;			
			else throw_runtime_error("Invalid direction");			
		}
		else if (my_bt)
		{
			using namespace oomph::BinaryTreeNames;
			oomph::BinaryTree *other_bt = dynamic_cast<oomph::BinaryTree *>(other->tree_pt());
			if (!other_bt) throw_runtime_error("Cannot connect a BinaryTree with a non-BinaryTree for a periodic boundary");
			myroot=my_bt->root_pt(); otherroot=other_bt->root_pt();
			if (mydir==-1) my_root_dir=L;
			else if (mydir==1) my_root_dir=R;			
			else throw_runtime_error("Invalid direction");
			if (otherdir==-1) other_root_dir=L;
			else if (otherdir==1) other_root_dir=R;			
			else throw_runtime_error("Invalid direction");			
		}
		if (myroot && otherroot)
		{
			myroot->set_neighbour_periodic(my_root_dir);
			otherroot->set_neighbour_periodic(other_root_dir);
			myroot->neighbour_pt(my_root_dir)=otherroot;
			otherroot->neighbour_pt(other_root_dir)=myroot;
		}
		
		// Otherwise, we can't do anything
	}

	// Called by oomph-lib's tree-based mesh refinement before a son element is otherwise set up:
	// makes sure the new (son) element has a code, inheriting it from its father since
	// sons are constructed generically without going through the normal Python-driven creation.
	void BulkElementBase::pre_build(oomph::Mesh *&, oomph::Vector<oomph::Node *> &)
	{
		if (!this->jitcode)
		{
			BulkElementBase *cast_father_element_pt = dynamic_cast<BulkElementBase *>(this->father_element_pt());
			if (!cast_father_element_pt)
			{
				throw_runtime_error("Trying to build an element without a code during pre_build...");
			}
			else
				this->jitcode = cast_father_element_pt->jitcode;
		}
	}

	// Called by oomph-lib's tree-based mesh refinement right after a son element has been created
	// from a father (bisection/quadsection/octsection) and its nodes set up, to transfer this
	// element type's own data down to the son: initial element size (halved/quartered/eighthed
	// depending on binary/quad/octree), non-nodal coordinates for elements whose nodal dimension
	// exceeds their local dimension, and every non-continuous field storage (DG, DL, D0) which
	// oomph-lib's generic node-based interpolation cannot handle. DG and constant/D0 fields are
	// just evaluated at the son's local coordinate mapped into the father; DL storage (assumed to
	// hold a constant + linear-gradient representation, one value plus one gradient component per
	// spatial direction) is transferred using the standard oomph-lib "restrict to child octant"
	// formulas (value +/- 0.5*gradient per split direction, gradient halved).
	void BulkElementBase::further_build()
	{

		if (!this->tree_pt()->father_pt())
		{
			throw_runtime_error("Try to split an element, but found not father...");
			this->ensure_external_data();
			return;
		}
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		BulkElementBase *father = dynamic_cast<BulkElementBase *>(this->tree_pt()->father_pt()->object_pt());
		if (!father)
			throw_runtime_error("Try to split an element, but found not father...");
		// A son sits under the same root as its father, so it carries the same stamped root index.
		// Without this a mesh refined AFTER being distributed would hand out fresh elements with -1
		// and become unwritable again - the stamp is only laid down while the mesh is still whole.
		this->global_root_index = father->global_root_index;
		// The son's address is the father's path with one more step appended, packed the way
		// Mesh::get_element_structural_keys() packs it (3 bits per level, +1 so that son 0 is not a
		// no-op). Recomputing it from the tree is not an option once the mesh has been distributed:
		// the tree no longer reaches back past the distribution.
		if (father->global_root_path >= 0)
		{
			long which = -1;
			oomph::Tree *f = this->tree_pt() ? this->tree_pt()->father_pt() : NULL;
			if (f)
			{
				for (unsigned sn = 0; sn < f->nsons(); sn++)
					if (f->son_pt(sn) == this->tree_pt()) { which = (long)sn; break; }
			}
			if (which >= 0) this->global_root_path = father->global_root_path * 8 + (which + 1);
		}

		oomph::QuadTree *quadtree_pt = dynamic_cast<oomph::QuadTree *>(Tree_pt);
		oomph::BinaryTree *binarytree_pt = dynamic_cast<oomph::BinaryTree *>(Tree_pt);
		oomph::OcTree *octree_pt = dynamic_cast<oomph::OcTree *>(Tree_pt);
		int nsons = this->tree_pt()->father_pt()->nsons();

		this->set_nlagrangian_and_ndim(this->jitcode->get_func_table()->lagr_dim, this->jitcode->get_func_table()->nodal_dim);

		for (unsigned int i = 0; i < ninternal_data(); i++)
			internal_data_pt(i)->set_time_stepper(node_pt(0)->time_stepper_pt(), false);


		if (binarytree_pt)
		{
			initial_cartesian_nondim_size = 0.5 * father->initial_cartesian_nondim_size;			
		}
		else if (quadtree_pt)
		{
			initial_cartesian_nondim_size = 0.25 * father->initial_cartesian_nondim_size;
		}
		else if (octree_pt)
		{
			initial_cartesian_nondim_size = 0.125 * father->initial_cartesian_nondim_size;
		}

		for (unsigned t = 0; t < node_pt(0)->time_stepper_pt()->ntstorage(); t++)
		{
			if (this->nodal_dimension() != this->dim())
			{
				for (unsigned int l = 0; l < this->nnode(); l++)
				{
					// We need to map the nodes correctly
					oomph::Vector<double> sfather;
					this->get_nodal_s_in_father(l, sfather);
					oomph::Vector<double> x_prev(this->nodal_dimension());
					BulkElementBase *father_el_pt = dynamic_cast<BulkElementBase *>(tree_pt()->father_pt()->object_pt());
					father_el_pt->get_x(t, sfather, x_prev);
					for (unsigned int i = this->dim(); i < this->nodal_dimension(); i++)
					{
						//  std::cout << "BUILD NODE " << l << " has position " << i << " of " << this->node_pt(l)->x(t,i) << " at time index "  << t << std::endl;
						this->node_pt(l)->x(t, i) = x_prev[i]; // TODO: Also lagrangian?
					}
				}
			}

			const std::vector<std::vector<unsigned>> &space_node_to_elem = this->get_nodal_space_index_to_element_index_map();
			//DG 	
			for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
			{
				auto * space_info=functable->present_dg_spaces[i_space];
				for (unsigned l=0;l<this->get_eleminfo()->nnode_of_space[space_info->space_index];l++)
				{
					oomph::Vector<double> sfather,father_data;				
					this->get_nodal_s_in_father(space_node_to_elem[space_info->space_index][l], sfather);
					father->get_DG_fields_at_s(space_info->space_index,t,sfather,father_data);
					for (unsigned iindex=0;iindex<space_info->numfields;iindex++)
					{
						this->internal_data_pt(space_info->internal_offset_new+iindex)->set_value(t,l,father_data[iindex]);
					}
				}
			}


			//DL and D0
			unsigned DL_offset=functable->info_DL.internal_offset_new;
			for (unsigned int iindex = DL_offset; iindex < DL_offset+functable->info_DL.numfields; iindex++)
			{
				if (binarytree_pt)
				{
					using namespace oomph::BinaryTreeNames;
					int son_type = binarytree_pt->son_type();
					if (son_type == L)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - 0.5 * father->internal_data_pt(iindex)->value(t, 1));
					else if (son_type == R)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(0) + 0.5 * father->internal_data_pt(iindex)->value(t, 1));
					for (unsigned j = 1; j < internal_data_pt(iindex)->nvalue(); j++)
						internal_data_pt(iindex)->set_value(t, j, father->internal_data_pt(iindex)->value(t, j) / 2);
				}
				else if (quadtree_pt)
				{
					using namespace oomph::QuadTreeNames;
					int son_type = quadtree_pt->son_type();
					for (unsigned j = 1; j < internal_data_pt(iindex)->nvalue(); j++)
						internal_data_pt(iindex)->set_value(t, j, father->internal_data_pt(iindex)->value(t, j) / 2);
					double sx = 0.5 * father->internal_data_pt(iindex)->value(t, 1);
					double sy = 0.5 * father->internal_data_pt(iindex)->value(t, 2);
					if (son_type == SW)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx - sy);
					else if (son_type == NW)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx + sy);
					else if (son_type == SE)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx - sy);
					else if (son_type == NE)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx + sy);
				}
				else if (octree_pt)
				{
					using namespace oomph::OcTreeNames;
					int son_type = octree_pt->son_type();
					for (unsigned j = 1; j < internal_data_pt(iindex)->nvalue(); j++)
						internal_data_pt(iindex)->set_value(t, j, father->internal_data_pt(iindex)->value(t, j) / 2);
					double sx = 0.5 * father->internal_data_pt(iindex)->value(t, 1);
					double sy = 0.5 * father->internal_data_pt(iindex)->value(t, 2);
					double sz = 0.5 * father->internal_data_pt(iindex)->value(t, 3);

					if (son_type == LDB)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx - sy - sz);
					else if (son_type == RDB)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx - sy - sz);
					else if (son_type == LUB)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx + sy - sz);
					else if (son_type == RUB)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx + sy - sz);
					else if (son_type == LDF)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx - sy + sz);
					else if (son_type == RDF)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx - sy + sz);
					else if (son_type == LUF)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) - sx + sy + sz);
					else if (son_type == RUF)
						internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0) + sx + sy + sz);					
				}
				else
					internal_data_pt(iindex)->set_value(t, 0, father->internal_data_pt(iindex)->value(t, 0)); // TODO: Correct interpolation here, i.e. e.g for Triangle and 3d
			}

			unsigned iD0=0;
			for (unsigned int iindex = functable->info_DL.numfields+DL_offset; iindex < DL_offset+functable->info_DL.numfields + functable->info_D0.numfields; iindex++) // D0 fields
			{
				double factor = 1;
				if (functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iD0] != 0.0) // TODO: Consider on DL as well
				{
					factor = pow(nsons, -functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iD0]);
				}
				internal_data_pt(iindex)->set_value(t, 0, factor * father->internal_data_pt(iindex)->value(t, 0));
				iD0++;
			}
		}
		this->set_integration_scheme(father->integral_pt());
		this->ensure_external_data();
	}

	// Inverts the (affine) son -> father local map at s_father for a SIMPLEX son of a simplex
	// father. The son's dim+1 vertices in father coordinates come from get_nodal_s_in_father; the
	// barycentric weights that reproduce s_father from those vertices are, the map being affine,
	// the same weights that reproduce the point from the son's own vertex coordinates. Returns
	// false if the point lies outside this son (a negative weight), which is how the caller picks
	// the containing son out of the father's offspring.
	bool BulkElementBase::son_local_from_father_simplex(const oomph::Vector<double> &s_father, oomph::Vector<double> &s_son)
	{
		const unsigned dim = this->dim();
		const unsigned nv = dim + 1; // simplex vertices = local nodes 0..dim
		std::vector<std::vector<double>> vf(nv, std::vector<double>(dim, 0.0)); // vertices in father coords
		std::vector<std::vector<double>> vs(nv, std::vector<double>(dim, 0.0)); // ... and in the son's own
		for (unsigned k = 0; k < nv; k++)
		{
			oomph::Vector<double> sf, ss;
			this->get_nodal_s_in_father(k, sf);
			this->local_coordinate_of_node(k, ss);
			for (unsigned i = 0; i < dim; i++) { vf[k][i] = sf[i]; vs[k][i] = ss[i]; }
		}
		// sum_k lambda_k*vf[k] = s_father together with sum_k lambda_k = 1: a (dim+1)-square system,
		// solved by Gauss-Jordan with partial pivoting (nv is 3 or 4, so this is a handful of flops).
		std::vector<std::vector<double>> a(nv, std::vector<double>(nv + 1, 1.0));
		for (unsigned i = 0; i < dim; i++)
		{
			for (unsigned k = 0; k < nv; k++) a[i][k] = vf[k][i];
			a[i][nv] = s_father[i];
		}
		for (unsigned k = 0; k < nv; k++) a[dim][k] = 1.0;
		a[dim][nv] = 1.0;
		for (unsigned c = 0; c < nv; c++)
		{
			unsigned piv = c;
			for (unsigned r = c + 1; r < nv; r++)
				if (std::abs(a[r][c]) > std::abs(a[piv][c])) piv = r;
			if (std::abs(a[piv][c]) < 1e-14) return false; // degenerate son
			std::swap(a[c], a[piv]);
			for (unsigned r = 0; r < nv; r++)
			{
				if (r == c) continue;
				const double f = a[r][c] / a[c][c];
				for (unsigned q = c; q <= nv; q++) a[r][q] -= f * a[c][q];
			}
		}
		std::vector<double> lambda(nv, 0.0);
		for (unsigned k = 0; k < nv; k++)
		{
			lambda[k] = a[k][nv] / a[k][k];
			if (lambda[k] < -1e-9) return false; // outside this son
		}
		s_son.resize(dim, 0.0);
		for (unsigned i = 0; i < dim; i++)
		{
			s_son[i] = 0.0;
			for (unsigned k = 0; k < nv; k++) s_son[i] += lambda[k] * vs[k][i];
		}
		return true;
	}

	// Refinement can leave a father's INTERIOR nodes orphaned. A bubble (centroid) node survives a
	// split only if some son happens to hold a node at that exact point; when none does, no leaf
	// references it, so adapt_mesh() deletes it as obsolete and deactivate_element() nulls the
	// father's pointer to it. Merging the sons back then makes oomph-lib loop over the father's
	// nodes and dereference that null -- a segfault in adapt_mesh(), one adaptation after the one
	// that actually caused it. Recreate the missing nodes here instead, restricting the fine
	// solution: locate the point in the son that contains it and sample position, Lagrangian
	// coordinates and values there.
	//
	// Which enriched elements this bites is pure geometry: a 2d C1TB/C2TB centroid coincides with
	// the centre son's centroid, and a C2TB tet centroid with a son edge-mid node, so those bubbles
	// are reused during refinement and never orphaned. The 5-node C1TB tet has no such coincidence,
	// and is what this repair exists for.
	void BulkElementBase::restore_orphaned_interior_nodes(oomph::Mesh *&mesh_pt)
	{
		const unsigned nnod = this->nnode();
		bool any_missing = false;
		for (unsigned n = 0; n < nnod; n++)
			if (this->node_pt(n) == NULL) { any_missing = true; break; }
		if (!any_missing) return;

		const unsigned dim = this->dim();
		// Only a simplex can have an interior node that no son inherits (a quad/brick centre node is
		// a corner of every son). The vertex-based inverse map above is a simplex map, so refuse
		// anything else rather than silently placing the node somewhere wrong.
		if (this->nvertex_node() != dim + 1)
			throw_runtime_error("Cannot restore an orphaned interior node on a non-simplex element");
		if (!Tree_pt || Tree_pt->nsons() == 0)
			throw_runtime_error("Cannot restore an orphaned interior node without the son elements");
		if (!mesh_pt)
			throw_runtime_error("Cannot restore an orphaned interior node without a mesh to add it to");

		oomph::TimeStepper *time_stepper_pt = NULL;
		for (unsigned n = 0; n < nnod && !time_stepper_pt; n++)
			if (this->node_pt(n)) time_stepper_pt = this->node_pt(n)->time_stepper_pt();
		if (!time_stepper_pt) throw_runtime_error("All nodes of this element are missing, cannot restore any");

		const unsigned ntstorage = time_stepper_pt->ntstorage();
		const unsigned ndim_node = this->nodal_dimension();
		for (unsigned n = 0; n < nnod; n++)
		{
			if (this->node_pt(n)) continue;
			oomph::Vector<double> s;
			this->local_coordinate_of_node(n, s);
			BulkElementBase *src_son = NULL;
			oomph::Vector<double> s_son(dim, 0.0);
			for (unsigned ison = 0; ison < Tree_pt->nsons() && !src_son; ison++)
			{
				BulkElementBase *son = dynamic_cast<BulkElementBase *>(Tree_pt->son_pt(ison)->object_pt());
				if (son && son->son_local_from_father_simplex(s, s_son)) src_son = son;
			}
			if (!src_son)
				throw_runtime_error("Orphaned interior node " + std::to_string(n) + " lies in none of the sons");

			// A node the sons did not inherit can only be interior to the father, so it is never a
			// boundary node and construct_node (rather than construct_boundary_node) is right. Check
			// it rather than assume it: a boundary node rebuilt as a plain one would silently drop
			// out of the mesh's boundary lookup and its boundary conditions with it. Barycentric
			// coordinates of a simplex are (s_0..s_dim-1, 1-sum); a zero one means we are on a facet.
			double bary_last = 1.0;
			bool on_facet = false;
			for (unsigned i = 0; i < dim; i++) { bary_last -= s[i]; if (std::abs(s[i]) < 1e-10) on_facet = true; }
			if (std::abs(bary_last) < 1e-10) on_facet = true;
			if (on_facet)
				throw_runtime_error("Orphaned node " + std::to_string(n) + " is on an element facet, not interior");

			oomph::Node *new_node_pt = this->construct_node(n, time_stepper_pt);
			for (unsigned t = 0; t < ntstorage; t++)
			{
				oomph::Vector<double> x(ndim_node, 0.0);
				src_son->get_x(t, s_son, x);
				for (unsigned i = 0; i < ndim_node; i++) new_node_pt->x(t, i) = x[i];
				oomph::Vector<double> prev_values;
				src_son->get_interpolated_values(t, s_son, prev_values);
				const unsigned n_var = std::min((unsigned)new_node_pt->nvalue(), (unsigned)prev_values.size());
				for (unsigned k = 0; k < n_var; k++) new_node_pt->set_value(t, k, prev_values[k]);
			}
			// The Lagrangian (reference) coordinates of a moving mesh are not touched by the Eulerian
			// loop above; leaving them at the construct_node() default of 0 makes an undeformed mesh
			// look grossly deformed to any solid/mesh residual. Same step as in RefineableTElement<2>::build.
			if (oomph::SolidNode *solid_node_pt = dynamic_cast<oomph::SolidNode *>(new_node_pt))
			{
				const unsigned n_lagr = solid_node_pt->nlagrangian();
				const unsigned nson_nod = src_son->nnode();
				oomph::Shape psi(nson_nod);
				src_son->shape(s_son, psi);
				for (unsigned i = 0; i < n_lagr; i++)
				{
					double xi_i = 0.0;
					for (unsigned l = 0; l < nson_nod; l++)
						if (oomph::SolidNode *sn = dynamic_cast<oomph::SolidNode *>(src_son->node_pt(l)))
							xi_i += psi(l) * sn->xi(i);
					solid_node_pt->xi(i) = xi_i;
				}
			}
			mesh_pt->add_node_pt(new_node_pt);
		}
	}

	// Called by oomph-lib's tree-based mesh refinement when four/two/eight son elements are
	// merged back into their father during unrefinement: reconstructs the father's non-nodal
	// field storage (DG, DL, D0) from the sons' data, since oomph-lib's generic node-based
	// unrefinement cannot handle these. DG fields are averaged at coincident node locations
	// (accumulated then divided by the number of contributing sons); DL fields are rebuilt as
	// value + gradient (average of son values, and centered-difference slopes between sons on
	// opposite sides) -- noted as non-conservative and ignoring axisymmetric weighting; D0
	// fields are simple averages (optionally re-scaled by discontinuous_refinement_exponents for
	// fields that should scale differently under mesh coarsening, e.g. densities vs. totals).
	// TODO: Split this into the particular elements
	void BulkElementBase::rebuild_from_sons(oomph::Mesh *&mesh_pt)
	{
		// Before anything reads node_pt(): put back any interior node that the split orphaned.
		restore_orphaned_interior_nodes(mesh_pt);

		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		// Quad tree
		oomph::QuadTree *quadtree_pt = dynamic_cast<oomph::QuadTree *>(Tree_pt);
		oomph::BinaryTree *binarytree_pt = dynamic_cast<oomph::BinaryTree *>(Tree_pt);
		oomph::OcTree *octree_pt = dynamic_cast<oomph::OcTree *>(Tree_pt);
		if (functable->integration_order)
		{
			this->set_integration_order(functable->integration_order);
		}

		// DG fields
		const std::vector<std::vector<int>> & this_elem_to_space_nodal_index = this->get_element_index_to_nodal_space_index_map();

		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto * space_info=functable->present_dg_spaces[i_space];
			for (unsigned t = 0; t < node_pt(0)->time_stepper_pt()->ntstorage(); t++)
			{
				if (space_info->numfields_new)
				{
					const unsigned Nn=this->get_eleminfo()->nnode_of_space[space_info->space_index];
					std::vector<double> denom(Nn,0.0);
					for (unsigned int iindex = space_info->internal_offset_new; iindex < space_info->internal_offset_new+space_info->numfields_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) this->internal_data_pt(iindex)->set_value(t,in,0.0); //Set to 0
					}
					for (unsigned ison = 0; ison < this->required_nsons(); ison++)
					{
						BulkElementBase* son=dynamic_cast<BulkElementBase*>(Tree_pt->son_pt(ison)->object_pt());
						const std::vector<unsigned> &son_space_node_to_elem = son->get_nodal_space_index_to_element_index_map()[space_info->space_index];
						for (unsigned int in=0;in<Nn;in++)
						{
							oomph::Vector<double> s;				
							son->get_nodal_s_in_father(son_space_node_to_elem[in], s);
							oomph::Node * my_node=this->get_node_at_local_coordinate(s);
							if (my_node)
							{
								int nn=this->get_node_number(my_node);
								if (nn>=0)
								{
									nn=this_elem_to_space_nodal_index[space_info->space_index][nn];
									if (nn>=0)
									{
										for (unsigned int iindex = space_info->internal_offset_new; iindex < space_info->internal_offset_new+space_info->numfields_new; iindex++) 
										{
											double sonval=son->internal_data_pt(iindex)->value(t, in);
											this->internal_data_pt(iindex)->set_value(t,nn,this->internal_data_pt(iindex)->value(t,nn)+sonval); //Accumulate the son values
										}
										denom[nn]+=1.0;
									}
								}
							}
						}
					}
					for (unsigned int iindex = space_info->internal_offset_new; iindex < space_info->internal_offset_new+space_info->numfields_new; iindex++) 
					{
						for (unsigned int in=0;in<Nn;in++) 
						{
							if (denom[in]<0.1) throw_runtime_error("Should not happen");
							this->internal_data_pt(iindex)->set_value(t,in,this->internal_data_pt(iindex)->value(t,in)/denom[in]); 
						}
					}
				}
			}
		}

		// DL and D0 fields and initial size
		if (quadtree_pt)
		{
			using namespace oomph::QuadTreeNames;
			for (unsigned t = 0; t < node_pt(0)->time_stepper_pt()->ntstorage(); t++)
			{
				for (unsigned int iindex = functable->info_DL.internal_offset_new; iindex < functable->info_DL.internal_offset_new+functable->info_DL.numfields; iindex++) // DL fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					double av = 0.0;
					for (unsigned ison = 0; ison < 4; ison++)
					{
						av += quadtree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					internal_data_pt(iindex)->set_value(t, 0, 0.25 * av);
					double slope1 = quadtree_pt->son_pt(SE)->object_pt()->internal_data_pt(iindex)->value(t, 0) - quadtree_pt->son_pt(SW)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					double slope2 = quadtree_pt->son_pt(NE)->object_pt()->internal_data_pt(iindex)->value(t, 0) - quadtree_pt->son_pt(NW)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 1, 0.5 * (slope1 + slope2));
					slope1 = quadtree_pt->son_pt(NE)->object_pt()->internal_data_pt(iindex)->value(t, 0) - quadtree_pt->son_pt(SE)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope2 = quadtree_pt->son_pt(NW)->object_pt()->internal_data_pt(iindex)->value(t, 0) - quadtree_pt->son_pt(SW)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 2, 0.5 * (slope1 + slope2));
				}
				for (unsigned int iindex = functable->info_D0.internal_offset_new; iindex < functable->info_D0.internal_offset_new + functable->info_D0.numfields; iindex++) // D0 fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					// TODO: Time history loop!
					double av = 0.0;
					for (unsigned ison = 0; ison < 4; ison++)
					{
						av += quadtree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					double avg_factor = 0.25;
					if (functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new] != 0.0) // TODO: Consider on DL as well
					{
						avg_factor = avg_factor * pow(avg_factor, -functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new]);
					}
					internal_data_pt(iindex)->set_value(t, 0, avg_factor * av);
				}
			}
			initial_cartesian_nondim_size = 0;
			for (unsigned ison = 0; ison < 4; ison++)
			{
				initial_cartesian_nondim_size += dynamic_cast<BulkElementBase *>(quadtree_pt->son_pt(ison)->object_pt())->initial_cartesian_nondim_size;
			}
			// std::cout << "REBUILT FROM SONS " << dynamic_cast<oomph::RefineableElement*>(this)->macro_elem_pt() << std::endl;
		}
		else if (binarytree_pt)
		{
			using namespace oomph::BinaryTreeNames;
			for (unsigned t = 0; t < node_pt(0)->time_stepper_pt()->ntstorage(); t++)
			{
				for (unsigned int iindex = functable->info_DL.internal_offset_new; iindex < functable->info_DL.internal_offset_new+functable->info_DL.numfields; iindex++) // DL fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					double av = 0.0;
					for (unsigned ison = 0; ison < 2; ison++)
					{
						av += binarytree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					internal_data_pt(iindex)->set_value(t, 0, 0.5 * av);
					double slope1 = binarytree_pt->son_pt(R)->object_pt()->internal_data_pt(iindex)->value(t, 0) - binarytree_pt->son_pt(L)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 1, slope1);
				}
				for (unsigned int iindex = functable->info_D0.internal_offset_new; iindex < functable->info_D0.internal_offset_new + functable->info_D0.numfields; iindex++) // D0 fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					double av = 0.0;
					for (unsigned ison = 0; ison < 2; ison++)
					{
						av += binarytree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					double avg_factor = 0.5;
					if (functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new] != 0.0) // TODO: Consider on DL as well
					{
						avg_factor = avg_factor * pow(avg_factor, -functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new]);
					}
					internal_data_pt(iindex)->set_value(t, 0, avg_factor * av);
				}
			}
			initial_cartesian_nondim_size = 0;
			for (unsigned ison = 0; ison < 2; ison++)
			{
				initial_cartesian_nondim_size += dynamic_cast<BulkElementBase *>(binarytree_pt->son_pt(ison)->object_pt())->initial_cartesian_nondim_size;
			}
		}
		else if (octree_pt)
		{
			using namespace oomph::OcTreeNames;
			for (unsigned t = 0; t < node_pt(0)->time_stepper_pt()->ntstorage(); t++)
			{
				for (unsigned int iindex = functable->info_DL.internal_offset_new; iindex < functable->info_DL.internal_offset_new+functable->info_DL.numfields; iindex++) // DL fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					double av = 0.0;
					for (unsigned ison = 0; ison < 8; ison++)
					{
						av += octree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					internal_data_pt(iindex)->set_value(t, 0, 0.125 * av);

					double slope1 = octree_pt->son_pt(RDB)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LDB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					double slope2 = octree_pt->son_pt(RUB)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LUB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					double slope3 = octree_pt->son_pt(RDF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LDF)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					double slope4 = octree_pt->son_pt(RUF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LUF)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 1, 0.25 * (slope1 + slope2 + slope3 + slope4));

					slope1 = octree_pt->son_pt(LUB)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LDB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope2 = octree_pt->son_pt(RUB)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(RDB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope3 = octree_pt->son_pt(LUF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LDF)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope4 = octree_pt->son_pt(RUF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(RDF)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 2, 0.25 * (slope1 + slope2 + slope3 + slope4));

					slope1 = octree_pt->son_pt(LDF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LDB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope2 = octree_pt->son_pt(RDF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(RDB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope3 = octree_pt->son_pt(LUF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(LUB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					slope4 = octree_pt->son_pt(RUF)->object_pt()->internal_data_pt(iindex)->value(t, 0) - octree_pt->son_pt(RUB)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					internal_data_pt(iindex)->set_value(t, 3, 0.25 * (slope1 + slope2 + slope3 + slope4));
				}
				for (unsigned int iindex = functable->info_D0.internal_offset_new; iindex < functable->info_D0.internal_offset_new + functable->info_D0.numfields; iindex++) // D0 fields
				{
					// XXX TODO: Allow for other interpolation methods. In particular, this does not conserve (which does not matter for e.g. pressure) and does not consider axisymmetry
					// TODO: Time history loop!
					double av = 0.0;
					for (unsigned ison = 0; ison < 8; ison++)
					{
						av += octree_pt->son_pt(ison)->object_pt()->internal_data_pt(iindex)->value(t, 0);
					}
					double avg_factor = 0.125;
					if (functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new] != 0.0) // TODO: Consider on DL as well
					{
						avg_factor = avg_factor * pow(avg_factor, -functable->discontinuous_refinement_exponents[functable->info_D0.buffer_offset_basebulk + iindex-functable->info_D0.internal_offset_new]);
					}
					internal_data_pt(iindex)->set_value(t, 0, avg_factor * av);
				}
			}
			initial_cartesian_nondim_size = 0;
			for (unsigned ison = 0; ison < 8; ison++)
			{
				initial_cartesian_nondim_size += dynamic_cast<BulkElementBase *>(octree_pt->son_pt(ison)->object_pt())->initial_cartesian_nondim_size;
			}
		}

		else
			throw_runtime_error("IMPLEMENT");
	}

	// Son face -> father face for the isotropic split schemes currently implemented. The map depends
	// only on the element SHAPE and the split scheme, never on the polynomial order, so everything is
	// dispatched here rather than duplicated over the C1/C2/C1TB/... variants of each shape.
	//
	// son_type is the son index handed to construct_son() by DynamicTree::dynamic_split_if_required,
	// which is exactly the tree's son_type, i.e. the QuadTreeNames (SW=0,SE=1,NW=2,NE=3) /
	// OcTreeNames (LDB=0..RUF=7) ordering that the build() machinery assumes.
	int BulkElementBase::face_index_in_father(const int &my_face_index, const unsigned &son_type, const BulkElementBase *father_el) const
	{
		const unsigned d = this->dim();

		// Pyramid father: the heterogeneous "red" split (6 sub-pyramids + 4 tets). Dispatch on the FATHER (a
		// tet son of a pyramid must NOT go through the tet-in-tet branch below). `this` is the son; son_type
		// 0..5 => a sub-pyramid (5 faces), 6..9 => a tet (4 faces). Evaluated from son_vertices_in_father (the
		// son's vertices in FATHER local coords) against the 5 father face planes -- matching
		// PyramidElementC1's facet definitions: F0 s1=0 {0,1,4}, F1 s0+s2=1 {1,2,4}, F2 s1+s2=1 {2,3,4},
		// F3 s0=0 {0,3,4}, F4 s2=0 (quad base) {0,1,2,3}. A son face lies on father face F iff all its corner
		// nodes satisfy F's plane; coordinates are 0, 1/2 or 1 so the test is exact.
		if (father_el && dynamic_cast<const oomph::RefineablePyramidElement *>(father_el))
		{
			static const int X = FACE_INTERIOR_IN_FATHER;
			oomph::Vector<oomph::Vector<double>> sv;
			oomph::RefineablePyramidElement::son_vertices_in_father((int)son_type, sv);
			std::vector<std::vector<int>> FN; // this son's faces -> its own vertex indices
			if (son_type < 6)
			{
				FN = {{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {0, 3, 4}, {0, 1, 2, 3}}; // sub-pyramid: 4 tri + quad base
				if (my_face_index < 0 || my_face_index > 4) return X;
			}
			else
			{
				FN = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}}; // tet: face k opposite vertex k
				if (my_face_index < 0 || my_face_index > 3) return X;
			}
			const std::vector<int> &corners = FN[my_face_index];
			const double tol = 1e-12;
			for (int F = 0; F < 5; F++)
			{
				bool all_on = true;
				for (int vi : corners)
				{
					const double s0 = sv[vi][0], s1 = sv[vi][1], s2 = sv[vi][2];
					double resid;
					if (F == 0) resid = s1;
					else if (F == 1) resid = s0 + s2 - 1.0;
					else if (F == 2) resid = s1 + s2 - 1.0;
					else if (F == 3) resid = s0;
					else resid = s2;
					if (std::fabs(resid) > tol) { all_on = false; break; }
				}
				if (all_on) return F;
			}
			return X;
		}

		if (d == 1)
		{
			// BinaryTree: son 0 = L (s<0 half), son 1 = R. Faces are the two end points -1/+1.
			if (son_type == 0) return (my_face_index == -1 ? -1 : FACE_INTERIOR_IN_FATHER);
			return (my_face_index == 1 ? 1 : FACE_INTERIOR_IN_FATHER);
		}

		if (dynamic_cast<const oomph::QElementBase *>(this))
		{
			// Tensor-product bisection: the son occupies the half of the father selected, per
			// direction, by one bit of son_type (QuadTree: bit0 = E, bit1 = N; OcTree: bit0 = R,
			// bit1 = U, bit2 = F). A son face survives as a father face iff it is the outer face in
			// that direction, i.e. its sign matches the son's half; the opposite face is interior.
			// Face index conventions (get_vertex_nodes_of_face): +/-1 = s0 (W/E, L/R), +/-2 = s1
			// (S/N, D/U), +/-3 = s2 (B/F).
			const int dir = std::abs(my_face_index);        // 1, 2 or 3
			if (dir < 1 || dir > (int)d) return FACE_INTERIOR_IN_FATHER;
			const bool son_is_upper = ((son_type >> (dir - 1)) & 1u) != 0u;
			const bool face_is_upper = (my_face_index > 0);
			return (son_is_upper == face_is_upper ? my_face_index : FACE_INTERIOR_IN_FATHER);
		}

		if (d == 2 && dynamic_cast<const oomph::TElementBase *>(this))
		{
			// Triangle 1->4 split on the QuadTree son names. Derived from
			// RefineableTElement<2>::setup_father_bounds (refineable_telements.cpp): the father edge
			// names are E = v0-v1, W = v1-v2, S = v2-v0, and the son corner nodes coinciding with
			// father vertices are SE=v0, NW=v1, SW=v2. pyoomph's triangle face indices (see
			// BulkElementTri2dC1::get_vertex_nodes_of_face) are face 0 = {v2,v1} = W, face 1 =
			// {v2,v0} = S, face 2 = {v0,v1} = E. Reading Father_bound for each son node and
			// intersecting over the two end nodes of each son edge gives:
			//   SW (corner at v2): face0 -> W(0), face1 -> S(1), face2 interior
			//   SE (corner at v0): face1 -> S(1), face2 -> E(2), face0 interior
			//   NW (corner at v1): face0 -> W(0), face2 -> E(2), face1 interior
			//   NE (the central son): every face interior
			// Cross-check: each father face is covered by exactly two sons, the central son by none.
			static const int X = FACE_INTERIOR_IN_FATHER;
			static const int tri_map[4][3] = {
				/* SW */ {0, 1, X},
				/* SE */ {X, 1, 2},
				/* NW */ {0, X, 2},
				/* NE */ {X, X, X}};
			if (son_type > 3 || my_face_index < 0 || my_face_index > 2) return FACE_INTERIOR_IN_FATHER;
			return tri_map[son_type][my_face_index];
		}

		if (d == 3 && dynamic_cast<const oomph::TElementBase *>(this))
		{
			// Tetrahedron 1->8 split. Rather than hard-coding a table (the four octahedron sons are
			// orientation-corrected, so the son-local vertex numbering is not obvious), evaluate it
			// from the very same geometric description the split itself uses:
			// RefineableTElement<3>::son_vertices_in_father gives the son's 4 vertices in FATHER
			// local coordinates, and the father's faces are the coordinate planes s0=0 (face 0),
			// s1=0 (face 1), s2=0 (face 2) and s0+s1+s2=1 (face 3) -- face k is opposite vertex k,
			// matching BulkElementTetra3dC1::get_vertex_nodes_of_face. Son face k consists of the
			// three son vertices other than k; it lies on a father face iff all three satisfy that
			// face's equation. All coordinates involved are 0, 1/2 or 1, so the test is exact.
			if (son_type > 7 || my_face_index < 0 || my_face_index > 3) return FACE_INTERIOR_IN_FATHER;
			oomph::Vector<oomph::Vector<double>> sv;
			oomph::RefineableTElement<3>::son_vertices_in_father((int)son_type, sv);
			const double tol = 1e-12;
			for (int f = 0; f < 4; f++)
			{
				bool all_on = true;
				for (int v = 0; v < 4 && all_on; v++)
				{
					if (v == my_face_index) continue; // the vertex opposite this son face
					const double s0 = sv[v][0], s1 = sv[v][1], s2 = sv[v][2];
					double resid;
					if (f == 0) resid = s0;
					else if (f == 1) resid = s1;
					else if (f == 2) resid = s2;
					else resid = s0 + s1 + s2 - 1.0;
					if (std::fabs(resid) > tol) all_on = false;
				}
				if (all_on) return f;
			}
			return FACE_INTERIOR_IN_FATHER;
		}

		if (d == 3 && dynamic_cast<const oomph::RefineableWedgeElement *>(this))
		{
			// Wedge 1->8 split. Evaluated from son_vertices_in_father (the son's 6 vertices in FATHER local
			// coordinates), like the tet: the father's 5 faces are the planes s2=0 (face 0), s2=1 (face 1),
			// s0=0 (face 2), s1=0 (face 3) and s0+s1=1 (face 4) -- matching BulkElementWedge3dC1::
			// get_vertex_nodes_of_face. A son face (its corner nodes listed below) lies on father face F iff
			// all its corners satisfy F's plane equation; coordinates are all 0, 1/2 or 1 so the test is exact.
			static const int X = FACE_INTERIOR_IN_FATHER;
			if (son_type > 7 || my_face_index < 0 || my_face_index > 4) return X;
			static const int FACE_NODES[5][4] = {{0, 1, 2, -1}, {3, 4, 5, -1}, {0, 2, 3, 5}, {0, 1, 3, 4}, {1, 2, 4, 5}};
			oomph::Vector<oomph::Vector<double>> sv;
			oomph::RefineableWedgeElement::son_vertices_in_father((int)son_type, sv);
			const double tol = 1e-12;
			for (int f = 0; f < 5; f++)
			{
				bool all_on = true;
				for (int k = 0; k < 4 && all_on; k++)
				{
					const int vi = FACE_NODES[my_face_index][k];
					if (vi < 0) continue; // triangular face has only 3 nodes
					const double s0 = sv[vi][0], s1 = sv[vi][1], s2 = sv[vi][2];
					double resid;
					if (f == 0) resid = s2;
					else if (f == 1) resid = s2 - 1.0;
					else if (f == 2) resid = s0;
					else if (f == 3) resid = s1;
					else resid = s0 + s1 - 1.0;
					if (std::fabs(resid) > tol) all_on = false;
				}
				if (all_on) return f;
			}
			return X;
		}

		// Any other shape (pyramids) has no implemented refinement scheme, so this is only reachable if such
		// an element is ever split -- fail loudly rather than silently dropping the boundary tags of its sons.
		throw_runtime_error(std::string("face_index_in_father is not implemented for the element type ") + typeid(*this).name());
		return FACE_INTERIOR_IN_FATHER;
	}

	// Default split scheme: isotropic subdivision into required_nsons() sons of the same type.
	const RefinementPattern *BulkElementBase::refinement_pattern() const
	{
		return IsotropicSameTypeRefinementPattern::instance();
	}

	// --- IsotropicSameTypeRefinementPattern ---------------------------------------------------
	// Reproduces the historical dynamic_split() behaviour: N = required_nsons() sons, each a
	// create_son_instance() of the same element type.
	unsigned IsotropicSameTypeRefinementPattern::nsons(const BulkElementBase *parent) const
	{
		return parent->required_nsons();
	}

	BulkElementBase *IsotropicSameTypeRefinementPattern::construct_son(const BulkElementBase *parent, unsigned) const
	{
		return parent->create_son_instance();
	}

	const IsotropicSameTypeRefinementPattern *IsotropicSameTypeRefinementPattern::instance()
	{
		static const IsotropicSameTypeRefinementPattern the_instance;
		return &the_instance;
	}

	// --- PyramidMixedRefinementPattern (pyramid -> 6 pyramids + 4 tets) -----------------------
	unsigned PyramidMixedRefinementPattern::nsons(const BulkElementBase *) const { return 10; }

	BulkElementBase *PyramidMixedRefinementPattern::construct_son(const BulkElementBase *parent, unsigned ison) const
	{
		// Sons 0..5 are sub-pyramids (same shape/order as the parent), sons 6..9 are tetrahedra of the
		// matching order. Both are built with the parent's physics via the concrete pyramid element's
		// factories. The parent can be a C1 or a C2 pyramid; the tet-son factory is virtual-dispatched to
		// build a tet of the same order.
		if (ison < 6) return parent->create_son_instance();
		if (const BulkElementPyramid3dC1 *pyr1 = dynamic_cast<const BulkElementPyramid3dC1 *>(parent))
			return pyr1->create_tet_son_instance();
		if (const BulkElementPyramid3dC2 *pyr2 = dynamic_cast<const BulkElementPyramid3dC2 *>(parent))
			return pyr2->create_tet_son_instance();
		throw_runtime_error("PyramidMixedRefinementPattern applied to a non-pyramid element");
	}

	const PyramidMixedRefinementPattern *PyramidMixedRefinementPattern::instance()
	{
		static const PyramidMixedRefinementPattern the_instance;
		return &the_instance;
	}

	// oomph-lib refinement hook: creates the son elements (via the current refinement_pattern())
	// and initializes their refinement level and initial size fraction; the sons are then filled in
	// by pre_build()/further_build() as the mesh machinery proceeds.
	void BulkElementBase::dynamic_split(oomph::Vector<BulkElementBase *> &son_pt) const
	{
		// std::cout << "DYN SPLIT " << std::endl;
		const RefinementPattern *pattern = this->refinement_pattern();
		int son_refine_level = Refine_level + 1;
		unsigned n_sons = pattern->nsons(this);
		son_pt.resize(n_sons);
		for (unsigned i = 0; i < n_sons; i++)
		{
			// std::cout << "C SON INST" << std::endl;
			son_pt[i] = pattern->construct_son(this, i);
			// std::cout << "SET REF" << std::endl;
			son_pt[i]->set_refinement_level(son_refine_level);
			son_pt[i]->initial_cartesian_nondim_size = this->initial_cartesian_nondim_size / ((double)n_sons);
			// Propagate the per-face boundary tags: each son face that is part of a father face
			// inherits that father face's boundaries; son faces interior to the father get nothing.
			// This is what keeps boundary-element identification exact under (non-uniform)
			// refinement, without depending on the ancestry surviving (the tree forest may be
			// re-rooted, see TemplatedMeshBase2d::setup_quadtree_forest).
			if (!face_boundaries.empty())
			{
				for (int son_face : son_pt[i]->get_possible_face_indices())
				{
					const int father_face = son_pt[i]->face_index_in_father(son_face, i, this);
					if (father_face == FACE_INTERIOR_IN_FATHER) continue;
					const std::vector<unsigned> *fb = this->get_face_boundaries(father_face);
					if (fb) son_pt[i]->set_face_boundaries(son_face, *fb);
				}
			}
		}
	}

	// Fill this element as a C1 son of a pyramid father (mixed 6-pyramid + 4-tet red split); see the header.
	void BulkElementBase::build_as_pyramid_son(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt)
	{
		using oomph::RefineablePyramidElement;
		const unsigned n_node = this->nnode();

		oomph::Tree *my_tree = this->tree_pt();
		const int son_type = my_tree->son_type();
		oomph::FiniteElement *father_el_pt = dynamic_cast<oomph::FiniteElement *>(my_tree->father_pt()->object_pt());
		oomph::RefineableElement *father_re = dynamic_cast<oomph::RefineableElement *>(father_el_pt);
		// Curved-boundary support: the son takes over the father's macro element together with its own
		// region of the macro reference domain, expressed as its vertices in the father's coordinates.
		// Set up after sv is known, below.
		oomph::TimeStepper *time_stepper_pt = father_el_pt->node_pt(0)->time_stepper_pt();
		const unsigned ntstorage = time_stepper_pt->ntstorage();
		const unsigned nfath = father_el_pt->nnode();
#ifdef OOMPH_HAS_MPI
		// Propagate the halo-ownership tag father->son (see RefineableTElement<3>::build): without it a
		// refined simplex/mixed son is mis-classified under MPI and the distributed halo/haloed node lists
		// drift between ranks. Serial builds have a non-halo (-1) father, so this is a no-op there.
		if (father_el_pt->is_halo()) this->set_halo(father_el_pt->non_halo_proc_ID());
#endif

		oomph::Vector<oomph::Vector<double>> sv;
		RefineablePyramidElement::son_vertices_in_father(son_type, sv);
		const unsigned nvert = sv.size(); // 5 for a pyramid son, 4 for a tet son
		// The son GEOMETRY is C1 (sub-parametric even for a C2 field element): son-local -> father-local is
		// the son's C1 shape interpolating its vertices' father-local coords. A pyramid son's C1 shape has
		// the 1/(1-s2) apex singularity, so a node exactly at the son apex (s2=1) is mapped directly to the
		// son's apex vertex sv[4] rather than through the singular shape.
		const bool is_pyr_son = (dynamic_cast<oomph::RefineablePyramidElement *>(this) != nullptr);

		// Curved boundaries: take over the father's macro element together with this son's region of the
		// macro reference domain -- its own vertices, expressed in the father's local coordinates, which
		// is exactly sv. Because the conversion runs through the father's C1 shape functions it does not
		// care that a pyramid father can have tet sons: the two shapes never meet in one expression.
		if (father_el_pt->macro_elem_pt() != 0)
		{
			BulkElementBase *father_be = dynamic_cast<BulkElementBase *>(father_el_pt);
			if (father_be)
			{
				std::vector<std::vector<double>> son_vertices(nvert, std::vector<double>(3, 0.0));
				for (unsigned int v = 0; v < nvert; v++)
					for (unsigned int i = 0; i < 3; i++) son_vertices[v][i] = sv[v][i];
				this->inherit_macro_element_from_father(father_be, son_vertices);
			}
		}

		for (unsigned j = 0; j < n_node; j++)
		{
			oomph::Vector<double> s_son(3);
			this->local_coordinate_of_node(j, s_son);
			oomph::Vector<double> s(3, 0.0);
			if (is_pyr_son && s_son[2] >= 1.0 - 1e-9)
			{
				s = sv[4];
			}
			else
			{
				oomph::Shape gpsi(nvert);
				this->shape_at_s_C1(s_son, gpsi);
				for (unsigned k = 0; k < nvert; k++)
					for (int d = 0; d < 3; d++) s[d] += gpsi[k] * sv[k][d];
			}

			// (1) Reuse a father node coincident with this position (shared vertex). Its values are already
			// correct (it is an existing father node) -- do NOT re-interpolate them: the pyramid shape's
			// 1/(1-s2) is singular at the apex s2=1 (the father apex is exactly such a reused node), so
			// get_interpolated_values there would return inf and corrupt the shared node.
			oomph::Node *created_node_pt = father_el_pt->get_node_at_local_coordinate(s);
			if (created_node_pt != 0)
			{
				this->node_pt(j) = created_node_pt;
				continue;
			}

			// Generating father nodes = those with POSITIVE father shape at s (for C1: the endpoints of the
			// edge this node bisects, or the four base corners for the base centre). The registry key is the
			// (father node, rounded weight) PAIRS, not the bare node set: for C2 two distinct interior points
			// can share the same positive-node set (the 1/4 and 3/4 points of a father edge), so the weight is
			// needed to tell them apart, while a shared face/edge node still gets identical pairs from every
			// adjacent father (off-face node shapes vanish on the face) -- shared, not torn. See the header.
			oomph::Shape psi(nfath);
			father_el_pt->shape(s, psi);
			std::vector<oomph::Node *> gen;
			oomph::RefineablePyramidElement::SharedNodeKey reg_key;
			for (unsigned l = 0; l < nfath; l++)
				if (psi(l) > 1e-6)
				{
					gen.push_back(father_el_pt->node_pt(l));
					reg_key.insert(std::make_pair(father_el_pt->node_pt(l), (long long)std::llround(psi(l) * 1e6)));
				}

			// (2) Reuse a node already created this round -- by another of this father's sons, OR (at level
			// >= 2) by an adjacent tet-of-pyramid whose ordinary tet build keys the same shared face node on
			// the same father pointers via this same registry (RefineableTElement<3>::in_pyramid_forest).
			if (!reg_key.empty())
			{
				auto it = RefineablePyramidElement::Shared_node_registry.find(reg_key);
				if (it != RefineablePyramidElement::Shared_node_registry.end()) { this->node_pt(j) = it->second; continue; }
			}

			// (2b) Cross-round: reuse a node built in an EARLIER round -- notably by a neighbour refined
			// before this one -- which the per-round registry above cannot see. Without this a shared node is
			// DUPLICATED under multi-level (non-uniform) refinement, tearing the mesh. Same key, looked up in
			// the snapshot rebuilt from the live mesh at the start of the round: the nodes carry the key they
			// were born with (pyoomph::Node::refinement_generating_key), and both sides of a facet compute the
			// same key, because both compute it from their own element at the level where the node is born and
			// those two elements share the facet's nodes. Only for genuine shared (non-interior) nodes.
			if (!reg_key.empty())
			{
				if (oomph::Node *ex = oomph::RefineablePyramidElement::find_node_in_snapshot(reg_key)) { this->node_pt(j) = ex; continue; }
			}

			// (3) Build a new node (boundary iff all generating nodes share a boundary; pinned iff all pinned).
			std::set<unsigned> boundaries;
			bool have_bounds = false;
			for (oomph::Node *g : gen)
			{
				oomph::BoundaryNodeBase *bg = dynamic_cast<oomph::BoundaryNodeBase *>(g);
				std::set<unsigned> *sg = 0;
				if (bg) bg->get_boundaries_pt(sg);
				if (!sg) { boundaries.clear(); break; }
				if (!have_bounds) { boundaries = *sg; have_bounds = true; }
				else
				{
					std::set<unsigned> inter;
					std::set_intersection(boundaries.begin(), boundaries.end(), sg->begin(), sg->end(), std::inserter(inter, inter.begin()));
					boundaries.swap(inter);
				}
			}

			if (!boundaries.empty())
			{
				created_node_pt = this->construct_boundary_node(j, time_stepper_pt);
				const unsigned nval = created_node_pt->nvalue();
				for (unsigned k = 0; k < nval; k++)
				{
					bool all_pinned = true;
					for (oomph::Node *g : gen) if (!g->is_pinned(k)) { all_pinned = false; break; }
					if (all_pinned) created_node_pt->pin(k);
				}
				for (std::set<unsigned>::iterator it = boundaries.begin(); it != boundaries.end(); ++it)
				{
					mesh_pt->add_boundary_node(*it, created_node_pt);
					if (mesh_pt->boundary_coordinate_exists(*it))
					{
						oomph::Vector<double> z;
						for (oomph::Node *g : gen)
						{
							oomph::Vector<double> zg;
							dynamic_cast<oomph::BoundaryNodeBase *>(g)->get_coordinates_on_boundary(*it, zg);
							if (z.empty()) z.resize(zg.size(), 0.0);
							for (unsigned zi = 0; zi < zg.size(); zi++) z[zi] += zg[zi] / gen.size();
						}
						created_node_pt->set_coordinates_on_boundary(*it, z);
					}
				}
			}
			else
			{
				created_node_pt = this->construct_node(j, time_stepper_pt);
			}

			this->node_pt(j) = created_node_pt;
			new_node_pt.push_back(created_node_pt);
			for (unsigned t = 0; t < ntstorage; t++)
			{
				oomph::Vector<double> xp(3);
				father_el_pt->get_x(t, s, xp);
				for (int d = 0; d < 3; d++) created_node_pt->x(t, d) = xp[d];
			}
			if (oomph::SolidNode *sn = dynamic_cast<oomph::SolidNode *>(created_node_pt))
			{
				const unsigned nl = sn->nlagrangian();
				for (unsigned i = 0; i < nl; i++)
				{
					double xi = 0.0;
					for (unsigned l = 0; l < nfath; l++)
						if (oomph::SolidNode *fn = dynamic_cast<oomph::SolidNode *>(father_el_pt->node_pt(l))) xi += psi(l) * fn->xi(i);
					sn->xi(i) = xi;
				}
			}
			for (unsigned t = 0; t < ntstorage; t++)
			{
				oomph::Vector<double> prev;
				father_re->get_interpolated_values(t, s, prev);
				const unsigned nv = std::min((unsigned)created_node_pt->nvalue(), (unsigned)prev.size());
				for (unsigned k = 0; k < nv; k++) created_node_pt->set_value(t, k, prev[k]);
			}
			mesh_pt->add_node_pt(created_node_pt);
			if (!reg_key.empty())
			{
				RefineablePyramidElement::Shared_node_registry[reg_key] = created_node_pt;
				// Key remembered on the node for the next round's snapshot -- see
				// pyoomph::Node::refinement_generating_key and step (2b) above.
				if (pyoomph::NodeWithFieldIndicesBase *pn = dynamic_cast<pyoomph::NodeWithFieldIndicesBase *>(created_node_pt))
					pn->set_refinement_generating_key(std::vector<std::pair<oomph::Node *, long long>>(reg_key.begin(), reg_key.end()));
			}
		}
	}

	// Fill this element as a son of a brick father in a MIXED forest (see the header). Mirrors
	// build_as_pyramid_son node-for-node, but the son node's father-local coordinate is the octree 1->8
	// affine image get_nodal_s_in_father(j) (no son-vertex/apex handling -- a brick has no singular vertex),
	// and every new node is shared through RefineablePyramidElement::Shared_node_registry keyed on the father
	// brick's positive-shape nodes + rounded weight -- identical keys to an adjacent pyramid/wedge on a shared
	// quad face (off-face shapes vanish there), and to another brick son on a shared brick face.
	// The son's C1 vertices expressed in its father's local coordinates, for the mixed-forest builders
	// that map node-by-node rather than exposing a vertex list. Matches each reference vertex of `shape`
	// to the son node sitting there, then asks the builder where that node lies in the father.
	static bool collect_son_vertex_coords_in_father(BulkElementBase *son, const MacroElementShape &shape,
	                                                std::vector<std::vector<double>> &sv)
	{
		std::vector<std::vector<double>> refv;
		macro_reference_vertices(shape, refv);
		sv.assign(refv.size(), std::vector<double>(3, 0.0));
		for (unsigned int v = 0; v < refv.size(); v++)
		{
			bool found = false;
			for (unsigned int j = 0; j < son->nnode() && !found; j++)
			{
				oomph::Vector<double> sj;
				son->local_coordinate_of_node(j, sj);
				double d = 0.0;
				for (unsigned int i = 0; i < refv[v].size() && i < sj.size(); i++) d = std::max(d, std::fabs(sj[i] - refv[v][i]));
				if (d < 1e-10)
				{
					oomph::Vector<double> sf(3, 0.0);
					son->get_nodal_s_in_father(j, sf);
					for (unsigned int i = 0; i < 3 && i < sf.size(); i++) sv[v][i] = sf[i];
					found = true;
				}
			}
			if (!found) return false;
		}
		return true;
	}

	void BulkElementBase::build_as_brick_son(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt)
	{
		using oomph::RefineablePyramidElement;
		const unsigned n_node = this->nnode();

		oomph::FiniteElement *father_el_pt = dynamic_cast<oomph::FiniteElement *>(this->tree_pt()->father_pt()->object_pt());
		oomph::RefineableElement *father_re = dynamic_cast<oomph::RefineableElement *>(father_el_pt);
		// As in build_as_pyramid_son, but a brick son maps node-by-node rather than from a vertex list,
		// so recover its eight vertices in the father's coordinates first.
		if (father_el_pt->macro_elem_pt() != 0)
		{
			BulkElementBase *father_be = dynamic_cast<BulkElementBase *>(father_el_pt);
			std::vector<std::vector<double>> son_vertices;
			if (father_be && collect_son_vertex_coords_in_father(this, MacroElementShape::Brick3d, son_vertices))
				this->inherit_macro_element_from_father(father_be, son_vertices);
		}
		oomph::TimeStepper *time_stepper_pt = father_el_pt->node_pt(0)->time_stepper_pt();
		const unsigned ntstorage = time_stepper_pt->ntstorage();
		const unsigned nfath = father_el_pt->nnode();
#ifdef OOMPH_HAS_MPI
		// Propagate the halo-ownership tag father->son (see RefineableTElement<3>::build): without it a
		// refined simplex/mixed son is mis-classified under MPI and the distributed halo/haloed node lists
		// drift between ranks. Serial builds have a non-halo (-1) father, so this is a no-op there.
		if (father_el_pt->is_halo()) this->set_halo(father_el_pt->non_halo_proc_ID());
#endif

		for (unsigned j = 0; j < n_node; j++)
		{
			// Son node j's father-local coordinate (octree 1->8 affine map).
			oomph::Vector<double> s(3);
			this->get_nodal_s_in_father(j, s);

			// (1) Reuse a father node coincident with this position (its values are already correct).
			oomph::Node *created_node_pt = father_el_pt->get_node_at_local_coordinate(s);
			if (created_node_pt != 0) { this->node_pt(j) = created_node_pt; continue; }

			// Generating father nodes = those with POSITIVE father shape at s; key on (node, rounded weight)
			// pairs (the same weight-augmented key the pyramid/wedge builds use).
			oomph::Shape psi(nfath);
			father_el_pt->shape(s, psi);
			std::vector<oomph::Node *> gen;
			RefineablePyramidElement::SharedNodeKey reg_key;
			for (unsigned l = 0; l < nfath; l++)
				if (psi(l) > 1e-6)
				{
					gen.push_back(father_el_pt->node_pt(l));
					reg_key.insert(std::make_pair(father_el_pt->node_pt(l), (long long)std::llround(psi(l) * 1e6)));
				}

			// (2) Reuse a node another son (of this brick or an adjacent brick/pyramid/wedge) built this round.
			if (!reg_key.empty())
			{
				auto it = RefineablePyramidElement::Shared_node_registry.find(reg_key);
				if (it != RefineablePyramidElement::Shared_node_registry.end()) { this->node_pt(j) = it->second; continue; }
			}

			// (2b) Cross-round: reuse a node built in an EARLIER round -- notably by a neighbour refined
			// before this one -- which the per-round registry above cannot see. Without this a shared node is
			// DUPLICATED under multi-level (non-uniform) refinement, tearing the mesh. Same key, looked up in
			// the snapshot rebuilt from the live mesh at the start of the round: the nodes carry the key they
			// were born with (pyoomph::Node::refinement_generating_key), and both sides of a facet compute the
			// same key, because both compute it from their own element at the level where the node is born and
			// those two elements share the facet's nodes. Only for genuine shared (non-interior) nodes.
			if (!reg_key.empty())
			{
				if (oomph::Node *ex = oomph::RefineablePyramidElement::find_node_in_snapshot(reg_key)) { this->node_pt(j) = ex; continue; }
			}

			// (3) Build a new node (boundary iff all generating nodes share a boundary; pinned iff all pinned).
			std::set<unsigned> boundaries;
			bool have_bounds = false;
			for (oomph::Node *g : gen)
			{
				oomph::BoundaryNodeBase *bg = dynamic_cast<oomph::BoundaryNodeBase *>(g);
				std::set<unsigned> *sg = 0;
				if (bg) bg->get_boundaries_pt(sg);
				if (!sg) { boundaries.clear(); break; }
				if (!have_bounds) { boundaries = *sg; have_bounds = true; }
				else
				{
					std::set<unsigned> inter;
					std::set_intersection(boundaries.begin(), boundaries.end(), sg->begin(), sg->end(), std::inserter(inter, inter.begin()));
					boundaries.swap(inter);
				}
			}

			if (!boundaries.empty())
			{
				created_node_pt = this->construct_boundary_node(j, time_stepper_pt);
				const unsigned nval = created_node_pt->nvalue();
				for (unsigned k = 0; k < nval; k++)
				{
					bool all_pinned = true;
					for (oomph::Node *g : gen) if (!g->is_pinned(k)) { all_pinned = false; break; }
					if (all_pinned) created_node_pt->pin(k);
				}
				for (std::set<unsigned>::iterator it = boundaries.begin(); it != boundaries.end(); ++it)
				{
					mesh_pt->add_boundary_node(*it, created_node_pt);
					if (mesh_pt->boundary_coordinate_exists(*it))
					{
						oomph::Vector<double> z;
						for (oomph::Node *g : gen)
						{
							oomph::Vector<double> zg;
							dynamic_cast<oomph::BoundaryNodeBase *>(g)->get_coordinates_on_boundary(*it, zg);
							if (z.empty()) z.resize(zg.size(), 0.0);
							for (unsigned zi = 0; zi < zg.size(); zi++) z[zi] += zg[zi] / gen.size();
						}
						created_node_pt->set_coordinates_on_boundary(*it, z);
					}
				}
			}
			else
			{
				created_node_pt = this->construct_node(j, time_stepper_pt);
			}

			this->node_pt(j) = created_node_pt;
			new_node_pt.push_back(created_node_pt);
			for (unsigned t = 0; t < ntstorage; t++)
			{
				oomph::Vector<double> xp(3);
				father_el_pt->get_x(t, s, xp);
				for (int d = 0; d < 3; d++) created_node_pt->x(t, d) = xp[d];
			}
			if (oomph::SolidNode *sn = dynamic_cast<oomph::SolidNode *>(created_node_pt))
			{
				const unsigned nl = sn->nlagrangian();
				for (unsigned i = 0; i < nl; i++)
				{
					double xi = 0.0;
					for (unsigned l = 0; l < nfath; l++)
						if (oomph::SolidNode *fn = dynamic_cast<oomph::SolidNode *>(father_el_pt->node_pt(l))) xi += psi(l) * fn->xi(i);
					sn->xi(i) = xi;
				}
			}
			for (unsigned t = 0; t < ntstorage; t++)
			{
				oomph::Vector<double> prev;
				father_re->get_interpolated_values(t, s, prev);
				const unsigned nv = std::min((unsigned)created_node_pt->nvalue(), (unsigned)prev.size());
				for (unsigned k = 0; k < nv; k++) created_node_pt->set_value(t, k, prev[k]);
			}
			mesh_pt->add_node_pt(created_node_pt);
			if (!reg_key.empty())
			{
				RefineablePyramidElement::Shared_node_registry[reg_key] = created_node_pt;
				// Key remembered on the node for the next round's snapshot -- see
				// pyoomph::Node::refinement_generating_key and step (2b) above.
				if (pyoomph::NodeWithFieldIndicesBase *pn = dynamic_cast<pyoomph::NodeWithFieldIndicesBase *>(created_node_pt))
					pn->set_refinement_generating_key(std::vector<std::pair<oomph::Node *, long long>>(reg_key.begin(), reg_key.end()));
			}
		}
	}

	///////////////////////////////
	// RefineableSolidLineElement::build is oomph-lib's own pattern (mirrored here for the 1d solid/Lagrangian
	// case, which isn't provided out of the box) for constructing a refined son element: it copies/interpolates
	// the father element's nodal (Eulerian and Lagrangian) positions and history values onto the son's nodes,
	// selecting the left or right half of the father's local coordinate range depending on son_type.

	void RefineableSolidLineElement::build(oomph::Mesh *&mesh_pt, oomph::Vector<oomph::Node *> &new_node_pt,
										   bool &was_already_built,
										   std::ofstream &new_nodes_file)
	{
		using namespace oomph::BinaryTreeNames;
		oomph::RefineableQElement<1>::build(mesh_pt, new_node_pt, was_already_built, new_nodes_file);
		if (was_already_built)
			return;
		int son_type = Tree_pt->son_type();
		RefineableSolidLineElement *father_el_pt = dynamic_cast<RefineableSolidLineElement *>(Tree_pt->father_pt()->object_pt());
#ifdef PARANOID
		if (static_cast<oomph::SolidNode *>(father_el_pt->node_pt(0))->nlagrangian_type() != 1)
		{
			throw oomph::OomphLibError(
				"We can't handle generalised nodal positions (yet).\n",
				OOMPH_CURRENT_FUNCTION,
				OOMPH_EXCEPTION_LOCATION);
		}
#endif

		oomph::Vector<double> s_left(1);
		oomph::Vector<double> s_right(1);

		oomph::Vector<double> s(1);
		oomph::Vector<double> xi(1);
		oomph::Vector<double> xi_fe(1);
		oomph::Vector<double> x(1);
		oomph::Vector<double> x_fe(1);

		// In order to set up the vertex coordinates we need to know which
		// type of son the current element is
		switch (son_type)
		{
		case L:
			s_left[0] = -1.0;
			s_right[0] = 0.0;
			break;

		case R:
			s_left[0] = 0.0;
			s_right[0] = 1.0;
			break;
		}

		// Pass the undeformed macro element onto the son
		//  hierher why can I read this?
		if (father_el_pt->undeformed_macro_elem_pt() != 0)
		{
			throw_runtime_error("TODO: Check this");
			Undeformed_macro_elem_pt = father_el_pt->undeformed_macro_elem_pt();
			s_macro_ll(0) = father_el_pt->s_macro_ll(0) + 0.5 * (s_left[0] + 1.0) * (father_el_pt->s_macro_ur(0) - father_el_pt->s_macro_ll(0));
			s_macro_ur(0) = father_el_pt->s_macro_ll(0) + 0.5 * (s_right[0] + 1.0) * (father_el_pt->s_macro_ur(0) - father_el_pt->s_macro_ll(0));
		}

		unsigned n = 0;
		unsigned n_p = nnode_1d();
		for (unsigned i0 = 0; i0 < n_p; i0++)
		{
			s[0] = s_left[0] + (s_right[0] - s_left[0]) * double(i0) / double(n_p - 1);
			n = i0;
			father_el_pt->get_x_and_xi(s, x_fe, x, xi_fe, xi);
			oomph::SolidNode *elastic_node_pt = static_cast<oomph::SolidNode *>(node_pt(n));
			elastic_node_pt->x(0) = x_fe[0];
			if (Use_undeformed_macro_element_for_new_lagrangian_coords)
			{
				elastic_node_pt->xi(0) = xi[0];
			}
			else
			{
				elastic_node_pt->xi(0) = xi_fe[0];
			}
			oomph::TimeStepper *time_stepper_pt = father_el_pt->node_pt(0)->time_stepper_pt();
			unsigned ntstorage = time_stepper_pt->ntstorage();
			if (ntstorage != 1)
			{
				for (unsigned t = 1; t < ntstorage; t++)
				{
					elastic_node_pt->x(t, 0) = father_el_pt->interpolated_x(t, s, 0);
				}
			}
		}

		this->set_integration_scheme(father_el_pt->integral_pt());
	}
}
