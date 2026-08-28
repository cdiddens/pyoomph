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


// Element lifecycle and the JIT element_info/shape-buffer plumbing: construction and destruction,
// create_from_template, fill_element_info/free_element_info, the shape-buffer allocator, the
// macro-element (curved-boundary) mapping, the D0/DL/DG field accessors, node construction, and the
// paraview/Z2 output hooks. The rest of BulkElementBase is split by responsibility across the
// elements_*.cpp files listed in elements.hpp.

#include "macroelements.hpp"
#include "elements.hpp"
#include "elements_concrete.hpp"
#include "exception.hpp"
#include "problem.hpp"
#include "nodes.hpp"
#include "meshtemplate.hpp"
#include "expressions.hpp"
#include "timestepper.hpp"

#include <mutex>

extern "C"
{
	double _pyoomph_invoke_callback(void *functab, int jitindex, double *args, int numargs)
	{
		JITFuncSpec_Table_FiniteElement_t *ft = (JITFuncSpec_Table_FiniteElement_t *)functab;
		pyoomph::CustomMathExpressionBase *expr = (pyoomph::CustomMathExpressionBase *)ft->callback_infos[jitindex].cb_obj; // TODO: This may not be multiple inherited!!
		return expr->_call(args, numargs);
	}
	
	void _pyoomph_invoke_multi_ret(void * functab, int jitindex,int flag,double * arg_list,double * result_list, double * derivative_matrix, int numargs, int numret)
	{
		JITFuncSpec_Table_FiniteElement_t *ft = (JITFuncSpec_Table_FiniteElement_t *)functab;
		pyoomph::CustomMultiReturnExpressionBase *expr = (pyoomph::CustomMultiReturnExpressionBase *)ft->multi_ret_infos[jitindex].cb_obj; // TODO: This may not be multiple inherited!!		
		expr->_call(flag,arg_list, numargs,result_list,numret,derivative_matrix);
	}	

	// Called from generated code inside the integration-point loop. The element comes back out of
	// the eleminfo the generated function was handed; it used to come from a process-wide
	// _currently_assembled_element, which made the whole assembly path unparallelisable and left a
	// dangling pointer behind after a remesh.
	void _pyoomph_fill_shape_buffer_for_point(const JITElementInfo_t *eleminfo, unsigned index, JITFuncSpec_RequiredShapes_FiniteElement_t *required, int flag)
	{
		((pyoomph::BulkElementBase *)eleminfo->elem_ptr)->fill_shape_buffer_for_integration_point(index, *required, flag);
	}
}

namespace pyoomph
{
	size_t __shape_buffer_mem_usage = 0;
	void *counted_calloc(size_t num, size_t size)
	{
		__shape_buffer_mem_usage += num * size;
		return calloc(num, size);
	}

	// my_alloc/my_free/my_alloc_or_free: recursive helpers to allocate (or free) a nested,
	// variable-depth C array (used for the multi-dimensional shape-function buffers inside
	// JITShapeInfo_t, whose depth depends on which derivatives/spaces the generated code needs).
	// The scalar overload is the recursion base case (nothing to do for a non-pointer leaf type).
	template <typename T>
	void my_alloc(T) {}

	// Allocates a firstdim-sized array of T, then recurses on each of its elements with the
	// remaining "extra" dimensions (if any), building up a jagged/nested array of the requested depth.
	template <typename T, typename... Extra>
	void my_alloc(T * PYOOMPH_RESTRICT &  dest, size_t firstdim, const Extra &...extra)
	{
		if (!firstdim)
		{
			dest = NULL;
			return;
		}
		dest = (T *)counted_calloc(firstdim, sizeof(T));
		constexpr size_t remaining = sizeof...(extra);
		if (remaining > 0)
		{
			for (size_t c = 0; c < firstdim; c++)
			{
				my_alloc(dest[c], extra...);
			}
		}
	}

	template <typename T>
	void my_free(T) {}

	// Mirror of my_alloc: recursively frees a nested array previously built by my_alloc, then frees
	// the top-level pointer itself and nulls it out.
	template <typename T, typename... Extra>
	void my_free(T * PYOOMPH_RESTRICT &  dest, size_t firstdim, const Extra &...extra)
	{
		if (!dest)
			return;
		constexpr size_t remaining = sizeof...(extra);
		if (remaining > 0)
		{
			for (size_t c = 0; c < firstdim; c++)
			{
				my_free(dest[c], extra...);
			}
		}
		free(dest);
		dest = NULL;
	}

	// Convenience wrapper dispatching to my_alloc or my_free depending on "alloc", so call sites can
	// toggle allocation/deallocation of a shape buffer with a single boolean flag.
	template <typename T, typename... Extra>
	void my_alloc_or_free(bool alloc, T * PYOOMPH_RESTRICT &  dest, size_t firstdim, const Extra &...extra)
	{
		if (alloc)
			my_alloc(dest, firstdim, extra...);
		else
			my_free(dest, firstdim, extra...);
	}

	// Selects the per-shape integration-scheme map (see IntegrationSchemeStorage's member maps)
	// matching the requested (triangular/tetrahedral vs. quad/brick, spatial dimension, bubble-enriched) combination.
	std::map<unsigned, oomph::Integral *> &IntegrationSchemeStorage::get_integral_order_map(bool tri, unsigned edim, bool bubble)
	{
		if (tri)
		{
			if (bubble)
			{
				if (edim == 2)
					return T2dTB;
				else if (edim == 3)
					return T3dTB;
				else
					throw_runtime_error("Implement");
			}
			else
			{
				if (edim == 1)
					return T1d;
				else if (edim == 2)
					return T2d;
				else if (edim == 3)
					return T3d;
				else
					throw_runtime_error("Implement");
			}
		}
		else
		{
			if (edim == 1)
				return Q1d;
			else if (edim == 2)
				return Q2d;
			else if (edim == 3)
				return Q3d;
			else if (edim==4)
				return Wedge3d;
			else if (edim==5)
				return Pyramid3d;
			else
				throw_runtime_error("Implement");
		}
		throw_runtime_error("Invalid combination of tri/quad, edim and bubble");
	}

	// Deletes every oomph::Integral owned by "map" and clears it; called for each per-shape map from the destructor.
	void IntegrationSchemeStorage::clean_up_map(std::map<unsigned, oomph::Integral *> &map)
	{
		for (auto m : map)
		{
			delete m.second;
			m.second = NULL;
		}
		map.clear();
	}

	IntegrationSchemeStorage::~IntegrationSchemeStorage()
	{
		clean_up_map(Q1d);
		clean_up_map(Q2d);
		clean_up_map(Q3d);
		clean_up_map(T1d);
		clean_up_map(T2d);
		clean_up_map(T3d);
		clean_up_map(T2dTB);
		clean_up_map(T3dTB);
		clean_up_map(Wedge3d);
		clean_up_map(Pyramid3d);
	}

	// Pre-populates the integration-scheme maps with a fixed set of standard Gauss/TGauss/Wedge/Pyramid
	// quadrature rules at a handful of common orders; get_integration_scheme() falls back to the
	// closest available order above the requested one if an exact match isn't pre-built here.
	IntegrationSchemeStorage::IntegrationSchemeStorage()
	{
		Q1d[2] = new oomph::Gauss<1, 2>();
		Q1d[3] = new oomph::Gauss<1, 3>();
		Q1d[4] = new oomph::Gauss<1, 4>();

		Q2d[2] = new oomph::Gauss<2, 2>();
		Q2d[3] = new oomph::Gauss<2, 3>();
		Q2d[4] = new oomph::Gauss<2, 4>();

		Q3d[2] = new oomph::Gauss<3, 2>();
		Q3d[3] = new oomph::Gauss<3, 3>();
		Q3d[4] = new oomph::Gauss<3, 4>();

		T1d[2] = new oomph::TGauss<1, 2>();
		T1d[3] = new oomph::TGauss<1, 3>();
		T1d[4] = new oomph::TGauss<1, 4>();
		//   T1d[5]=new oomph::TGauss<1,5>(); // IS NOT IMPLEMETED!

		T2d[2] = new oomph::TGauss<2, 2>();
		T2d[3] = new oomph::TGauss<2, 3>();
		T2d[4] = new oomph::TGauss<2, 4>();
		//   T2d[5]=new oomph::TGauss<2,5>();  // Has the wrong weighting factor! element volume is twice as large!
		//  T2d[13]=new oomph::TGauss<2,13>(); // Has the wrong weighting factor! element volume is twice as large!

		T2dTB[3] = new oomph::TBubbleEnrichedGauss<2, 3>();

		T3d[2] = new oomph::TGauss<3, 2>();
		T3d[3] = new oomph::TGauss<3, 3>();
		T3d[5] = new oomph::TGauss<3, 5>();

		//TODO: Having a C1TB here?
		T3dTB[3] = new oomph::TBubbleEnrichedGauss<3, 3>();

		Wedge3d[2] = new oomph::WedgeGaussC1();
		Wedge3d[3] = new oomph::WedgeGaussC2();

		Pyramid3d[2]= new oomph::PyramidGaussC1();
		Pyramid3d[3] = new oomph::PyramidGaussC2();
	}

	// Returns the cached scheme for the exact requested order if available; otherwise picks the
	// scheme with the closest order strictly greater than requested (never a lower order, so
	// accuracy is not silently reduced), or the highest available order if none is greater.
	oomph::Integral *IntegrationSchemeStorage::get_integration_scheme(bool tris, unsigned edim, unsigned order, bool bubble)
	{
		std::map<unsigned, oomph::Integral *> &map = this->get_integral_order_map(tris, edim, bubble);
		if (map.count(order))
		{
			// std::cout << "FOR " << (tris ? "TRI" : "QUAD") << " OF DIM " << edim << " ORDER " << order << " WE HAVE " << typeid(map[order]).name();
			return map[order];
		}
		unsigned closestdist = 10000;
		oomph::Integral *ret = NULL;
		unsigned maxorder = 0;
		for (auto mentry : map)
		{
			maxorder = std::max(maxorder, mentry.first);
			if (order < mentry.first)
			{
				if (mentry.first - order < closestdist)
				{
					closestdist = mentry.first - order;
					ret = mentry.second;
				}
			}
		}

		if (ret)
		{
			return ret;
		}
		return map[maxorder];
	}


	// What the (process-wide, shared) shape buffer must be able to hold. The optional parts are kept
	// out of the buffer unless some code actually asks for them, and the buffer is grown
	// when a later instance needs more than the one already allocated (see BulkElementBase's ctor).
	//
	// The second-derivative arrays are sized from the nodal dimension rather than from the global
	// MAX_NODES/MAX_NODAL_DIM constants, because d2_d2x2_shape_dcoord is
	// nnode*MAX_N2DERIV*nnode*ndim*nnode*ndim doubles per space: 38 MB per space per buffer (i.e.
	// 765 MB over the five-buffer chain and four spaces) at nnode=27/ndim=3, but 93 kB for a 2D
	// problem. Bounding nnode by 3^ndim is safe - the largest element of dimension d is the C2
	// Q-element with 3^d nodes, and an element's dimension never exceeds the nodal dimension.
	struct ShapeBufferCaps
	{
		bool hessian_moving_mesh = false;   // int_pt_weights_d2_coords, d2_dx2_shape_dcoord, ...
		bool second_derivs = false;         // d2x_shapes, d2S_shapes
		bool second_derivs_dcoord = false;  // d_d2x_shape_dcoord (moving mesh)
		bool second_derivs_hessian = false; // d2_d2x2_shape_dcoord (moving mesh + analytic Hessian)
		unsigned second_deriv_nodal_dim = 0;
		unsigned second_deriv_nodes() const
		{
			if (!second_deriv_nodal_dim) return 0; // nothing requested -> allocate nothing
			unsigned n = 1;
			for (unsigned d = 0; d < second_deriv_nodal_dim; d++) n *= 3;
			return n;
		}
		// True if a buffer allocated for *this also satisfies everything "need" asks for.
		bool covers(const ShapeBufferCaps &need) const
		{
			return (hessian_moving_mesh || !need.hessian_moving_mesh) &&
				   (second_derivs || !need.second_derivs) &&
				   (second_derivs_dcoord || !need.second_derivs_dcoord) &&
				   (second_derivs_hessian || !need.second_derivs_hessian) &&
				   (second_deriv_nodal_dim >= need.second_deriv_nodal_dim);
		}
		void absorb(const ShapeBufferCaps &need)
		{
			hessian_moving_mesh |= need.hessian_moving_mesh;
			second_derivs |= need.second_derivs;
			second_derivs_dcoord |= need.second_derivs_dcoord;
			second_derivs_hessian |= need.second_derivs_hessian;
			if (need.second_deriv_nodal_dim > second_deriv_nodal_dim) second_deriv_nodal_dim = need.second_deriv_nodal_dim;
		}
	};

	// Defined below, next to the single-buffer allocator it recurses into.
	void alloc_dealloc_all_shape_buffers(bool do_alloc, JITShapeInfo_t **buff, const ShapeBufferCaps &caps);

	// Owner of the shape buffers. A "buffer" here is really the five-deep chain built by
	// alloc_dealloc_all_shape_buffers (the element's own, its bulk, its bulk's bulk, its opposite and
	// that one's bulk), which an interface element addresses through shape_info->bulk_shapeinfo rather
	// than through the neighbour's own member - so one chain serves the whole assembly of one element.
	//
	// There is deliberately no buffer per element: the chain reaches hundreds of megabytes for a 3D C2
	// element with analytic Hessians on a moving mesh (see the sizing note on ShapeBufferCaps above),
	// and an element only needs it while it is actually being assembled. What used to be a single
	// process-wide buffer is now a small pool, handed out one per concurrently assembling thread, so
	// that the element loop can be parallelised without every element writing into the same arrays.
	// In a serial run the pool holds exactly one chain, i.e. the memory footprint is unchanged and the
	// buffer contents follow the same sequence they always did.
	//
	// require() may only be called while nothing is assembling. It grows every chain in the pool,
	// which frees and reallocates their inner arrays; a thread mid-assembly would be left reading
	// freed memory. Element construction is the only caller and never overlaps assembly.
	class ShapeBufferPool
	{
		std::vector<JITShapeInfo_t *> all;   // every chain ever built, owned here
		std::vector<JITShapeInfo_t *> idle;  // the subset nobody is currently assembling into
		ShapeBufferCaps caps;                // what every chain in `all` was built with
		std::mutex mtx;

	public:
		void require(const ShapeBufferCaps &needed)
		{
			std::lock_guard<std::mutex> lock(mtx);
			if (caps.covers(needed))
				return;
			// Free with the caps each chain was allocated with (otherwise the recursive free walks the
			// wrong depths), then rebuild at the widened capabilities.
			for (auto *b : all)
				alloc_dealloc_all_shape_buffers(false, &b, caps);
			caps.absorb(needed);
			for (auto *b : all)
				alloc_dealloc_all_shape_buffers(true, &b, caps);
		}

		JITShapeInfo_t *acquire()
		{
			std::lock_guard<std::mutex> lock(mtx);
			if (!idle.empty())
			{
				JITShapeInfo_t *b = idle.back();
				idle.pop_back();
				return b;
			}
			JITShapeInfo_t *b = NULL;
			alloc_dealloc_all_shape_buffers(true, &b, caps);
			all.push_back(b);
			return b;
		}

		void release(JITShapeInfo_t *b)
		{
			if (!b)
				return;
			std::lock_guard<std::mutex> lock(mtx);
			idle.push_back(b);
		}
	};

	static ShapeBufferPool shape_buffer_pool;

	// The chain the calling thread is assembling into, acquired on first use and held until
	// release_thread_shape_buffer(). Holding it for the thread's lifetime rather than for one
	// element's assembly keeps the buffer contents in exactly the order the single global buffer
	// produced them, and costs one chain per worker thread rather than one per element.
	static thread_local JITShapeInfo_t *__thread_shape_buffer = NULL;

	// Whether any space anywhere in this required-shapes tree (including the bulk/opposite
	// sub-structures) asks for second spatial derivatives.
	static bool required_shapes_want_d2x(const JITFuncSpec_RequiredShapes_FiniteElement_t *r)
	{
		if (!r) return false;
		if (r->Pos.d2x_psi || r->DL.d2x_psi || r->D0.d2x_psi) return true;
		for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
			if (r->continuous_spaces[i].d2x_psi) return true;
		return required_shapes_want_d2x(r->bulk_shapes) || required_shapes_want_d2x(r->opposite_shapes);
	}
	thread_local DynamicJITCode *BulkElementBase::__CurrentJITCode = NULL;
	thread_local unsigned BulkElementBase::zeta_time_history = 0;
	thread_local unsigned BulkElementBase::zeta_coordinate_type = 0; // 0 means Lagrangian, 1 Eulerian, on co-dimensional meshes it will be the boundary coordinate (if set)
	bool BulkElementBase::detect_inverted_elements=false;
	unsigned BulkElementBase::inverted_elements_detected=0;
	std::string BulkElementBase::inverted_element_message;

	// Default (unimplemented) hook; concrete element types with higher-order interpolation on
	// faces override this to return the boundary node at the given local face index/position.
	oomph::Node *BulkElementBase::boundary_node_pt(const int &, const unsigned int)
	{
		throw_runtime_error("Implement");
	}

	// Rebuilds this element's external-data list from scratch based on the JIT code's
	// linked external data (data shared/globally coupled across elements, e.g. global parameters).
	const JITFuncSpec_RequiredShapes_FiniteElement_t *BulkElementBase::attachment_required_shapes(const JITFuncSpec_Table_FiniteElement_t *ft)
	{
		static const bool __disabled = getenv("PYOOMPH_DISABLE_ASSEMBLY_EXTDATA_SPLIT") != NULL;
		return __disabled ? &(ft->merged_required_shapes) : &(ft->assembly_required_shapes);
	}

	void BulkElementBase::ensure_external_data()
	{
		this->flush_external_data();
		for (auto &e : jitcode->linked_external_data.get_required_external_data())
		{
			this->add_external_data(e);
		}
	}

	// --- The get_D0/DL/DG_nodal_data / get_D0/DL/DG_buffer_index / get_D0/DL/DG_local_equation
	// family below are simple accessors translating between a discontinuous field's index (D0:
	// element-constant, DL: discontinuous-Lagrange, DG: discontinuous on a given interpolation
	// space) and the underlying oomph-lib Data object / generated-code buffer offset / local
	// equation number, using the offsets recorded in the JIT function table (ft) for this element's
	// generated code.
    oomph::Data *BulkElementBase::get_D0_nodal_data(const unsigned &fieldindex)
    {
		auto * ft=this->get_jit_code()->get_func_table();
		
        return this->internal_data_pt(ft->info_D0.internal_offset_new+fieldindex);
    }

	oomph::Data *BulkElementBase::get_DL_nodal_data(const unsigned &fieldindex)
    {
		auto * ft=this->get_jit_code()->get_func_table();
        return this->internal_data_pt(ft->info_DL.internal_offset_new+fieldindex);
    }

	oomph::Data *BulkElementBase::get_DG_nodal_data(const unsigned &space_index,const unsigned &fieldindex)
	{
		auto * ft=this->get_jit_code()->get_func_table();		
		const JITFuncSpec_Table_FiniteElement_SpaceInfo_t & space_info = ft->dg_spaces[space_index];
		return this->internal_data_pt(space_info.internal_offset_new+fieldindex);
	}

	unsigned BulkElementBase::get_DG_buffer_index(const unsigned &space_index,const unsigned &fieldindex)
	{
		auto * ft=this->get_jit_code()->get_func_table();
		const JITFuncSpec_Table_FiniteElement_SpaceInfo_t & space_info = ft->dg_spaces[space_index];
		return space_info.buffer_offset_basebulk+fieldindex;
	}

	unsigned BulkElementBase::get_DL_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_jit_code()->get_func_table();
        return ft->info_DL.buffer_offset_basebulk+ fieldindex;
    }

	unsigned BulkElementBase::get_D0_buffer_index(const unsigned &fieldindex)
    {
		auto * ft=this->get_jit_code()->get_func_table();
        return ft->info_D0.buffer_offset_basebulk+ fieldindex;
    }


	int BulkElementBase::get_DG_local_equation(const unsigned &space_index,const unsigned &fieldindex,const unsigned & nodeindex)
	{
		
		auto * ft=this->get_jit_code()->get_func_table();
		const JITFuncSpec_Table_FiniteElement_SpaceInfo_t & space_info = ft->dg_spaces[space_index];
		return this->internal_local_eqn(space_info.internal_offset_new+fieldindex,nodeindex);
	}

    int BulkElementBase::get_DL_local_equation(const unsigned &fieldindex,const unsigned & nodeindex)
	{
	  auto * ft=this->get_jit_code()->get_func_table();
	  return this->internal_local_eqn(ft->info_DL.internal_offset_new+fieldindex,nodeindex);						
	}
    int BulkElementBase::get_D0_local_equation(const unsigned &fieldindex)
	{
	  auto * ft=this->get_jit_code()->get_func_table();
	  return this->internal_local_eqn(ft->info_D0.internal_offset_new+fieldindex,0);								
	}

	// Maps a local coordinate s of this element into the reference domain of the macro element it is
	// attached to, or returns an empty vector if there is no macro element.
	//
	// The Q family keeps oomph's own bookkeeping: QElementBase tracks the son's region as an
	// axis-aligned box (s_macro_ll/s_macro_ur) that oomph's build() maintains, and the affine map into
	// it is exactly what Macro_element_vertex_s would reproduce. Everything else uses the general
	// vertex-coordinate form, which is the only one that survives a simplex son (not a sub-box of its
	// father) or a mixed forest (a son whose shape differs from its father's).
	std::vector<double> BulkElementBase::get_macro_element_coordinate_at_s(oomph::Vector<double> s)
	{
		if (!macro_elem_pt()) return {};
		unsigned el_dim = dim();
		oomph::QElementBase *qelem = dynamic_cast<oomph::QElementBase *>(this);
		if (qelem)
		{
			std::vector<double> s_macro(el_dim,0);
			for (unsigned i = 0; i < el_dim; i++)
			{
					s_macro[i] = qelem->s_macro_ll(i) + 0.5 * (s[i] + 1.0) * (qelem->s_macro_ur(i) - qelem->s_macro_ll(i));
			}
			return s_macro;
		}
		return this->macro_coordinate_from_local(s);
	}

	// Interpolate the son's vertex positions in the macro reference domain with this element's own C1
	// shape functions. On a root element Macro_element_vertex_s is empty and the map is the identity.
	std::vector<double> BulkElementBase::macro_coordinate_from_local(const oomph::Vector<double> &s) const
	{
		if (Macro_element_vertex_s.empty()) return std::vector<double>(s.begin(), s.end());
		const unsigned nvert = this->nvertex_node();
		oomph::Shape psi(nvert);
		this->shape_at_s_C1(s, psi);
		const unsigned mdim = Macro_element_vertex_s[0].size();
		std::vector<double> s_macro(mdim, 0.0);
		for (unsigned int v = 0; v < nvert && v < Macro_element_vertex_s.size(); v++)
		{
			for (unsigned int i = 0; i < mdim; i++) s_macro[i] += psi[v] * Macro_element_vertex_s[v][i];
		}
		return s_macro;
	}

	// Pass the macro element from father to son, converting the son's vertex coordinates (given in the
	// father's local frame) into the macro element's reference domain via the father's own map. Because
	// the conversion goes through the father's C1 shape functions rather than through any box, it does
	// not care whether father and son have the same shape.
	void BulkElementBase::inherit_macro_element_from_father(BulkElementBase *father_pt, const std::vector<std::vector<double>> &son_vertices_in_father)
	{
		if (!father_pt || !father_pt->macro_elem_pt()) return;
		this->set_macro_elem_pt(father_pt->macro_elem_pt());
		Macro_element_vertex_s.clear();
		Macro_element_vertex_s.reserve(son_vertices_in_father.size());
		for (auto &sv : son_vertices_in_father)
		{
			oomph::Vector<double> sf(sv.size());
			for (unsigned int i = 0; i < sv.size(); i++) sf[i] = sv[i];
			Macro_element_vertex_s.push_back(father_pt->macro_coordinate_from_local(sf));
		}
	}

	// Shared body of the simplex/mixed families' get_x_from_macro_element overrides.
	void BulkElementBase::get_x_from_generic_macro_element(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &x) const
	{
		std::vector<double> s_macro = this->macro_coordinate_from_local(s);
		oomph::Vector<double> s_macro_o(s_macro.size());
		for (unsigned int i = 0; i < s_macro.size(); i++) s_macro_o[i] = s_macro[i];
		const_cast<oomph::MacroElement *>(this->Macro_elem_pt)->macro_map(t, s_macro_o, x);
	}

	// A moving (ALE) mesh solves for the nodal positions, so snapping them onto the template geometry
	// would fight the solve. There the macro element keeps doing only what it does today: shape the
	// initial configuration, through Problem.map_nodes_on_macro_elements(). Note that this guard is on
	// moving_nodes, i.e. on whether the coordinates are dofs -- NOT on whether they are pinned; a
	// boundary that is held still gets the FE placement. The alternative -- the macro element driving
	// the Lagrangian coordinate instead of the position, via the undeformed macro element -- was
	// prototyped and works, but is deliberately not built; dev_docs/macro_elements.md 7.1 records what
	// it takes and 7.2 why it has to be an explicit per-boundary declaration rather than a default.
	bool BulkElementBase::macro_element_may_set_positions() const
	{
		if (!this->jitcode) return true;
		const JITFuncSpec_Table_FiniteElement_t *ft = this->jitcode->get_func_table();
		return !(ft && ft->moving_nodes);
	}

	// Position of the macro-element mapping at local coordinate s. Returns an empty vector if this
	// element has no macro element (or is not one of the shapes whose macro coordinate can be formed
	// yet -- see get_macro_element_coordinate_at_s).
	std::vector<double> BulkElementBase::get_macro_element_position_at_s(oomph::Vector<double> s)
	{
		std::vector<double> s_macro = this->get_macro_element_coordinate_at_s(s);
		if (s_macro.empty()) return {};
		oomph::Vector<double> s_macro_o(s_macro.size());
		for (unsigned int i = 0; i < s_macro.size(); i++) s_macro_o[i] = s_macro[i];
		oomph::Vector<double> r(this->nodal_dimension(), 0.0);
		macro_elem_pt()->macro_map(s_macro_o, r);
		return std::vector<double>(r.begin(), r.end());
	}

	// For elements built on a macro-element (structured, e.g. curved-boundary) mesh, snaps every
	// node's Eulerian position exactly onto the macro element's geometric mapping (evaluated at that
	// node's local coordinate), so the initial mesh exactly represents the macro-element geometry
	// (e.g. a curved domain boundary) rather than just the polynomial interpolation between corners.
	// Currently only implemented for Q-family (quad/brick) elements; a no-op (TODO) for T-family elements.
	void BulkElementBase::map_nodes_on_macro_element() // Does only work for bulk elems
	{
		if (!macro_elem_pt())
			return;
		const unsigned el_dim = dim();
		oomph::Vector<double> s(el_dim);
		oomph::Vector<double> r(this->nodal_dimension(), 0.0);
		for (unsigned int ni = 0; ni < this->nnode(); ni++)
		{
			// A hanging node's position is dictated by its masters; snapping it onto the geometry would
			// break agreement with the coarse neighbour that constrains it, and the boundary is anyway
			// only as accurate as that neighbour's edge. Leave it to the hanging-node machinery.
			if (this->node_pt(ni)->is_hanging()) continue;
			this->local_coordinate_of_node(ni, s);
			std::vector<double> s_macro = this->get_macro_element_coordinate_at_s(s);
			if (s_macro.empty()) continue;
			oomph::Vector<double> s_macro_o(s_macro.size());
			for (unsigned int i = 0; i < s_macro.size(); i++) s_macro_o[i] = s_macro[i];
			macro_elem_pt()->macro_map(s_macro_o, r);
			// Every history level, not just the present one. The macro geometry does not depend on time,
			// so one evaluation serves them all -- but leaving t>=1 behind on the straight interpolant is
			// not merely untidy. It makes the node's stored history disagree with what get_x(t,...)
			// returns, which (a) presents a mesh at rest as though it had been moving between t=1 and
			// t=0, and (b) makes oomph's synchronise_nonhanging_nodes -- which compares exactly those two
			// at every t while distributing -- classify conforming nodes as needing repair, corrupting
			// the geometry of a distributed curved mesh. Serial runs never compare them, which is why
			// the original "TODO: Time loop" here went unnoticed.
			oomph::Node *nod_pt = this->node_pt(ni);
			const unsigned nt = nod_pt->ntstorage();
			for (unsigned int id = 0; id < r.size() && id < nod_pt->ndim(); id++)
				for (unsigned int t = 0; t < nt; t++)
					nod_pt->x(t, id) = r[id];
		}
	}

	// Undo the FE overwrite that oomph's solid build performs. RefineableSolidQElement<2>/<3>::build
	// calls the plain RefineableQElement build (which does place new nodes through the macro map) and
	// then unconditionally resets every node to the FE interpolation -- deliberately, since in a solid
	// problem the position is a dof; its own comment says "If you wish to reposition nodes on
	// curvilinear boundaries of a domain to their exact positions on those boundaries you'll have to do
	// this yourself". This is that. It cannot be done from further_build(), which runs inside the build
	// being corrected; the concrete Q classes call it after theirs returns.
	//
	// Safe with respect to hanging nodes even though they are not yet marked as such here: oomph sets
	// up hanging nodes after every element has been built (refineable_mesh.cc, build at :1028 vs
	// setup_hanging_nodes at :1259), and that pass overwrites a hanging node's position with the coarse
	// neighbour's interpolation, which is what it must be.
	void BulkElementBase::reapply_macro_element_positions()
	{
		if (!macro_elem_pt() || !this->macro_element_may_set_positions()) return;
		this->map_nodes_on_macro_element();
	}

	// Factory that instantiates the concrete BulkElement* subclass matching a MeshTemplateElement's
	// geometric type index (the same meshio-style codes as get_meshio_type_index()) and the current
	// JIT code's "dominant space" (the highest interpolation order actually used by any
	// field, C1/C1TB/C2/C2TB/...). Where the dominant space is lower order than the template
	// element's own geometry (e.g. a template C2 triangle but the code only ever uses C1 fields), a
	// lower-order element is created instead together with "nodemap": the subset (and reordering) of
	// the template element's node indices that should be used to populate the new, lower-order
	// element's nodes.
	BulkElementBase *BulkElementBase::create_from_template(MeshTemplate *mt, MeshTemplateElement *el)
	{
		BulkElementBase *res = NULL;
		std::vector<int> nodemap;
		std::string domspace=std::string(BulkElementBase::__CurrentJITCode->get_func_table()->dominant_space);
		if (el->get_geometric_type_index() == 1)
		{
			res = new BulkElementLine1dC1();
		}
		else if (el->get_geometric_type_index() == 2)
		{
			if ( domspace == "C1" || domspace=="C1TB")
			{
				nodemap = {0, 2};
				res = new BulkElementLine1dC1();
			}
			else
			{
			  res = new BulkElementLine1dC2();
			}
		}
		else if (el->get_geometric_type_index() == 3)
		{
		  if (dynamic_cast<MeshTemplateElementTriC1TB*>(el))
		  {
			res = new BulkElementTri2dC1TB();
		  }
		  else
		  {
 		   res = new BulkElementTri2dC1();
		  }
		}
		else if (el->get_geometric_type_index() == 4)
		{
			if (dynamic_cast<MeshTemplateElementTetraC1TB*>(el))
			{
				res = new BulkElementTetra3dC1TB();
			}
			else
			{
				res = new BulkElementTetra3dC1();
			}
		}
		else if (el->get_geometric_type_index() == 44)
			res = new BulkElementTetra3dC1TB();
		else if (el->get_geometric_type_index() == 6)
			res = new BulkElementQuad2dC1();
		else if (el->get_geometric_type_index() == 8)
		{			
			if ( domspace == "C1" || domspace=="C1TB")
			{
				nodemap = {0, 2, 6, 8};
				res = new BulkElementQuad2dC1();
			}
			else
			{
				res = new BulkElementQuad2dC2();
			}
		}
		else if (el->get_geometric_type_index() == 9)
		{
			if (domspace == "C1")
			{
				nodemap = {0, 1, 2};
				res = new BulkElementTri2dC1();
			}
			else if (domspace == "C1TB")
			{
				nodemap = {0, 1, 2,6};
				res = new BulkElementTri2dC1TB();
			}			
			else if (domspace == "C2" || domspace == "")
			{
				res = new BulkElementTri2dC2();
			}
			else
			{
				res = new BulkElementTri2dC2TB();
			}
		}
		else if (el->get_geometric_type_index() == 10) // Tetra C2
		{
			if (domspace == "C1")
			{
				nodemap = {0, 1, 2, 3};
				res = new BulkElementTetra3dC1();
			}
			else if (domspace=="C1TB")
			{
			 if (!dynamic_cast<MeshTemplateElementTetraC2TB*>(el))
			 {
			  throw_runtime_error("Strange: Tetra C1TB element should be created, but the template element is not a C2TB one, which is required for the bubble node in the center");
			 }
			 nodemap = {0, 1, 2, 3,14};
			 res = new BulkElementTetra3dC1TB();
			}
			else if (domspace == "C2")
			{
				res = new BulkElementTetra3dC2();
			}
			else
			{
				res = new BulkElementTetra3dC2TB();
			}
		}
		else if (el->get_geometric_type_index() == 11)
		{
			res = new BulkElementBrick3dC1();
		}
		else if (el->get_geometric_type_index() == 14)
		{
			if (domspace == "C1" || domspace=="C1TB")
			{
				throw_runtime_error("TODO: Restrict nodes");
			}
			else
			{
				res = new BulkElementBrick3dC2();
			}
		}
		else if (el->get_geometric_type_index() == 0)
		{
			res= new PointElement0d();
		}
		else if (el->get_geometric_type_index() == 13)
		{
			if (domspace!="C1") throw_runtime_error("Found a wedge/prism element, which cannot be generalized to the space "+domspace);
			res= new BulkElementWedge3dC1();
		}
		else if (el->get_geometric_type_index() == 26)
		{
			if (domspace == "C1")
			{
			  nodemap = {0, 1, 2, 12, 13, 14};
			  res= new BulkElementWedge3dC1();
			}
			else if (domspace=="C2")
			{
              res= new BulkElementWedge3dC2();
			}
			else
			{
				throw_runtime_error("Found a wedge/prism element, which cannot be generalized to the space "+domspace);				
			}
		}
		else if (el->get_geometric_type_index() == 15)
		{
			if (domspace == "C1")
			{			  
			  res= new BulkElementPyramid3dC1();
			}			
			else
			{
				throw_runtime_error("Pyramids are not implemented yet for space "+domspace);				
			}
		}
		else if (el->get_geometric_type_index() == 27)
		{
			if (domspace == "C1")
			{
			  nodemap = {0, 1, 2, 3, 4};
			  res= new BulkElementPyramid3dC1();
			}
			else if (domspace=="C2")
			{
              res= new BulkElementPyramid3dC2();
			}
			else
			{
				throw_runtime_error("Found a pyramid element, which cannot be generalized to the space "+domspace);				
			}
		}
		else
		{
			throw_runtime_error("Undefined element type: " + std::to_string(el->get_geometric_type_index()));
		}
		if (el->get_node_indices().size() < res->nnode())
			throw_runtime_error("Too few nodes in the template element: " + std::to_string(el->get_node_indices().size()) + " vs. " + std::to_string(res->nnode()) + " element type: " + std::to_string(el->get_geometric_type_index()) + " , space: " + domspace);
		if (nodemap.empty())
		{
			for (unsigned int i = 0; i < res->nnode(); i++)
			{
				res->node_pt(i) = mt->get_nodes()[el->get_node_indices()[i]]->oomph_node;
				if (!mt->get_nodes()[el->get_node_indices()[i]]->oomph_node)
				{
					throw_runtime_error("Missing a NODE!");
				}
			}
		}
		else
		{
			for (unsigned int i = 0; i < res->nnode(); i++)
			{
				res->node_pt(i) = mt->get_nodes()[el->get_node_indices()[nodemap[i]]]->oomph_node;
			}
		}

		for (unsigned int i = 0; i < res->ninternal_data(); i++)
			res->internal_data_pt(i)->set_time_stepper(res->node_pt(0)->time_stepper_pt(), false);

		
		res->initial_cartesian_nondim_size = res->size();
		res->initial_quality_factor = res->get_quality_factor();

		// The element's own code, not the ambient __CurrentJITCode: by this point `res` exists and
		// carries it, so there is no reason to go back through the construction side channel.
		if (res->get_jit_code()->get_func_table()->integration_order)
		{
			res->set_integration_order(res->get_jit_code()->get_func_table()->integration_order);
		}
		return res;
	}


	// Allocates (do_alloc=true) or frees (do_alloc=false) every array member of a single
	// JITShapeInfo_t buffer used to hold shape-function values/derivatives for one element during
	// residual assembly. Unless FIXED_SIZE_SHAPE_BUFFER is defined, every array is sized generously
	// to fixed upper bounds (MAX_NODES/MAX_NODAL_DIM/MAX_TIME_WEIGHTS/MAX_HANG/MAX_FIELDS) covering
	// the largest supported element type, so the same buffer can be reused across all element types
	// without reallocation. The optional parts selected by "caps" (analytic Hessians on a moving
	// (ALE) mesh, and the second-spatial-derivative family) are left out - and their pointers NULLed
	// - unless some code actually asks for them.
	void alloc_dealloc_single_shape_buffer(bool do_alloc, JITShapeInfo_t * PYOOMPH_RESTRICT *buff, const ShapeBufferCaps &caps)
	{
		const bool with_analytical_hessian_moving_mesh = caps.hessian_moving_mesh;
		if (!(*buff))
		{
			if (do_alloc)
			{
				(*buff) = new JITShapeInfo_t;
			}
			else
			{
				return;
			}
		}

#ifndef FIXED_SIZE_SHAPE_BUFFER

		const int MAX_NODES = 27; // Should be max 27 for 3^3 (Brick C2)
		const int MAX_NODAL_DIM = 3;
		const int MAX_TIME_WEIGHTS = 7;
		const int MAX_HANG = 16; // Should be max 3
		const int MAX_FIELDS = 32;

		my_alloc_or_free(do_alloc, (*buff)->t, MAX_TIME_WEIGHTS);
		my_alloc_or_free(do_alloc, (*buff)->dt, MAX_TIME_WEIGHTS);
		my_alloc_or_free(do_alloc, (*buff)->int_pt_weights_d_coords, MAX_NODAL_DIM, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->elemsize_d_coords, MAX_NODAL_DIM, MAX_NODES);
		my_alloc_or_free(do_alloc, (*buff)->elemsize_Cart_d_coords, MAX_NODAL_DIM, MAX_NODES);
		for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
		{
			my_alloc_or_free(do_alloc, (*buff)->d_dx_shape_dcoord[i], MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);			
		}
				
		my_alloc_or_free(do_alloc, (*buff)->d_dx_shape_dcoord_DL, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);

		for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
		{
			my_alloc_or_free(do_alloc, (*buff)->shapes[i], MAX_NODES);
			for (unsigned k = 0; k < 3; k++)
				my_alloc_or_free(do_alloc, (*buff)->dx_shapes[k][i], MAX_NODES, MAX_NODAL_DIM);
			my_alloc_or_free(do_alloc, (*buff)->dX_shapes[i], MAX_NODES, MAX_NODAL_DIM);
			my_alloc_or_free(do_alloc, (*buff)->dS_shapes[i], MAX_NODES, MAX_NODAL_DIM);
		}

		my_alloc_or_free(do_alloc, (*buff)->shape_DL, MAX_NODES);
		for (unsigned k = 0; k < 3; k++)
			my_alloc_or_free(do_alloc, (*buff)->dx_shape_DL[k], MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dX_shape_DL, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->dS_shape_DL, MAX_NODES, MAX_NODAL_DIM);

		for (unsigned k = 0; k < 3; k++)
			my_alloc_or_free(do_alloc, (*buff)->normal[k], MAX_NODAL_DIM);
		for (unsigned k = 0; k < 3; k++)
			my_alloc_or_free(do_alloc, (*buff)->dnormal_dx[k], MAX_NODAL_DIM, MAX_NODAL_DIM);

		my_alloc_or_free(do_alloc, (*buff)->timestepper_weights_dt_BDF1, MAX_TIME_WEIGHTS);
		my_alloc_or_free(do_alloc, (*buff)->timestepper_weights_dt_BDF2, MAX_TIME_WEIGHTS);
		my_alloc_or_free(do_alloc, (*buff)->timestepper_weights_dt_Newmark2, MAX_TIME_WEIGHTS);
		my_alloc_or_free(do_alloc, (*buff)->timestepper_weights_d2t_Newmark2, MAX_TIME_WEIGHTS);
#else
		if (do_alloc)
			__shape_buffer_mem_usage += sizeof(JITShapeInfo_t);
#endif

		my_alloc_or_free(do_alloc, (*buff)->d_normal_dcoord, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
		my_alloc_or_free(do_alloc, (*buff)->d_dnormal_dx_dcoord, MAX_NODAL_DIM, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);

		if (with_analytical_hessian_moving_mesh || !do_alloc)
		{
			my_alloc_or_free(do_alloc, (*buff)->int_pt_weights_d2_coords, MAX_NODAL_DIM, MAX_NODAL_DIM, MAX_NODES, MAX_NODES);
			my_alloc_or_free(do_alloc, (*buff)->elemsize_d2_coords, MAX_NODAL_DIM, MAX_NODAL_DIM, MAX_NODES, MAX_NODES);
			my_alloc_or_free(do_alloc, (*buff)->elemsize_Cart_d2_coords, MAX_NODAL_DIM, MAX_NODAL_DIM, MAX_NODES, MAX_NODES);	
			for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
			{
				my_alloc_or_free(do_alloc, (*buff)->d2_dx2_shape_dcoord[i], MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
			}			
			my_alloc_or_free(do_alloc, (*buff)->d2_dx2_shape_dcoord_DL, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);

			my_alloc_or_free(do_alloc, (*buff)->d2_normal_d2coord, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
			my_alloc_or_free(do_alloc, (*buff)->d2_dnormal_dx_d2coord, MAX_NODAL_DIM, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM, MAX_NODES, MAX_NODAL_DIM);
		}
		else
		{
			(*buff)->int_pt_weights_d2_coords = NULL;
			(*buff)->elemsize_d2_coords=NULL;
			(*buff)->elemsize_Cart_d2_coords=NULL;
			for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
			{
				(*buff)->d2_dx2_shape_dcoord[i] = NULL;
			}
			(*buff)->d2_dx2_shape_dcoord_DL = NULL;
			(*buff)->d2_normal_d2coord = NULL;
			(*buff)->d2_dnormal_dx_d2coord = NULL;
		}

		// Second spatial derivatives. Sized from the nodal dimension, not from MAX_NODES/MAX_NODAL_DIM
		// - see ShapeBufferCaps for why that matters for d2_d2x2_shape_dcoord in particular.
		{
			const int D2_NODES = (int)caps.second_deriv_nodes();
			const int D2_DIM = (int)caps.second_deriv_nodal_dim;

			if (caps.second_derivs || !do_alloc)
			{
				for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
				{
					for (unsigned k = 0; k < 3; k++)
						my_alloc_or_free(do_alloc, (*buff)->d2x_shapes[k][i], D2_NODES, MAX_N2DERIV);
					my_alloc_or_free(do_alloc, (*buff)->d2S_shapes[i], D2_NODES, MAX_N2DERIV);
				}
				for (unsigned k = 0; k < 3; k++)
					my_alloc_or_free(do_alloc, (*buff)->d2x_shape_DL[k], D2_NODES, MAX_N2DERIV);
				my_alloc_or_free(do_alloc, (*buff)->d2S_shape_DL, D2_NODES, MAX_N2DERIV);
			}
			else
			{
				for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
				{
					for (unsigned k = 0; k < 3; k++) (*buff)->d2x_shapes[k][i] = NULL;
					(*buff)->d2S_shapes[i] = NULL;
				}
				for (unsigned k = 0; k < 3; k++) (*buff)->d2x_shape_DL[k] = NULL;
				(*buff)->d2S_shape_DL = NULL;
			}

			if (caps.second_derivs_dcoord || !do_alloc)
			{
				for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
					my_alloc_or_free(do_alloc, (*buff)->d_d2x_shape_dcoord[i], D2_NODES, MAX_N2DERIV, D2_NODES, D2_DIM);
				my_alloc_or_free(do_alloc, (*buff)->d_d2x_shape_dcoord_DL, D2_NODES, MAX_N2DERIV, D2_NODES, D2_DIM);
			}
			else
			{
				for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++) (*buff)->d_d2x_shape_dcoord[i] = NULL;
				(*buff)->d_d2x_shape_dcoord_DL = NULL;
			}

			if (caps.second_derivs_hessian || !do_alloc)
			{
				for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++)
					my_alloc_or_free(do_alloc, (*buff)->d2_d2x2_shape_dcoord[i], D2_NODES, MAX_N2DERIV, D2_NODES, D2_DIM, D2_NODES, D2_DIM);
				my_alloc_or_free(do_alloc, (*buff)->d2_d2x2_shape_dcoord_DL, D2_NODES, MAX_N2DERIV, D2_NODES, D2_DIM, D2_NODES, D2_DIM);
			}
			else
			{
				for (unsigned int i = 0; i < NUM_CONTINUOUS_SPACES; i++) (*buff)->d2_d2x2_shape_dcoord[i] = NULL;
				(*buff)->d2_d2x2_shape_dcoord_DL = NULL;
			}
		}

#ifndef FIXED_SIZE_SHAPE_BUFFER
		if (do_alloc)
		{
			// hanginfo_Pos is indexed [nodal coordinate direction][local node]; each field's
			// per-master local_eqn is now a plain scalar, so no extra allocation is needed for it.
			my_alloc((*buff)->hanginfo_Pos, MAX_NODAL_DIM);
			for (unsigned int d = 0; d < MAX_NODAL_DIM; d++)
			{
				my_alloc((*buff)->hanginfo_Pos[d], MAX_NODES);
				for (unsigned int l = 0; l < MAX_NODES; l++)
				{
					my_alloc((*buff)->hanginfo_Pos[d][l].masters, MAX_HANG);
				}
			}

			// hanginfo is the unified per-field buffer for the nodal_data buffer, indexed [global
			// field index][local node]; this covers continuous, DG, DL and D0 fields alike.
			my_alloc((*buff)->hanginfo, MAX_FIELDS);
			for (unsigned int f = 0; f < MAX_FIELDS; f++)
			{
				my_alloc((*buff)->hanginfo[f], MAX_NODES);
				for (unsigned int l = 0; l < MAX_NODES; l++)
				{
					my_alloc((*buff)->hanginfo[f][l].masters, MAX_HANG);
				}
			}
		}
		else
		{
			for (unsigned int d = 0; d < MAX_NODAL_DIM; d++)
			{
				for (unsigned int l = 0; l < MAX_NODES; l++)
				{
					my_free((*buff)->hanginfo_Pos[d][l].masters, MAX_HANG);
				}
				my_free((*buff)->hanginfo_Pos[d], MAX_NODES);
			}
			my_free((*buff)->hanginfo_Pos, MAX_NODAL_DIM);

			for (unsigned int f = 0; f < MAX_FIELDS; f++)
			{
				for (unsigned int l = 0; l < MAX_NODES; l++)
				{
					my_free((*buff)->hanginfo[f][l].masters, MAX_HANG);
				}
				my_free((*buff)->hanginfo[f], MAX_NODES);
			}
			my_free((*buff)->hanginfo, MAX_FIELDS);
		}
#endif

		if (do_alloc)
		{
			(*buff)->bulk_shapeinfo = NULL;
			(*buff)->opposite_shapeinfo = NULL;
			// `new JITShapeInfo_t` leaves this indeterminate, and the ordinary residual/Jacobian path
			// only ever reads it - the multi-assemble path is the sole writer, and restores false.
			(*buff)->during_shared_multi_assembling = false;
		}
		else
		{
			if ((*buff)->bulk_shapeinfo)
			{
				alloc_dealloc_single_shape_buffer(false, &((*buff)->bulk_shapeinfo), caps);
				delete (*buff)->bulk_shapeinfo;
			}
			if ((*buff)->opposite_shapeinfo)
			{
				alloc_dealloc_single_shape_buffer(false, &((*buff)->opposite_shapeinfo), caps);
				delete (*buff)->opposite_shapeinfo;
			}
			// delete *buff; //XXX: The main default shape buffer is not deallocated by default! Otherwise, reallocation does not work since DefaultShapeBuffer will be different than BulkElementBase::shape_buffer
		}
	}

	// Allocates (or frees) a full chain of shape buffers: the buffer itself, plus its
	// bulk_shapeinfo and opposite_shapeinfo (used by FaceElements to access the bulk/opposite
	// element's shape info), and one level further (bulk-of-opposite, bulk-of-bulk) since those
	// are also dereferenced when assembling flux/interface contributions.
	void alloc_dealloc_all_shape_buffers(bool do_alloc, JITShapeInfo_t **buff, const ShapeBufferCaps &caps)
	{
		if (do_alloc)
		{
			__shape_buffer_mem_usage = 0;
			alloc_dealloc_single_shape_buffer(true, buff, caps);
			alloc_dealloc_single_shape_buffer(true, &((*buff)->bulk_shapeinfo), caps);
			alloc_dealloc_single_shape_buffer(true, &((*buff)->opposite_shapeinfo), caps);
			alloc_dealloc_single_shape_buffer(true, &((*buff)->opposite_shapeinfo->bulk_shapeinfo), caps);
			alloc_dealloc_single_shape_buffer(true, &((*buff)->bulk_shapeinfo->bulk_shapeinfo), caps);
		//	std::cout << "Allocated " << __shape_buffer_mem_usage / (1024.0 * 1024.0) << " MB for the shape buffer" << std::endl;
		}
		else
		{
			// Deallocation of bulk_shapeinfo and opposite_shapeinfo is done in alloc_dealloc_single_shape_buffer
			alloc_dealloc_single_shape_buffer(false, buff, caps);
		}
	}

	// Constructs the element and widens the shape-buffer pool to whatever this code needs. The
	// element does not own a buffer; it borrows one from the pool for the duration of an assembly
	// (see ShapeBufferPool and get_shape_info below).
	BulkElementBase::BulkElementBase()
	{
		memset(&eleminfo, 0, sizeof(eleminfo));

		jitcode = BulkElementBase::__CurrentJITCode;
		if (!jitcode)
		{
			throw_runtime_error("Element generated without jit code");
		}

		const JITFuncSpec_Table_FiniteElement_t *ft = this->jitcode->get_func_table();
		// Set once here (eleminfo was memset just above and is never zeroed again), so that generated
		// code can reach this code's table without a file-scope global inside the .so.
		eleminfo.functable = const_cast<JITFuncSpec_Table_FiniteElement_t *>(ft);

		ShapeBufferCaps needed;
		needed.hessian_moving_mesh = ft->hessian_generated && ft->moving_nodes;
		if (required_shapes_want_d2x(&(ft->merged_required_shapes)))
		{
			needed.second_derivs = true;
			needed.second_deriv_nodal_dim = ft->nodal_dim;
			// Same gate as require_dxdshape in fill_shape_info_at_s: with an FD position Jacobian the
			// shape sensitivities are never asked for.
			needed.second_derivs_dcoord = ft->moving_nodes && !ft->fd_position_jacobian;
			needed.second_derivs_hessian = needed.second_derivs_dcoord && ft->hessian_generated;
			// The Hessian branch of fill_shape_info_at_s writes d2_d2x2_shape_dcoord whenever both are
			// on, so the first-order sensitivities have to be there too.
			if (needed.second_derivs_hessian) needed.second_derivs_dcoord = true;
		}

		shape_buffer_pool.require(needed);

		this->set_nlagrangian_and_ndim(this->jitcode->get_func_table()->lagr_dim, this->jitcode->get_func_table()->nodal_dim);
		this->ensure_external_data();
		
	}

	// The shape buffer this thread assembles into. Acquired lazily so that no call path can reach the
	// buffer without one being held - the alternative, an RAII scope at every entry point, is one
	// missed entry point away from a null dereference at runtime.
	JITShapeInfo_t *BulkElementBase::get_shape_info() const
	{
		if (!__thread_shape_buffer)
			__thread_shape_buffer = shape_buffer_pool.acquire();
		return __thread_shape_buffer;
	}

	// Hands this thread's buffer back to the pool. Nothing calls it yet: with a serial element loop
	// the main thread simply keeps its one buffer for the life of the process, exactly as the old
	// single global buffer did. A parallel element loop calls it at the end of each worker so the
	// pool stays bounded by the number of threads actually in flight.
	void release_thread_shape_buffer()
	{
		shape_buffer_pool.release(__thread_shape_buffer);
		__thread_shape_buffer = NULL;
	}

	// The owned slots of eleminfo.nodal_coords[i], i.e. the ones this element new'ed and must delete:
	// the local-coordinate buffers, and for a face element the zeta buffers after them. The range is
	// RECORDED by fill_element_info(), which is the only place that knows it, rather than read back
	// out of jitcode->get_func_table() here: a destructor may not depend on another object's lifetime
	// -- DynamicJITCode is a nanobind-owned Python object and at interpreter shutdown it is released
	// before the mesh whose elements point at it, which segfaulted for every interface carrying an
	// interior-facet skeleton (that mesh is destroyed from inside its code generator's dealloc).
	//
	// It used to be DERIVED here instead, as nodal_dim + lagr_dim + dim (+ dim again for a face
	// element's zeta). That assumed every face element carries a zeta block, and one does not: a 1D
	// edge nested in a 2D face of a 3D bulk gets no boundary coordinate, so the generated Pos layout
	// has 7 fields where the derivation predicted 8 (tests/test_segment_ordering.py, the line-in-3D
	// case). fill_element_info()'s consistency check caught it before the wrong slots were freed,
	// which is what that check was for; recording the range removes the assumption altogether.
	//
	// Recording also settles a second hazard the derivation had: as_interface_element() is virtual, so
	// the old form was only correct while the object still *was* one. By the time ~BulkElementBase
	// runs the InterfaceElementBase sub-object is gone and it reports NULL - which is why
	// ~InterfaceElementBase calls free_element_info() itself and the later call from ~BulkElementBase
	// finds eleminfo.alloced already false. A plain recorded pair needs no such ordering.
	std::pair<unsigned, unsigned> BulkElementBase::owned_nodal_coord_range() const
	{
		return std::make_pair(owned_coord_begin, owned_coord_end);
	}

	// Frees all buffers set up by fill_element_info() (nodal coordinate/data/local-eqn arrays).
	// Safe to call multiple times; a no-op unless fill_element_info() actually allocated anything.
	void BulkElementBase::free_element_info()
	{
		if (!eleminfo.alloced)
			return;
		const std::pair<unsigned, unsigned> owned = this->owned_nodal_coord_range();
		const unsigned owned_begin = owned.first, owned_end = owned.second;
		for (unsigned int i = 0; i < eleminfo.nnode; i++)
		{
			if (eleminfo.nodal_data[i])
			{
				free(eleminfo.nodal_data[i]);
				eleminfo.nodal_data[i] = NULL;
			}
			if (eleminfo.nodal_local_eqn[i])
			{
				free(eleminfo.nodal_local_eqn[i]);
				eleminfo.nodal_local_eqn[i] = NULL;
			}
			if (eleminfo.pos_local_eqn[i])
			{
				free(eleminfo.pos_local_eqn[i]);
				eleminfo.pos_local_eqn[i] = NULL;
			}
			if (eleminfo.nodal_coords[i])
			{
				// The Eulerian and Lagrangian slots point into the node, but everything from the local
				// coordinates onwards (local coordinate buffer, then the FaceElement zeta buffers) is
				// new'ed in fill_element_info and owned here. This loop used to start one block later,
				// at +this->dim(), i.e. it freed the zeta buffers and leaked the local coordinates --
				// dim*nnode doubles per element per fill. fill_element_info() runs on every element
				// after every mesh adaption, so an adaptive run leaked ~1 MB per adaption (measured on
				// a 1900-dof adaptive Poisson problem) and grew without bound; under mpirun every rank
				// pays it separately.
				for (unsigned int j = owned_begin; j < owned_end; j++)
				{

					if (eleminfo.nodal_coords[i][j]) delete eleminfo.nodal_coords[i][j];
				}
				free(eleminfo.nodal_coords[i]);
				eleminfo.nodal_coords[i] = NULL;
			}
		}
		//std::cout << "NODAL COORDS DEALLOCATED FOR " << this << std::endl;
		if (eleminfo.nodal_coords)
		{
			free(eleminfo.nodal_coords);
			eleminfo.nodal_coords = NULL;
		}
		if (eleminfo.nodal_data)
		{
			free(eleminfo.nodal_data);
			eleminfo.nodal_data = NULL;
		}
		if (eleminfo.nodal_local_eqn)
		{
			free(eleminfo.nodal_local_eqn);
			eleminfo.nodal_local_eqn = NULL;
		}
		if (eleminfo.pos_local_eqn)
		{
			free(eleminfo.pos_local_eqn);
			eleminfo.pos_local_eqn = NULL;
		}
		//  if (eleminfo.global_parameters) {free(eleminfo.global_parameters); eleminfo.global_parameters=NULL;}
		// if (eleminfo.nullified_residual_dof) {free(eleminfo.nullified_residual_dof); eleminfo.nullified_residual_dof=NULL;}
		eleminfo.alloced = false;
	}

	BulkElementBase::~BulkElementBase()
	{
		free_element_info();
	}

	// Extra multiplicative factor for the integration measure in curvilinear coordinate systems
	// (e.g. axisymmetric r-factor), evaluated by JIT-generated code if the code table provides it;
	// 1.0 (Cartesian, no extra factor) otherwise.
	double BulkElementBase::geometric_jacobian(const oomph::Vector<double> &x)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		if (functable->GeometricJacobian)
		{
			return functable->GeometricJacobian(&eleminfo, &(x[0]));
		}
		else
			return 1.0;
	}

	// Builds the eleminfo struct (JITElementInfo_t), the flat table of pointers to nodal
	// coordinates/data/local-equation-numbers that JIT-compiled residual/Jacobian code reads
	// directly, in a layout that mirrors the field ordering the code generator assumed
	// (continuous spaces, then DG spaces, then DL, D0, and finally external ED0 fields).
	// Called every time equation numbers or nodal data pointers may have changed (e.g. after
	// mesh adaption or equation numbering) since it caches raw pointers, not indices.
	void BulkElementBase::fill_element_info(bool without_equations)
	{
		free_element_info();
		// The local equation numbering is about to be rebuilt, so the local dof -> contribution map
		// keyed on it is stale.
		local_dof_contribution_indices_valid = false;
		// Same for the cached hang classification: fill_element_info() is reached from
		// assign_eqn_numbers(), i.e. after every adapt/remesh/pin/constraint change, which is exactly
		// when a node can start or stop hanging. Only invalidated here, never recomputed - the
		// interface elements' equation remap vectors are rebuilt AFTER this point, so a scan taken
		// here would see a half-built element.
		hang_state_valid = false;

		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();

		/*eleminfo.nodal_coords = (double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT )malloc(eleminfo.nnode * sizeof(double **));		
		eleminfo.nodal_data = (double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT )calloc(eleminfo.nnode, sizeof(double **));
		eleminfo.nodal_local_eqn = (int * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT )calloc(eleminfo.nnode, sizeof(int *));
		eleminfo.pos_local_eqn = (int * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT )calloc(eleminfo.nnode, sizeof(int *));*/
		eleminfo.nodal_coords = (double ***)malloc(eleminfo.nnode * sizeof(double **));		
		eleminfo.nodal_data = (double ***)calloc(eleminfo.nnode, sizeof(double **));
		eleminfo.nodal_local_eqn = (int **)calloc(eleminfo.nnode, sizeof(int *));
		eleminfo.pos_local_eqn = (int **)calloc(eleminfo.nnode, sizeof(int *));

		// Global numfields . That might waste some memory, but it it necessary for having all aligned (in particular for additional interface fields)
		// TODO: Maybe split at least the D0 + ED0 to another storage
		unsigned numfields = 0;

		for (unsigned int i_space=0;i_space<functable->num_present_continuous_spaces;i_space++)
		{
			numfields += functable->present_continuous_spaces[i_space]->numfields;
		}

		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			if (eleminfo.nnode_of_space[functable->present_dg_spaces[i_space]->space_index])
			{
				numfields += functable->present_dg_spaces[i_space]->numfields;
			}
		}
		if (eleminfo.nnode_DL)
			numfields += functable->info_DL.numfields;
		numfields += functable->info_D0.numfields + functable->info_ED0.numfields;
		// InterfaceElementBase is the only pyoomph element that is an oomph::FaceElement, so this is
		// exactly the "am I a face element" test the code below wants - and it is a virtual call rather
		// than a dynamic_cast down the virtual-inheritance diamond. Hoisted too: it does not depend on
		// the node, and asking per node made fill_element_info one of the costliest cast sites here.
		InterfaceElementBase *const this_as_face = this->as_interface_element();
		const unsigned zeta_offset = eleminfo.nodal_dim + functable->lagr_dim + this->dim();
		// Recorded here, where the function table is still guaranteed to be alive, for
		// free_element_info() to use later without touching it (see owned_nodal_coord_range).
		// The slots below zeta_offset that are NOT owned are the Eulerian and Lagrangian ones, which
		// point into the node; everything from the local-coordinate block to the end is new'ed below.
		owned_coord_begin = eleminfo.nodal_dim + functable->lagr_dim;
		owned_coord_end = functable->info_Pos.numfields;
		bool face_zeta_defined = false;
		if (this_as_face && functable->info_Pos.numfields > zeta_offset)
		{
			oomph::Mesh *themesh = this->get_jit_code()->get_bulk_mesh();
			while (pyoomph::InterfaceMesh *im = dynamic_cast<pyoomph::InterfaceMesh *>(themesh))
			{
				themesh = im->get_bulk_mesh();
			}
			pyoomph::Mesh *pmesh = dynamic_cast<pyoomph::Mesh *>(themesh);
			face_zeta_defined = pmesh && pmesh->is_boundary_coordinate_defined(this_as_face->boundary_number_in_bulk_mesh());
		}
		for (unsigned int i = 0; i < eleminfo.nnode; i++)
		{
			oomph::Vector<double> snode(this->dim(),0.0);
			if (this->dim()>0)
			{
				this->local_coordinate_of_node(i,snode);
			}
						
			// eleminfo.nnode counts BUFFER slots, which is not the same as the number of oomph nodes:
			// the 0d ODE element (BulkElementODE0d) has one dummy slot, so that its D0 fields have
			// somewhere to write below, and no node at all. node_pt(0) there reads one pointer past a
			// zero-length array - harmless in practice only because that element also has nodal_dim
			// and lagr_dim 0, so the pointer is never dereferenced, but it is still an out-of-bounds
			// read and valgrind reports it on every run. A node-less slot gets no node.
			pyoomph::Node *const nod_pt = (i < this->nnode() ? static_cast<pyoomph::Node *>(node_pt(i)) : NULL);
			eleminfo.nodal_coords[i] = (double **)calloc(functable->info_Pos.numfields, sizeof(double *));
			
			for (unsigned int j = 0; j < eleminfo.nodal_dim; j++)
				eleminfo.nodal_coords[i][j] = nod_pt->variable_position_pt()->value_pt(j);
			for (unsigned int j = 0; j < functable->lagr_dim; j++)
				eleminfo.nodal_coords[i][eleminfo.nodal_dim + j] = &(nod_pt->xi(j));
			for (unsigned int j = 0; j < this->dim(); j++)
				eleminfo.nodal_coords[i][eleminfo.nodal_dim + functable->lagr_dim +j] = new double(snode[j]); // Local coordinate buffer

			// FaceElements additionally expose boundary (zeta) coordinates per node, appended after
			// Eulerian/Lagrangian/local-coordinate slots; these are newly allocated doubles (owned
			// here, freed in free_element_info) since Node does not store them contiguously.
			if (this_as_face)
			{
				const bool have_zeta = face_zeta_defined && nod_pt->as_boundary_node() && nod_pt->boundary_coordinates_have_been_set_up();
				for (unsigned int j=zeta_offset;j<functable->info_Pos.numfields;j++)
				{
					double zeta=0.0;
					if (have_zeta)
					{
						zeta= this->zeta_nodal(i,0,j-zeta_offset);							
					}
					eleminfo.nodal_coords[i][j] =   new double(zeta); // zeta coordinate buffer
				}
			}

			eleminfo.nodal_data[i] = (double **)calloc(numfields, sizeof(double *));
			eleminfo.nodal_local_eqn[i] = (int *)calloc(numfields, sizeof(int));
			for (unsigned int j = 0; j < numfields; j++)
				eleminfo.nodal_local_eqn[i][j] = -1;
			eleminfo.pos_local_eqn[i] = (int *)calloc(eleminfo.nodal_dim, sizeof(int));
			for (unsigned int j = 0; j < eleminfo.nodal_dim; j++)
				eleminfo.pos_local_eqn[i][j] = -1;
		}
		
		
		if (!without_equations)
		{
			for (unsigned int i = 0; i < eleminfo.nnode; i++)
			{
				for (unsigned int j = 0; j < eleminfo.nodal_dim; j++)
				{
					if (static_cast<pyoomph::Node *>(this->node_pt(i))->is_hanging())
					{
						eleminfo.pos_local_eqn[i][j] = -2; //->constrain
					}
					else
					{
						eleminfo.pos_local_eqn[i][j] = this->position_local_eqn(i, 0, j);
					}
				}
			}
		}
				
		
		const std::vector<std::vector<unsigned>> & space_to_element_node_index =this->get_nodal_space_index_to_element_index_map();

		unsigned local_field_offset = 0;
		for (unsigned int i_space=0;i_space<functable->num_present_continuous_spaces;i_space++)
		{
			auto *space_info=functable->present_continuous_spaces[i_space];
			for (unsigned int i = 0; i < eleminfo.nnode_of_space[space_info->space_index]; i++)
			{
				unsigned i_el = space_to_element_node_index[space_info->space_index][i];
				for (unsigned int j = 0; j < functable->present_continuous_spaces[i_space]->numfields_basebulk; j++)
				{
					unsigned value_index = j + space_info->nodal_offset_basebulk;
					eleminfo.nodal_data[i][value_index] = node_pt(i_el)->value_pt(value_index); // Warning: value_pt does not work for hanging nodes! Will be changed if necessary
					if (!without_equations) eleminfo.nodal_local_eqn[i][value_index] = this->nodal_local_eqn(i_el, value_index);
				}
			}
			local_field_offset += functable->present_continuous_spaces[i_space]->numfields_basebulk;			
		}


		
		
		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto *space_info=functable->present_dg_spaces[i_space];
			for (unsigned int i = 0; i < eleminfo.nnode_of_space[space_info->space_index]; i++)
			{
				for (unsigned int j = 0; j < space_info->numfields_basebulk; j++)
				{
					unsigned node_index = j + local_field_offset; // TODO: Use a better way to get the global field index for DG fields
					eleminfo.nodal_data[i][node_index] =  this->get_DG_nodal_data(space_info->space_index, j)->value_pt(this->get_DG_node_index(space_info->space_index, j,i)); 
					if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] =  this->get_DG_local_equation(space_info->space_index,j, i);
				}
			}
			local_field_offset += space_info->numfields_basebulk;
		}

		

						
				
		// Elemental (non-continuous) fields				
		// For interface elements,  there is a gap here for indexing. Fill be filled later		
		local_field_offset=0;
		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			if (eleminfo.nnode_of_space[functable->present_continuous_spaces[si]->space_index])
			{
				local_field_offset += functable->present_continuous_spaces[si]->numfields;
			}
		}
		for (unsigned int si=0;si<functable->num_present_dg_spaces;si++)
		{
			if (eleminfo.nnode_of_space[functable->present_dg_spaces[si]->space_index])
			{
				local_field_offset += functable->present_dg_spaces[si]->numfields;
			}
		}
      

		for (unsigned int i = 0; i < eleminfo.nnode_DL; i++)
		{
			for (unsigned int j = 0; j < functable->info_DL.numfields; j++)
			{
				unsigned node_index = j + local_field_offset;
				eleminfo.nodal_data[i][node_index] = this->get_DL_nodal_data( j)->value_pt(i);
				if (!without_equations) eleminfo.nodal_local_eqn[i][node_index] = this->get_DL_local_equation(j, i);
			}
		}

		local_field_offset += functable->info_DL.numfields;		

		for (unsigned int j = 0; j < functable->info_D0.numfields; j++)
		{
			unsigned node_index = j + local_field_offset;
			eleminfo.nodal_data[0][node_index] = this->get_D0_nodal_data(j)->value_pt(0);
			if (!without_equations) eleminfo.nodal_local_eqn[0][node_index] = this->get_D0_local_equation(j);
		}

		local_field_offset = 0;
		for (unsigned int i_space=0;i_space<functable->num_present_continuous_spaces;i_space++)
		{
			auto *space_info=functable->present_continuous_spaces[i_space];
			local_field_offset += space_info->numfields;
		}
		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto *space_info=functable->present_dg_spaces[i_space];
			local_field_offset += space_info->numfields;
		}
		local_field_offset+=functable->info_DL.numfields+functable->info_D0.numfields;

		// Create the information for the external dofs
		for (unsigned int i = 0; i < functable->info_ED0.numfields; i++)
		{

			unsigned node_index = i + local_field_offset;

			if (!without_equations)
			{
				//		std::cout << "NODE INDEX oF " << functable->fieldnames_ED0[i] << " IS " << node_index << std::endl;
				if (!jitcode->linked_external_data[i].data)
					throw_runtime_error("Element has an external data contribution, which is not assigned: " + std::string(functable->info_ED0.fieldnames[i]));
				if (jitcode->linked_external_data[i].elemental_index < 0)
				{
					// Read by output/integral expressions only, so it was deliberately not registered as
					// external data (reindex_elemental_data). Those evaluators want the value and never an
					// equation; -1 is what every other unassembled dof carries, so a residual or Jacobian
					// term that referenced it after all would be skipped rather than write somewhere wrong.
					int value_i = jitcode->linked_external_data[i].value_index;
					eleminfo.nodal_data[0][node_index] = jitcode->linked_external_data[i].data->value_pt(value_i);
					eleminfo.nodal_local_eqn[0][node_index] = -1;
					continue;
				}
				int extdata_i = jitcode->linked_external_data[i].elemental_index+functable->info_ED0.external_offset_bulk;
				if (extdata_i >= (int)this->nexternal_data())
					throw_runtime_error("Somehow the external data array was not done well when trying to index data: " + std::string(functable->info_ED0.fieldnames[i]) + "  ext_data_index is " + std::to_string(extdata_i) + ", but only " + std::to_string((int)this->nexternal_data()) + " ext data slots present. Happened in " + jitcode->get_file_name());
				int value_i = jitcode->linked_external_data[i].value_index;
				if (value_i < 0 || value_i >= (int)this->external_data_pt(extdata_i)->nvalue())
					throw_runtime_error("Somehow the external data array was not done, i.e. wrong value index, well when trying to index data: " + std::string(functable->info_ED0.fieldnames[i]) + " at value " + std::to_string(value_i));
				eleminfo.nodal_data[0][node_index] = this->external_data_pt(extdata_i)->value_pt(value_i); // This is a bit an issue. You cannot access this data if you don't need equations to be linked 
				eleminfo.nodal_local_eqn[0][node_index] = this->external_local_eqn(extdata_i, value_i);


			}
		}

		//	eleminfo.global_parameters=(double**)calloc(functable->numglobal_params,sizeof(double*));

		eleminfo.ndof = this->ndof();
		eleminfo.alloced = true;

		// Checking the nullified dofs
		/*
		for (unsigned int l=0;l<this->nnode();l++)
		{
		  BoundaryNode *bn=dynamic_cast<BoundaryNode*>(this->node_pt(l));
		  if (bn)
		  {
		   if (bn->nullified_dofs.count(jitcode))
		   {
			 if (!eleminfo.nullified_residual_dof) eleminfo.nullified_residual_dof=(bool*)calloc(eleminfo.ndof,sizeof(bool));
			 for (int i : bn->nullified_dofs[jitcode])
			 {
			   if (i<0)
			   {
				i=-i-1;
				i=this->position_local_eqn(l,0,i);
				if (i>=0) eleminfo.nullified_residual_dof[i]=true;
			   }
			   else
			   {
				this->nodal_local_eqn(l,i);
			   }
			 }
		   }
		  }
		}
		*/
	}

	// Number of nodal values required at every node: same for all nodes of this element type
	// (the total number of "base bulk" field values, i.e. all continuous-space fields the code
	// generator laid out for this element, regardless of which node index n is asked about).
	unsigned BulkElementBase::required_nvalue(const unsigned &) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		return functable->total_num_fields_basebulk;
	}

	// See the declaration in elements.hpp: one dynamic_cast per element object, then cached. The
	// QuadElementBase and BrickElementBase are siblings under QElementBase (2d vs 3d), and
	// TElementBase/wedge/pyramid are unrelated to both, so the families are mutually exclusive and
	// the test order only decides how early the common cases stop.
	BulkElementBase::ElementFamily BulkElementBase::element_family() const
	{
		if (element_family_cache == EF_UNKNOWN)
		{
			if (dynamic_cast<const oomph::BrickElementBase *>(this))
				element_family_cache = EF_BRICK;
			else if (dynamic_cast<const oomph::QuadElementBase *>(this))
				element_family_cache = EF_QUAD;
			else if (dynamic_cast<const oomph::TElementBase *>(this))
				element_family_cache = EF_SIMPLEX;
			else if (dynamic_cast<const oomph::RefineableWedgeElement *>(this))
				element_family_cache = EF_WEDGE;
			else if (dynamic_cast<const oomph::RefineablePyramidElement *>(this))
				element_family_cache = EF_PYRAMID;
			else
				element_family_cache = EF_OTHER;
		}
		return element_family_cache;
	}

	// oomph-lib node-construction hooks (plain / with explicit timestepper, interior / boundary):
	// create a pyoomph::Node (or BoundaryNode) with the right Lagrangian/nodal dimensions and
	// enough values to hold every base-bulk field.
	oomph::Node *BulkElementBase::construct_node(const unsigned &n)
	{
		unsigned ntot = this->required_nvalue(n);
		//	 std::cout << "NLAGR " <<  this->nlagrangian() << "  " << this->nnodal_lagrangian_type() << std::endl;
		node_pt(n) = new Node(this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}

	oomph::Node *BulkElementBase::construct_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt)
	{
		unsigned ntot = required_nvalue(n);
		//		 		 std::cout << "NLAGR " <<  this->nlagrangian() << "  " << this->nnodal_lagrangian_type() << std::endl;
		node_pt(n) = new Node(time_stepper_pt, this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}

	oomph::Node *BulkElementBase::construct_boundary_node(const unsigned &n)
	{	
		unsigned ntot = required_nvalue(n);
		node_pt(n) = new BoundaryNode(this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}

	oomph::Node *BulkElementBase::construct_boundary_node(const unsigned &n, oomph::TimeStepper *const &time_stepper_pt)
	{		
		unsigned ntot = required_nvalue(n);
		node_pt(n) = new BoundaryNode(time_stepper_pt, this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), ntot);
		return node_pt(n);
	}


	// Number of continuously-interpolated nodal values per node (same quantity as
	// required_nvalue(), exposed under the name oomph-lib's projection/interpolation code uses).
	unsigned BulkElementBase::ncont_interpolated_values() const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		return functable->total_num_fields_basebulk;
	}

	oomph::Vector<double> BulkElementBase::get_midpoint_s() // Set s=[0.5*(smin+smax), ... ] (but modified e.g. for tris)
	{
		return oomph::Vector<double>(this->dim(), 0.5 * (this->s_min() + this->s_max()));
	}

	// Evaluates a "local expression" at the element's midpoint local coordinate.
	double BulkElementBase::eval_local_expression_at_midpoint(unsigned index)
	{
		oomph::Vector<double> s = this->get_midpoint_s();
		return eval_local_expression_at_s(index, s);
	}

	// Creates a brand-new (not connected to the mesh's node array) Node at local coordinate s,
	// with its Lagrangian/Eulerian coordinates and history values obtained by interpolating this
	// element's own fields there. Used e.g. to sample the solution at an arbitrary point without
	// requiring an actual mesh node to exist there.
    pyoomph::Node *BulkElementBase::create_interpolated_node(const oomph::Vector<double> &s,bool as_boundary_node)
    {
		if (this->nnode()==0) return 0;
		pyoomph::Node *res;
		if (as_boundary_node)
		{
		 	res= new pyoomph::BoundaryNode(this->node_pt(0)->time_stepper_pt(),this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), this->required_nvalue(0));
		}
		else
		{
			res= new pyoomph::Node(this->node_pt(0)->time_stepper_pt(),this->nlagrangian(), this->nnodal_lagrangian_type(), this->nodal_dimension(), this->nnodal_position_type(), this->required_nvalue(0));	
		}
		

		oomph::Vector<double> xibuff(this->lagrangian_dimension(),0.0);
		this->interpolated_xi(s,xibuff);	
		for (unsigned i = 0; i < this->lagrangian_dimension(); i++) res->xi(i) = xibuff[i];

		for (unsigned ti = 0; ti < res->time_stepper_pt()->ntstorage(); ti++)
		{
			oomph::Vector<double> xbuff(this->nodal_dimension(),0.0);
			this->interpolated_x(ti,s,xbuff);	
			for (unsigned i = 0; i < this->nodal_dimension(); i++)
				res->x(ti, i) = xbuff[i];

			oomph::Vector<double> vbuff(res->nvalue(),0.0);
			this->get_interpolated_values(ti,s,vbuff);
			for (unsigned int i=0;i<vbuff.size();i++) res->set_value(ti,i,vbuff[i]);
			
		}
        
		return res;
    }

    oomph::Vector<double> BulkElementBase::get_Eulerian_midpoint_from_local_coordinate() // Set s=[0.5*(smin+smax), ... ] and evaluate the position
	{
		oomph::Vector<double> res(this->nodal_dimension(), 0.0);
		if (this->nnode() == 1)
		{
			for (unsigned int i = 0; i < this->nodal_dimension(); i++)
				res[i] = this->node_pt(0)->x(i);
			return res;
		}
		oomph::Vector<double> s = this->get_midpoint_s();
		this->interpolated_x(s, res);
		return res;
	}

	oomph::Vector<double> BulkElementBase::get_Lagrangian_midpoint_from_local_coordinate() // Set s=[0.5*(smin+smax), ... ] and evaluate the position
	{
		oomph::Vector<double> res(this->nlagrangian(), 0.0);
		if (this->nnode() == 1)
		{
			for (unsigned int i = 0; i < this->nlagrangian(); i++)
				res[i] = static_cast<pyoomph::Node *>(this->node_pt(0))->xi(i);
			return res;
		}
		oomph::Vector<double> s = this->get_midpoint_s();
	  oomph::Shape psi(this->nnode());
	  this->shape(s,psi);
	  const unsigned n_lagrangian = static_cast<pyoomph::Node *>(this->node_pt(0))->nlagrangian();
	  for(unsigned i=0;i<n_lagrangian;i++)
		{
		 res[i] = 0.0;
		 for(unsigned l=0;l<this->nnode();l++) 
		  {
		     res[i] += lagrangian_position_gen(l,0,i)*psi(l);		   
		  }
		}
				
		return res;
	}

	// Paraview output hooks inherited from oomph-lib are not used by pyoomph (output goes through
	// its own VTU/plotting machinery instead); left unimplemented on purpose.
	std::string BulkElementBase::scalar_name_paraview(const unsigned &) const
	{
		throw_runtime_error("NOT IMPLEMENTED");
	}

	// Looks up the nodal value index of a base-bulk continuous field by its generated name.
	// Returns -1 if no such field exists (e.g. it lives in a non-nodal/discontinuous space).
	int BulkElementBase::get_nodal_index_by_name(oomph::Node *, std::string fieldname)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();

		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			for (unsigned i = 0; i < functable->present_continuous_spaces[si]->numfields_basebulk; i++)
			{
				if (std::string(functable->present_continuous_spaces[si]->fieldnames[i]) == fieldname) return i+functable->present_continuous_spaces[si]->nodal_offset_basebulk;
			}
		}		
		return -1;
	}

	unsigned BulkElementBase::nscalar_paraview() const
	{
		throw_runtime_error("NOT IMPLEMENTED");
	}

	void BulkElementBase::scalar_value_paraview(std::ofstream &, const unsigned &, const unsigned &) const
	{
		throw_runtime_error("NOT IMPLEMENTED");
	}



	// Default fallback: elements without bubble enrichment treat the "TB" (bubble-enriched)
	// spaces C1TB/C2TB as identical to the plain C1/C2 spaces; subclasses that actually have
	// bubble functions override these.
	void BulkElementBase::shape_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		//  if (this->has_bubble()) throw_runtime_error("Implement for bubble-enriched elements");
		this->shape_at_s_C1(s, psi);
	}

	void BulkElementBase::shape_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi) const
	{
		//  if (this->has_bubble()) throw_runtime_error("Implement for bubble-enriched elements");
		this->shape_at_s_C2(s, psi);
	}

	void BulkElementBase::dshape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		//  if (this->has_bubble()) throw_runtime_error("Implement for bubble-enriched elements");
		this->dshape_local_at_s_C2(s, psi, dpsi);
	}

	void BulkElementBase::dshape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi) const
	{
		//  if (this->has_bubble()) throw_runtime_error("Implement for bubble-enriched elements");
		this->dshape_local_at_s_C1(s, psi, dpsi);
	}

	// ------------------------------------------------------------------------------------------
	// Second local derivatives of the shape functions. See the declarations in elements.hpp for the
	// slot convention (PYOOMPH_D2_SLOT, full square, not oomph-lib's N2deriv packing).
	// ------------------------------------------------------------------------------------------

	void BulkElementBase::remap_oomph_d2shape_packing(unsigned el_dim, unsigned nnode, const oomph::DShape &d2psids_oomph, oomph::DShape &d2psi)
	{
		// oomph-lib packs the local second derivatives dimension-dependently (1D: {00}; 2D:
		// {00,11,01}; 3D: {00,11,22,01,02,12}) - see FiniteElement::N2deriv. Unpack into the square
		// layout, writing both orders of the mixed entries.
		static const unsigned mixed_a[3] = {0, 0, 1};
		static const unsigned mixed_b[3] = {1, 2, 2};
		for (unsigned l = 0; l < nnode; l++)
		{
			for (unsigned k = 0; k < MAX_N2DERIV; k++) d2psi(l, k) = 0.0;
			for (unsigned a = 0; a < el_dim; a++)
				d2psi(l, PYOOMPH_D2_SLOT(a, a)) = d2psids_oomph(l, a);
			// The mixed slots follow the pure ones and are ordered 01, 02, 12 - but only the
			// combinations that exist for this element dimension are present.
			unsigned nmixed = (el_dim < 2 ? 0 : (el_dim == 2 ? 1 : 3));
			for (unsigned m = 0; m < nmixed; m++)
			{
				const double v = d2psids_oomph(l, el_dim + m);
				d2psi(l, PYOOMPH_D2_SLOT(mixed_a[m], mixed_b[m])) = v;
				d2psi(l, PYOOMPH_D2_SLOT(mixed_b[m], mixed_a[m])) = v;
			}
		}
	}

	void BulkElementBase::d2shape_local_pyoomph(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		const unsigned el_dim = this->dim();
		const unsigned n_node = this->nnode();
		if (!el_dim)
		{
			for (unsigned l = 0; l < n_node; l++)
				for (unsigned k = 0; k < MAX_N2DERIV; k++) d2psi(l, k) = 0.0;
			this->shape(s, psi);
			return;
		}
		// N2deriv[el_dim] entries: 1 / 3 / 6 for el_dim 1 / 2 / 3.
		oomph::DShape d2psids_oomph(n_node, (el_dim == 1 ? 1 : (el_dim == 2 ? 3 : 6)));
		this->d2shape_local(s, psi, dpsi, d2psids_oomph);
		remap_oomph_d2shape_packing(el_dim, n_node, d2psids_oomph, d2psi);
	}

	// The per-space defaults. Anything that reaches these has no implementation for that space; the
	// message is deliberately explicit because the alternative (silently returning zeros) would be
	// wrong numbers rather than an error.
	static void d2shape_local_not_implemented(const BulkElementBase *el, const std::string &space)
	{
		throw_runtime_error("Second spatial derivatives of the shape functions are not implemented for the '" + space +
							"' space of element type " + std::string(typeid(*el).name()) +
							".\nThey are needed for e.g. grad(grad(...)), div(grad(...)) or partial_x(...,2)."
							"\nUse the integrated-by-parts weak form instead, e.g. -weak(grad(u),grad(v)) plus the corresponding surface term.");
	}

	void BulkElementBase::d2shape_local_at_s_C1(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &, oomph::DShape &) const
	{
		d2shape_local_not_implemented(this, "C1");
	}

	void BulkElementBase::d2shape_local_at_s_C2(const oomph::Vector<double> &, oomph::Shape &, oomph::DShape &, oomph::DShape &) const
	{
		d2shape_local_not_implemented(this, "C2");
	}

	// DL shape functions are affine in the local coordinates in every dimension, so their local
	// second derivatives vanish identically. (The *physical* second derivative is still nonzero on
	// curved elements - it comes entirely from the X_{k,ab} term.)
	void BulkElementBase::d2shape_local_at_s_DL(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->dshape_local_at_s_DL(s, psi, dpsi);
		for (unsigned l = 0; l < this->eleminfo.nnode_DL; l++)
			for (unsigned k = 0; k < MAX_N2DERIV; k++) d2psi(l, k) = 0.0;
	}

	// Same fallback rule as the first-derivative versions: without bubble enrichment the TB spaces
	// are the plain ones.
	void BulkElementBase::d2shape_local_at_s_C1TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->d2shape_local_at_s_C1(s, psi, dpsi, d2psi);
	}

	void BulkElementBase::d2shape_local_at_s_C2TB(const oomph::Vector<double> &s, oomph::Shape &psi, oomph::DShape &dpsi, oomph::DShape &d2psi) const
	{
		this->d2shape_local_at_s_C2(s, psi, dpsi, d2psi);
	}

	bool BulkElementBase::supports_second_spatial_derivatives(std::string &why) const
	{
		// A point/ODE element has no local coordinates, so every spatial derivative is zero anyway and
		// the fill short-circuits before any d2shape_local is needed.
		if (!this->dim()) return true;
		why = "element type " + std::string(typeid(*this).name()) + " does not implement second derivatives of its shape functions";
		return false;
	}

	// Interpolates the discontinuous-Lagrange (DL) fields at local coordinate s from their
	// per-node internal-data storage (the DL internal data entries are stored contiguously right
	// after the DG spaces' internal data, hence dg_offset).
	void BulkElementBase::get_interpolated_fields_DL(const oomph::Vector<double> &s, std::vector<double> &res, const unsigned &t) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		res.resize(functable->info_DL.numfields);
		oomph::Shape psi(eleminfo.nnode_DL);
		this->shape_at_s_DL(s, psi);
		unsigned dg_offset=0;
		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto * space_info=functable->present_dg_spaces[i_space];
			dg_offset+=space_info->numfields_new;
		}		
		for (unsigned int fi = 0; fi < functable->info_DL.numfields; fi++)
		{
			res[fi] = 0.0;
			for (unsigned int l = 0; l < eleminfo.nnode_DL; l++)
			{
				res[fi] += psi[l] * this->internal_data_pt(dg_offset+fi)->value(t, l); // TODO: Better direct access of the buffer offset
			}
		}
	}

	// Returns the (spatially constant) D0 field values; D0 storage is single-valued per element,
	// so the local coordinate s is not actually needed for interpolation.
	void BulkElementBase::get_interpolated_fields_D0(const oomph::Vector<double> &, std::vector<double> &res, const unsigned &t) const
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		res.resize(functable->info_D0.numfields);
		unsigned dg_offset=0;
		for (unsigned int i_space=0;i_space<functable->num_present_dg_spaces;i_space++)
		{
			auto * space_info=functable->present_dg_spaces[i_space];
			dg_offset+=space_info->numfields_new;
		}		
		for (unsigned int fi = 0; fi < functable->info_D0.numfields; fi++)
		{
			res[fi] = this->internal_data_pt(functable->info_DL.numfields + dg_offset+fi)->value(t, 0);  // TODO: Better direct access of the buffer offset
		}
	}

	// oomph-lib hook: interpolates all continuous (nodal, base-bulk) field values at local
	// coordinate s and history index t, in the same field ordering used throughout eleminfo.
	void BulkElementBase::get_interpolated_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		const std::vector<std::vector<unsigned>> & node_to_element_index = this->get_nodal_space_index_to_element_index_map();
		values.resize(ncont_interpolated_values(),0.0);
		unsigned index=0;
		for (unsigned int si=0;si<functable->num_present_continuous_spaces;si++)
		{
			auto *space_info=functable->present_continuous_spaces[si];
			oomph::Shape psi(eleminfo.nnode_of_space[space_info->space_index]);
			this->shape_of_space(space_info->space_index, s, psi);			
			for (unsigned int fi = 0; fi < space_info->numfields_basebulk; fi++)
			{
				values[index] = 0.0;
				for (unsigned int l = 0; l < eleminfo.nnode_of_space[space_info->space_index]; l++)
				{
					values[index] += psi[l] * this->node_pt(node_to_element_index[space_info->space_index][l])->value(t, fi+ space_info->nodal_offset_basebulk);
				}
				index++;
			}		
		}
	}

	// Interpolates all discontinuous fields (DL followed by D0) at once, concatenating the results
	// of get_interpolated_fields_DL() and get_interpolated_fields_D0().
	void BulkElementBase::get_interpolated_discontinuous_values(const unsigned &t, const oomph::Vector<double> &s, oomph::Vector<double> &values)
	{
		const JITFuncSpec_Table_FiniteElement_t *functable = jitcode->get_func_table();
		oomph::Vector<double> resDL;
		oomph::Vector<double> resD0;
		if (functable->info_DL.numfields)
			this->get_interpolated_fields_DL(s, resDL, t);
		if (functable->info_D0.numfields)
			this->get_interpolated_fields_D0(s, resD0, t);
		values.resize(resDL.size() + resD0.size());
		for (unsigned int i = 0; i < resDL.size(); i++)
		{
			values[i] = resDL[i];
		}
		for (unsigned int i = 0; i < resD0.size(); i++)
		{
			values[i + resDL.size()] = resD0[i];
		}
	}

	// Plain-text oomph-lib output is not used by pyoomph (see scalar_name_paraview above).
	void BulkElementBase::output(std::ostream &, const unsigned &)
	{
		throw_runtime_error("Not implemented");
	}

	// Number of flux components used by the Z2 error estimator (for adaptive refinement); a
	// separate, typically smaller, set of fluxes is used when estimating errors for eigenvectors
	// (use_eigen_error_estimators), since eigenmodes usually need different quantities than the
	// primary solution to drive mesh refinement.
	unsigned BulkElementBase::num_Z2_flux_terms()
	{
		if (jitcode->get_problem()->get_use_eigen_error_estimators()) return jitcode->get_func_table()->num_Z2_flux_terms_for_eigen;
		else return jitcode->get_func_table()->num_Z2_flux_terms;
	}

	// The compound-flux grouping of this element's Z2 fluxes. Everything below returns the
	// single-group answer (which is also oomph-lib's own default) unless the generated code declared
	// groups, i.e. unless some error criterion on this domain asked to be normalised on its own.
	unsigned BulkElementBase::ncompound_fluxes()
	{
		auto *ft = jitcode->get_func_table();
		const unsigned n = (jitcode->get_problem()->get_use_eigen_error_estimators() ? ft->num_Z2_compound_fluxes_for_eigen : ft->num_Z2_compound_fluxes);
		return (n ? n : 1);
	}

	void BulkElementBase::get_Z2_compound_flux_indices(oomph::Vector<unsigned> &flux_index)
	{
		auto *ft = jitcode->get_func_table();
		const unsigned *idx = (jitcode->get_problem()->get_use_eigen_error_estimators() ? ft->Z2_flux_group_index_for_eigen : ft->Z2_flux_group_index);
		if (!idx) return; // caller initialised the vector to all-zero, which is the single-group case
		for (unsigned int i = 0; i < flux_index.size(); i++) flux_index[i] = idx[i];
	}

	// Per-group normalisation exponent and weight, read by LagrZ2ErrorEstimator once the mesh-global
	// flux norms are known. Defaults reproduce the historical behaviour exactly: divide by the full
	// norm, no weighting.
	double BulkElementBase::Z2_compound_flux_normalize_relative(const unsigned &g)
	{
		auto *ft = jitcode->get_func_table();
		const double *v = (jitcode->get_problem()->get_use_eigen_error_estimators() ? ft->Z2_group_normalize_relative_for_eigen : ft->Z2_group_normalize_relative);
		return (v ? v[g] : 1.0);
	}

	double BulkElementBase::Z2_compound_flux_weight(const unsigned &g)
	{
		auto *ft = jitcode->get_func_table();
		const double *v = (jitcode->get_problem()->get_use_eigen_error_estimators() ? ft->Z2_group_weight_for_eigen : ft->Z2_group_weight);
		return (v ? v[g] : 1.0);
	}

	// Evaluates the Z2-error-estimator flux vector at local coordinate s via the JIT-generated
	// GetZ2Fluxes (or GetZ2FluxesForEigen) function, used by oomph-lib's Z2 error estimator to
	// drive adaptive mesh refinement.
	void BulkElementBase::get_Z2_flux(const oomph::Vector<double> &s, oomph::Vector<double> &flux)
	{
		JITShapeInfo_t *const shape_info = this->get_shape_info();
		bool has_fluxes=(jitcode->get_problem()->get_use_eigen_error_estimators() ? jitcode->get_func_table()->GetZ2FluxesForEigen : jitcode->get_func_table()->GetZ2Fluxes );
		if (has_fluxes)
		{
			this->interpolate_hang_values(); // XXX This should be moved to somewhere else, after each update of any values
			this->prepare_shape_buffer_for_integration(jitcode->get_func_table()->shapes_required_Z2Fluxes, 0);
			double JLagr;
			this->fill_shape_info_at_s(s, 0, jitcode->get_func_table()->shapes_required_Z2Fluxes, JLagr, 0);
			this->set_remaining_shapes_appropriately(shape_info,jitcode->get_func_table()->shapes_required_Z2Fluxes);
			if (jitcode->get_problem()->get_use_eigen_error_estimators())
			{
				jitcode->get_func_table()->GetZ2FluxesForEigen(&eleminfo, shape_info, &(flux[0]));
			}
			else
			{
				jitcode->get_func_table()->GetZ2Fluxes(&eleminfo, shape_info, &(flux[0]));
			}
		}
	}

   // Total number of DG (discontinuous-Galerkin, per-node-but-not-shared) fields across all
   // present DG spaces; base_bulk_only restricts the count to fields defined in the bulk element
   // itself, excluding additional fields only present on interfaces.
   unsigned BulkElementBase::num_DG_fields(bool base_bulk_only)
   {
    auto *ft=jitcode->get_func_table();
    if (base_bulk_only)
    {
	  unsigned cnt=0;
	  for (unsigned int i=0;i<ft->num_present_dg_spaces;i++)
	  {
		cnt+=ft->present_dg_spaces[i]->numfields_basebulk;
	  }
	  return cnt;      
    }
    else
    {
	  unsigned cnt=0;
	  for (unsigned int i=0;i<ft->num_present_dg_spaces;i++)
	  {
		cnt+=ft->present_dg_spaces[i]->numfields;
	  }
	  return cnt;      
    }
   }
   
   // Interpolates all fields of one DG space at local coordinate s. Fields inherited from a
   // parent/bulk element ("external", the first `nexternal` of them) are read from external
   // data; genuinely new fields defined at this level are read from internal data.
   void BulkElementBase::get_DG_fields_at_s(unsigned int space_index, unsigned history_index,const oomph::Vector<double> &s, oomph::Vector<double> &result) const
   {
	auto *ft=jitcode->get_func_table();
	auto & space_info=ft->dg_spaces[space_index];
	result.resize(space_info.numfields);
     for (unsigned int i=0;i<space_info.numfields;i++) result[i]=0.0;
     oomph::Shape psi(eleminfo.nnode_of_space[space_info.space_index]);
	 this->shape_of_space(space_info.space_index, s, psi);     
     unsigned nexternal=space_info.numfields-space_info.numfields_new;
     for (unsigned int i=0;i<nexternal;i++)
     {
      for (unsigned int l=0;l<eleminfo.nnode_of_space[space_info.space_index];l++) 
      {
       result[i]+=external_data_pt(space_info.external_offset_bulk+i)->value(history_index,this->get_DG_node_index(space_info.space_index, i,l))*psi[l];
      } 
     }
     for (unsigned int i=nexternal;i<space_info.numfields;i++)
     {
      for (unsigned int l=0;l<eleminfo.nnode_of_space[space_info.space_index];l++) 
      {
       result[i]+=internal_data_pt(space_info.internal_offset_new+i-nexternal)->value(history_index,l)*psi[l];
      }
     }
   }

	// Allocates the oomph::Data objects backing this element's non-nodal field storage: one
	// internal Data (sized to the number of nodes in that space) per newly-defined DG field, one
	// per DL field (shared across all DL "nodes" of the element), and one single-value internal
	// Data per D0 field (element-constant). Must be called once the eleminfo node counts per
	// space are known, and before fill_element_info()/further_build() try to access this storage.
	void BulkElementBase::allocate_discontinous_fields()
	{
	   // DG Fields.
		//Only add the fields directly added in this dimension. Parent degrees will be external data	   
		for (unsigned int i_space=0;i_space<jitcode->get_func_table()->num_present_dg_spaces;i_space++)
		{
			auto * space_info=jitcode->get_func_table()->present_dg_spaces[i_space];
			if (eleminfo.nnode_of_space[space_info->space_index] > 0)
			{
				for (unsigned int fi = 0; fi < space_info->numfields_new; fi++)
				{
					this->add_internal_data(new oomph::Data(eleminfo.nnode_of_space[space_info->space_index]), false);
				}
			}
		}
		
			
		if (eleminfo.nnode_DL > 0)
		{
			for (unsigned int fi = 0; fi < jitcode->get_func_table()->info_DL.numfields; fi++)
			{
				this->add_internal_data(new oomph::Data(eleminfo.nnode_DL), false);
				//		          std::cout << "  AFTER DL " << fi << "  " << this->ninternal_data() << " INT DATA" << std::endl <<std::flush;

			}
		}

		for (unsigned int fi = 0; fi < jitcode->get_func_table()->info_D0.numfields; fi++)
		{
			this->add_internal_data(new oomph::Data(1), false);
		}

//          std::cout << "ALLOCATED " << this->ninternal_data() << " INT DATA" << std::endl <<std::flush;
		
	}

	// Global cache of pre-built oomph-lib integration schemes (Gauss rules etc.), shared across element instances.
	IntegrationSchemeStorage integration_scheme_storage;

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Static per-element-class lookup tables used throughout this file and by the JIT-generated code to
	// translate between the different node numbering schemes:
	//  - Dummy_Value_Interpolation_Map[space]: for each "dummy" node of that space (a node that isn't an
	//    independent dof of the space, e.g. a C1 dummy node on a C2-only element), the list of node indices
	//    {dummy_node, source_node_1, source_node_2, ...} whose average defines its (non-independent) value.
	//  - Nodal_Space_Index_To_Element_Index_Map[space]: maps the local node index within a given field space
	//    (C1/C1TB/C2/C2TB, i.e. "the n-th node that carries this space's dofs") to the local node index within
	//    the full element node numbering.
	//  - Element_Index_To_Nodal_Space_Index_Map[space]: the inverse mapping (element-local node index -> index
	//    within that space's own node numbering, or -1 if that node does not carry dofs of that space).
	//  - Possible_Face_Indices: the valid face_index values that can be passed to construct_face_element()/
	//    boundary_node_pt() etc. for that element type (e.g. {-2,-1,1,2} for the 4 faces of a quad).
	// These tables are all empty ({}) for spaces the given element type does not support.
	//////////////////////////////////////////////////////////////////////////////////////////////////////////

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementBase::Dummy_Value_Interpolation_Map=
	{
		{}, // C2TB
		{}, // C2
		{}, // C1TB
		{}  // C1
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementLine1dC2::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{{1,0,2}}, // C1TB
		{{1,0,2}}  // C1
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkTElementLine1dC2::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{{1,0,2}}, // C1TB
		{{1,0,2}}  // C1
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementQuad2dC2::Dummy_Value_Interpolation_Map={
		{}, // C2TB
		{}, // C2
		{{1, 0,2}, {3,0,6}, {5,2,8}, {7,6,8}, {4,0,2,6,8} }, // C1TB
		{{1, 0,2}, {3,0,6}, {5,2,8}, {7,6,8}, {4,0,2,6,8} }  // C1
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTri2dC1TB::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{ }, // C1TB
		{{3, 0,1,2} }  // C1
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTri2dC2::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{{3, 0,1}, {4, 1,2}, {5, 0, 2} }  // C1		
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTri2dC2TB::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{{6, 0, 1, 2}}, // C2
		{{3, 0,1}, {4, 1,2}, {5, 0, 2}}, // C1TB
		{{3, 0,1}, {4, 1,2}, {5, 0, 2}, {6, 0, 1, 2} }  // C1		
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementBrick3dC2::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{{1, 0, 2}, {3, 0, 6}, {5, 2, 8}, {7, 6, 8}, {4, 0,2,6,8},
		 {19, 18, 20}, {21, 18, 24}, {23, 20, 26}, {25, 24, 26}, {22, 18,20,24,26},
		 {9, 0, 18}, {11, 2, 20}, {15, 6, 24}, {17, 8, 26}, {10, 0,2,18,20}, {12, 0,6,18,24},  {16, 6,8,24,26}, {14, 2,8,20,26},
		 {13, 0,2,6,8,18,20,24,26}},  // C1TB		
		{{1, 0, 2}, {3, 0, 6}, {5, 2, 8}, {7, 6, 8}, {4, 0,2,6,8},
		 {19, 18, 20}, {21, 18, 24}, {23, 20, 26}, {25, 24, 26}, {22, 18,20,24,26},
		 {9, 0, 18}, {11, 2, 20}, {15, 6, 24}, {17, 8, 26}, {10, 0,2,18,20}, {12, 0,6,18,24},  {16, 6,8,24,26}, {14, 2,8,20,26},
		 {13, 0,2,6,8,18,20,24,26}}  // C1		
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTetra3dC1TB::Dummy_Value_Interpolation_Map={
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{{4, 0,1,2,3}}  // C1
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTetra3dC2::Dummy_Value_Interpolation_Map=
	{
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		{{4,0,1},{5,0,2},{6,0,3},{7,1,2},{8,2,3},{9,1,3}}  // C1		
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementTetra3dC2TB::Dummy_Value_Interpolation_Map=
	{
		{}, // C2TB 
		{{10,0,1,3},{11,0,1,2},{12,0,2,3},{13,1,2,3},{14,0,1,2,3}}, // C2
		{{4,0,1},{5,0,2},{6,0,3},{7,1,2},{8,2,3},{9,1,3},{10,0,1,3},{11,0,1,2},{12,0,2,3},{13,1,2,3}}, // C1TB
		{{4,0,1},{5,0,2},{6,0,3},{7,1,2},{8,2,3},{9,1,3},{10,0,1,3},{11,0,1,2},{12,0,2,3},{13,1,2,3},{14,0,1,2,3}}  // C1		
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementWedge3dC2::Dummy_Value_Interpolation_Map=
	{
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		// Bottom-layer edge mids are nodes 3,4,5 over corner pairs (0,1),(0,2),(1,2) -- the same pattern the
		// top layer (15,16,17 over (12,13),(12,14),(13,14)) already used. Entries 4 and 5 used to have their
		// pairs exchanged, so a C1-constrained field on a wedge was tied to the wrong two corners: node 4
		// sits at the 0-2 midpoint and node 5 at the 1-2 midpoint, not the other way round. Verified against
		// the actual nodal positions -- for an affine wedge the geometry and a C1 field interpolate
		// identically, so each listed corner set must average to the target node's position.
		{{3,0,1},{4,0,2},{5,1,2},{15,12,13},{16,12,14},{17,13,14},{6,0,12},{7,1,13},{8,2,14},{9,0,1,12,13},{10,0,2,12,14},{11,1,2,13,14}}  // C1
	};

	const std::vector<std::vector<std::vector<unsigned>>> BulkElementPyramid3dC2::Dummy_Value_Interpolation_Map=
	{
		{}, // C2TB 
		{}, // C2
		{}, // C1TB
		// Node 13 is the BASE QUAD CENTRE, so its C1 value is the average of all FOUR base corners -- the
		// pyramid's base interpolation is bilinear. It used to be listed as {13,0,2}, i.e. the mean of one
		// diagonal, which is only equal to the bilinear centre when v0+v2 == v1+v3 and therefore degraded
		// a C1-constrained field to something that is not the element's C1 space at all (visible as the
		// Green identity int(v)==int(u^2) failing by ~30% even on a NON-adaptive pyramid mesh).
		{{5,0,1},{6,1,2},{7,2,3},{8,0,3},{9,0,4},{10,1,4},{11,2,4},{12,3,4},{13,0,1,2,3}}  // C1
	};
}
