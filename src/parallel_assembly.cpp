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

// The OpenMP element loop. See dev_docs/openmp_assembly.md for the reasoning; the short version:
//
// The expensive half of an assembly is evaluating each element's dense block (56-87% of a Jacobian
// assembly, dev_docs/structural_assembly.md section 2), and that half is embarrassingly parallel
// now that commit 25b24ea gave every thread its own shape buffer. The cheap half, scattering those
// blocks into the CSR value array, is not: two elements sharing a dof write the same slot.
//
// Rather than serialise those writes with atomics (which also makes the summation order, and hence
// the last bits of every entry, depend on the thread schedule), the scatter is INVERTED into a
// gather. The elements are processed in chunks; within a chunk, phase 1 writes every element's
// block into a private slice of a scratch buffer, and phase 2 gives each thread a disjoint range of
// TARGET SLOTS and has it sum that slot's contributions. Because the gather index is stably sorted
// by slot, each slot's contributions arrive in element order - exactly the order the serial loop
// adds them in - and the chunks themselves run in element order. The result is bit-identical to the
// serial assembly, which is the point: --omp changes the time and nothing else.
//
// The price is the gather index (two ints per scatter entry) and the block scratch (bounded by the
// chunk size, ~32 MB), both built once per equation numbering.

#include "problem.hpp"
#include "elements.hpp"
#include "exception.hpp"
#include "thread_state.hpp"

#include "assembly_handler.h" // AssemblyHandler::ndof/eqn_number/get_all_vectors_and_matrices, called per element here
#include "bifurcation.hpp" // CustomMultiAssembleHandler, refused below

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#ifdef __linux__
#include <sched.h>
#endif
#include <numeric>

#ifdef PYOOMPH_HAS_OPENMP
#include <omp.h>
#endif
#ifdef PYOOMPH_HAS_GCD
#include <dispatch/dispatch.h>
#endif
#include <chrono>

// One umbrella for "the threaded element loop is compiled in", whichever backend provides it:
// OpenMP (Linux/Windows, and macOS when a libomp is linked) or GCD/libdispatch (macOS only). The
// GCD backend exists so a macOS wheel can thread the assembly WITHOUT shipping a second OpenMP
// runtime (LLVM libomp) alongside MKL's Intel libiomp5 - the two collide as "OMP: Error #15" and
// crash Pardiso on an Intel mac. The gather design (chunks in element order, a per-slot gather) is
// backend-agnostic and bit-identical to serial either way; only the parallel region below differs.
#if defined(PYOOMPH_HAS_OPENMP) || defined(PYOOMPH_HAS_GCD)
#define PYOOMPH_HAS_PARALLEL_ASSEMBLY 1
#endif

namespace
{
	// A portable monotonic clock in seconds, so the phase timings and the PYOOMPH_REPORT_OMP_ASSEMBLY
	// report do not depend on omp_get_wtime (which only exists with OpenMP). Used by both backends.
	inline double pa_now()
	{
		return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
	}
#ifdef PYOOMPH_HAS_GCD
	// dispatch_apply_f takes a C function pointer and a context; this trampoline lets that context be
	// an ordinary C++ lambda with by-reference capture, so the GCD blocks below can reference the same
	// locals the OpenMP region does without the block-capture gymnastics (atomics and the plan are not
	// copyable, which a plain ^{} capture would try to do).
	template <class F>
	void pa_apply_trampoline(void *ctx, size_t i) { (*static_cast<F *>(ctx))(i); }
#endif
}

namespace pyoomph
{

	void Problem::set_num_assembly_threads(unsigned n)
	{
		if (n < 1) n = 1;
		if (n == num_threads) return;
		num_threads = n;
		// The plan is deliberately independent of the thread count (see the chunking in
		// acquire_parallel_assembly_plan), so it survives this and a benchmark that alternates thread
		// counts does not pay for a rebuild each time.
		// A refusal already reported may not apply any more (going back to one thread, say), and a new
		// setting deserves to hear about the ones that still do.
		reported_parallel_refusals.clear();
#ifndef PYOOMPH_HAS_PARALLEL_ASSEMBLY
		if (n > 1)
			oomph::oomph_info << "pyoomph: set_num_threads(" << n << ") was asked for, but this build has no "
								 "threaded-assembly backend (neither OpenMP nor GCD) - the element loop stays "
								 "serial. The linear solver still gets the count."
							  << std::endl;
#endif
	}

	void Problem::report_parallel_refusal(const std::string &reason)
	{
		report_parallel_note("threaded assembly is not used here: " + reason + " (assembling this on one thread)");
	}

	// Once per distinct message, for the whole life of the Problem: these conditions are properties of
	// the configuration, so saying them once is informative and saying them per Newton step is noise.
	void Problem::report_parallel_note(const std::string &msg)
	{
		if (reported_parallel_refusals.count(msg)) return;
		reported_parallel_refusals.insert(msg);
		oomph::oomph_info << "pyoomph: " << msg << std::endl;
	}

	// The diagnostics that accumulate into process-wide counters, sets and timers from inside the
	// element loop. They are all off unless explicitly switched on, and when they ARE on their whole
	// purpose is to be believed - a threaded run would give them garbage - so the loop stays serial.
	static const char *const __assembly_diagnostic_env[] = {
		"PYOOMPH_REPORT_EXT_DATA", "PYOOMPH_REPORT_HANG_FILL_CACHE", "PYOOMPH_REPORT_HANG_FILL_TIME",
		"PYOOMPH_REPORT_NOHANG_DISPATCH", "PYOOMPH_PARANOID_HANG_FILL_CACHE", "PYOOMPH_POISON_UNREQUIRED",
		"PYOOMPH_PARANOID_ALE_IDENTITY", "PYOOMPH_FROZEN_FILL_BREAKDOWN", "PYOOMPH_TIME_ADD_RESIDUAL",
		"PYOOMPH_MEASURE_SKIP_HANG_FILLS", NULL};

	// How many CPUs this PROCESS is actually allowed to run on. Under mpirun that is normally one per
	// rank, which silently turns --omp into a no-op: measured on a 32 k-dof cavity, "mpirun -n 1 ...
	// --omp 4" was 0.99x of one thread, and the same run with --bind-to none was 1.69x. Worth a note
	// rather than a refusal - the run is still correct, it just gains nothing.
	static unsigned affinity_cpu_count()
	{
#ifdef __linux__
		cpu_set_t set;
		CPU_ZERO(&set);
		if (sched_getaffinity(0, sizeof(set), &set) == 0) return (unsigned)CPU_COUNT(&set);
#endif
		return 0; // "unknown", which must not be mistaken for "too few"
	}

	bool Problem::parallel_assembly_possible()
	{
		if (num_threads <= 1) return false; // The normal case: nothing to report, nothing to do
#ifndef PYOOMPH_HAS_PARALLEL_ASSEMBLY
		report_parallel_refusal("this build has no threaded-assembly backend (the CMake configuration found "
								"neither OpenMP nor, on macOS, enabled the GCD backend)");
		return false;
#else
		for (const char *const *e = __assembly_diagnostic_env; *e; e++)
		{
			if (getenv(*e))
			{
				report_parallel_refusal(std::string(*e) + " is set, and its counters are not per thread");
				return false;
			}
		}
		// A finite-difference Jacobian perturbs a NODAL value or position and re-enters the residual,
		// i.e. it writes state shared with every neighbouring element. Two elements of such a domain
		// cannot be assembled at the same time by any scheme. Detected per code, as the multi-assembly
		// already does (elements_assembly.cpp).
		for (auto *code : jit_codes)
		{
			if (!code) continue;
			const JITFuncSpec_Table_FiniteElement_t *ft = code->get_func_table();
			if (!ft) continue;
			if (ft->fd_jacobian || ft->fd_position_jacobian)
			{
				report_parallel_refusal(std::string("domain '") + (ft->domain_name ? ft->domain_name : "?") +
										"' uses a finite-difference Jacobian, which perturbs data shared "
										"between elements");
				return false;
			}
		}
		// A handler that has no per-element route at all. The threaded loop below assembles by calling
		// handler->get_residuals()/get_all_vectors_and_matrices() once per element, and
		// CustomMultiAssembleHandler answers both with throw_runtime_error("Residual called"): it is
		// driven from _assemble_multiassembly(), which sweeps the elements itself and asks each one for
		// several named contributions at once. Refusing here puts such a solve back on that route
		// instead of failing it -- deflated_solve.py died with "during threaded element assembly:
		// Residual called" the moment the tutorials were run with --omp 2. (Deflation no longer goes
		// through this handler at all, see dev_docs/deflation.md; the CustomBifurcationTracker family
		// still does, and is what this refusal is for now.)
		if (dynamic_cast<CustomMultiAssembleHandler *>(this->assembly_handler_pt()))
		{
			report_parallel_refusal("the active assembly handler assembles several contributions per "
									"element at once and has no per-element residual/Jacobian to call");
			return false;
		}
		// Not a refusal: the assembly is still right, it just will not get faster. Reported through the
		// same once-only channel because the cause and the fix are both invisible from the timings.
		const unsigned cpus = affinity_cpu_count();
		if (cpus && cpus < num_threads)
		{
			report_parallel_note(
				"the threaded assembly will not actually run in parallel: this process may use only " +
				std::to_string(cpus) + " CPU(s), but " + std::to_string(num_threads) + " threads were asked "
				"for, so they take turns on those instead of running side by side. Under mpirun that is the "
				"default - each rank is pinned to one core. Relaunch with 'mpirun --bind-to none', or with a "
				"binding that gives each rank as many cores as it has threads. Nothing is wrong with the "
				"result either way; it just will not get faster.");
		}
		return true;
#endif
	}

	// One serial sweep of interpolate_hang_values(), stamped with a fresh pass id.
	//
	// interpolate_hang_values() pushes a hanging node's flattened value back into the node itself
	// (elements_hanging.cpp), so it writes storage shared with every element that touches that node.
	// Doing it once here, serially, and letting the workers adopt the stamp means HangInterpGate::skip()
	// matches for every element and no worker writes a node at all. That is not merely a lock avoided:
	// it is the work done once instead of once per element.
	unsigned long Problem::prepare_hanging_values_for_parallel_assembly(const ParallelAssemblyPlan &plan)
	{
		HangInterpPassScope pass; // opens a fresh id on THIS thread
		for (size_t i = 0; i < plan.elements.size(); i++)
		{
			BulkElementBase *be = dynamic_cast<BulkElementBase *>(mesh_pt()->element_pt(plan.elements[i]));
			if (be) be->interpolate_hang_values();
		}
		return __hang_interp_pass; // still inside the scope, so this is the id just stamped
	}

	// Builds, or confirms, the gather index for this request. Everything here depends on the equation
	// numbering alone, so a converged Newton solve builds it once and reuses it on every step.
	int Problem::acquire_parallel_assembly_plan(const FrozenAssemblyRequest &req)
	{
		const unsigned long gen = this->get_jacobian_structure_id();
		if (!gen) return -1; // Value-dependent pattern: nothing may be precomputed from it
		oomph::AssemblyHandler *const handler = this->assembly_handler_pt();
		const unsigned long now = ++parallel_assembly_plan_clock;

		// Reuse test. The scatter-map POINTERS are part of the key on purpose: a rebuilt pattern moves
		// them, and comparing them is both cheaper and stricter than comparing the arrays.
		for (size_t p = 0; p < parallel_assembly_plans.size(); p++)
		{
			ParallelAssemblyPlan &plan = parallel_assembly_plans[p];
			if (plan.valid && plan.generation == gen && plan.el_lo == req.el_lo &&
				plan.el_hi_plus_one == req.el_hi_plus_one && plan.handler == (const void *)handler &&
				plan.n_vector == req.n_vector && plan.n_matrix == req.n_matrix && plan.n_map == req.n_map &&
				plan.res_row_map == (const void *)req.res_row_map && plan.res_row_map_n == req.res_row_map_n &&
				plan.map_ptr.size() == req.n_map)
			{
				bool same = true;
				for (unsigned g = 0; g < req.n_map; g++)
				{
					// The first matrix using map g carries its pointers.
					unsigned m0 = req.n_matrix;
					for (unsigned m = 0; m < req.n_matrix; m++) if (req.map_of_matrix[m] == g) { m0 = m; break; }
					if (m0 == req.n_matrix || plan.map_ptr[g] != req.scatter_slot[m0]) { same = false; break; }
				}
				if (same) { plan.last_used = now; return (int)p; }
			}
		}

		// Not found: take a free slot, else the least recently used one.
		size_t pslot = parallel_assembly_plans.size();
		for (size_t p = 0; p < parallel_assembly_plans.size(); p++)
			if (!parallel_assembly_plans[p].valid) { pslot = p; break; }
		if (pslot == parallel_assembly_plans.size())
		{
			if (parallel_assembly_plans.size() < max_parallel_assembly_plans)
				parallel_assembly_plans.push_back(ParallelAssemblyPlan());
			else
			{
				pslot = 0;
				for (size_t p = 1; p < parallel_assembly_plans.size(); p++)
					if (parallel_assembly_plans[p].last_used < parallel_assembly_plans[pslot].last_used) pslot = p;
			}
		}
		ParallelAssemblyPlan &plan = parallel_assembly_plans[pslot];
		plan.clear();
		plan.last_used = now;
		plan.generation = gen;
		plan.el_lo = req.el_lo;
		plan.el_hi_plus_one = req.el_hi_plus_one;
		plan.handler = (const void *)handler;
		plan.n_vector = req.n_vector;
		plan.n_matrix = req.n_matrix;
		plan.n_map = req.n_map;
		plan.res_row_map = (const void *)req.res_row_map;
		plan.res_row_map_n = req.res_row_map_n;

		// --- 1. The element slice, filtered exactly as the serial loops filter it
		for (unsigned long e = req.el_lo; e < req.el_hi_plus_one; e++)
		{
			oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(e);
#ifdef OOMPH_HAS_MPI
			if (elem_pt->is_halo()) continue;
#endif
			const unsigned nv = handler->ndof(elem_pt);
			if (!nv) continue;
			plan.elements.push_back((int)e);
			plan.nvar.push_back((int)nv);
		}
		const size_t nel = plan.elements.size();
		if (!nel) { plan.clear(); return -1; }
		// Does the pre-pass have anything to do at all? The classification is element-local and cached,
		// and it is the same test HangInterpGate::skip() makes.
		for (size_t i = 0; i < nel && !plan.needs_hang_prepass; i++)
		{
			BulkElementBase *be = dynamic_cast<BulkElementBase *>(mesh_pt()->element_pt(plan.elements[i]));
			if (be && be->would_interpolate_hang_values())
				plan.needs_hang_prepass = true;
		}

		// --- 2. Chunking. The scratch buffer holds one chunk's blocks for every matrix at once, so the
		// chunk length is chosen from that budget rather than from the element count. Sequential chunks
		// are what keeps the gather's summation order equal to the serial one.
		// 1 MB of doubles per matrix. Measured rather than reasoned: swept from 8 K to 4 M doubles at
		// four threads on an 8 MB L3, on both an expensive element kernel (2D Q2/Q1 Navier-Stokes,
		// 89 k dofs) and a cheap one (Q2 Poisson, 250 k dofs), the whole range from 16 K upwards lies
		// within ~5 % and this value is at or next to the best of it in both.
		//
		// The two ends fail for different reasons, and only one of them is a cliff. Below ~16 K the
		// chunk holds a few dozen elements and its two barriers stop being amortised: 8 K measured
		// 28 % worse. Above the cache, phase 2 - which reads the block buffer in TARGET-SLOT order,
		// i.e. essentially at random within the chunk, where the serial loop reads each block
		// sequentially out of L1 - roughly doubles, 3.8 ms to 6.8 ms on the Navier-Stokes case. That
		// is real but it is only ~6 % of the loop, so the total barely moves and there is no cliff at
		// the large end. (An earlier version of this comment claimed a 1.6x slowdown from a 32 MB
		// chunk. That measurement was confounded: the plan was being rebuilt on every call, which is
		// what actually cost the time. See the chunking note below.)
		long budget_doubles = 128L * 1024L;
		if (const char *env = getenv("PYOOMPH_ASSEMBLY_CHUNK_DOUBLES")) { long v = atol(env); if (v > 0) budget_doubles = v; }
		if (req.n_matrix) budget_doubles /= (long)req.n_matrix;
		if (budget_doubles < 1024) budget_doubles = 1024;
		// A floor on the chunk length as well: too short a chunk pays two barriers for too little work,
		// and leaves threads idle at both ends of it. Deliberately NOT a function of num_threads: the
		// plan would then have to be rebuilt whenever the thread count changed, and building it sorts
		// every scatter entry - as expensive as several assemblies.

		plan.block_base.resize(nel);
		plan.res_base.resize(nel);
		plan.chunk_start.push_back(0);
		long acc_block = 0, acc_res = 0;
		for (size_t i = 0; i < nel; i++)
		{
			const long nv = plan.nvar[i];
			const long need = nv * nv;
			// A minimum length, so that a chunk is never so short that its two barriers cost more than
			// its work - the one end of the range that measurably hurts (see the budget above). Small
			// on purpose: it only ever binds for elements so large that the budget is blown by a
			// handful of them (a 3D C2 block with an analytic Hessian), and there a handful already
			// carries far more work per barrier than the sweep's worst case did.
			const size_t min_chunk = 8;
			const size_t in_chunk = i - (size_t)plan.chunk_start.back();
			if (in_chunk >= min_chunk && acc_block + need > budget_doubles)
			{
				plan.chunk_block_size.push_back(acc_block);
				plan.chunk_res_size.push_back(acc_res);
				plan.chunk_start.push_back((long)i);
				acc_block = 0; acc_res = 0;
			}
			plan.block_base[i] = acc_block;
			plan.res_base[i] = acc_res;
			acc_block += need;
			acc_res += nv;
		}
		plan.chunk_block_size.push_back(acc_block);
		plan.chunk_res_size.push_back(acc_res);
		plan.chunk_start.push_back((long)nel);
		const unsigned nchunk = plan.nchunk();
		for (unsigned c = 0; c < nchunk; c++)
		{
			plan.max_block_size = std::max(plan.max_block_size, plan.chunk_block_size[c]);
			plan.max_res_size = std::max(plan.max_res_size, plan.chunk_res_size[c]);
		}

		// --- 3. The matrix gathers, one per deduplicated scatter map
		plan.map_ptr.assign(req.n_map, NULL);
		plan.mat_gather.resize(req.n_map);
		for (unsigned g = 0; g < req.n_map; g++)
		{
			unsigned m0 = req.n_matrix;
			for (unsigned m = 0; m < req.n_matrix; m++) if (req.map_of_matrix[m] == g) { m0 = m; break; }
			if (m0 == req.n_matrix) { plan.clear(); return -1; }
			plan.map_ptr[g] = req.scatter_slot[m0];
			const int *eoff = req.element_offset[m0];
			const int *slot = req.scatter_slot[m0];
			const int *src = req.scatter_source[m0];
			ParallelAssemblyPlan::Gather &gt = plan.mat_gather[g];
			gt.chunk_gather_start.push_back(0);
			std::vector<std::pair<int, int>> pairs;
			for (unsigned c = 0; c < nchunk; c++)
			{
				pairs.clear();
				for (long i = plan.chunk_start[c]; i < plan.chunk_start[c + 1]; i++)
				{
					const int e = plan.elements[i];
					const long base = plan.block_base[i];
					for (int k = eoff[e]; k < eoff[e + 1]; k++)
						pairs.push_back(std::make_pair(slot[k], (int)(base + src[k])));
				}
				// STABLE, and only on the slot: the pairs were pushed in element order and, within an
				// element, in the order the serial loop walks them, so stability is exactly what makes
				// each slot's sum reproduce the serial one bit for bit.
				std::stable_sort(pairs.begin(), pairs.end(),
								 [](const std::pair<int, int> &a, const std::pair<int, int> &b)
								 { return a.first < b.first; });
				for (size_t k = 0; k < pairs.size(); k++)
				{
					gt.slot.push_back(pairs[k].first);
					gt.src.push_back(pairs[k].second);
				}
				gt.chunk_gather_start.push_back((long)gt.slot.size());
			}
		}

		// --- 4. The residual gather. Same construction, with the equation number (or, on a distributed
		// problem, its row in this rank's my_eqns) as the slot.
		{
			ParallelAssemblyPlan::Gather &gt = plan.res_gather;
			gt.chunk_gather_start.push_back(0);
			std::vector<std::pair<int, int>> pairs;
			for (unsigned c = 0; c < nchunk; c++)
			{
				pairs.clear();
				for (long i = plan.chunk_start[c]; i < plan.chunk_start[c + 1]; i++)
				{
					oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(plan.elements[i]);
					const long base = plan.res_base[i];
					const int nv = plan.nvar[i];
					for (int l = 0; l < nv; l++)
					{
						const unsigned eqn = handler->eqn_number(elem_pt, l);
						int row = (int)eqn;
						if (req.res_row_map)
						{
							const unsigned *found = std::lower_bound(req.res_row_map, req.res_row_map + req.res_row_map_n, eqn);
							if (found == req.res_row_map + req.res_row_map_n || *found != eqn) continue;
							row = (int)(found - req.res_row_map);
						}
						pairs.push_back(std::make_pair(row, (int)(base + l)));
					}
				}
				std::stable_sort(pairs.begin(), pairs.end(),
								 [](const std::pair<int, int> &a, const std::pair<int, int> &b)
								 { return a.first < b.first; });
				for (size_t k = 0; k < pairs.size(); k++)
				{
					gt.slot.push_back(pairs[k].first);
					gt.src.push_back(pairs[k].second);
				}
				gt.chunk_gather_start.push_back((long)gt.slot.size());
			}
		}

		plan.valid = true;
		if (getenv("PYOOMPH_REPORT_OMP_ASSEMBLY"))
			oomph::oomph_info << "pyoomph OMP assembly: built plan slot " << pslot << " for " << nel
							  << " elements in " << nchunk << " chunks, "
							  << (plan.mat_gather.empty() ? 0 : plan.mat_gather[0].slot.size())
							  << " matrix gather entries" << std::endl;
		return (int)pslot;
	}

#ifdef PYOOMPH_HAS_PARALLEL_ASSEMBLY
	namespace
	{
		// The per-thread state a worker must own for the duration of the region: its own residual-form
		// channel, the pre-pass's hang stamp (thread_local, so it does not inherit), and - on the way
		// out - its shape-buffer chain handed back to the pool.
		class AssemblyWorkerScope
		{
			CurrentResJacThreadScope crj;
			unsigned long prev_pass;

		public:
			AssemblyWorkerScope(unsigned long pass_id) : prev_pass(__hang_interp_pass)
			{
				__hang_interp_pass = pass_id;
			}
			~AssemblyWorkerScope()
			{
				__hang_interp_pass = prev_pass;
				release_thread_shape_buffer();
			}
		};

		// The half-open range of a gather array that thread `t` of `nt` owns, moved outwards to the
		// nearest slot boundaries so that no slot is touched by two threads. Equal slots are adjacent
		// because the array is sorted by slot.
		inline void gather_range_for_thread(const int *slot, long lo, long hi, int t, int nt, long &a, long &b)
		{
			const long n = hi - lo;
			a = lo + (n * t) / nt;
			b = lo + (n * (t + 1)) / nt;
			// `a < hi` is not redundant: a can start at hi (an empty share), and slot[hi] is one past
			// the last entry of this chunk - which for the LAST chunk is one past the whole array.
			while (a > lo && a < hi && slot[a - 1] == slot[a]) a++;
			while (b > lo && b < hi && slot[b - 1] == slot[b]) b++;
			if (a > b) a = b;
		}
	}
#endif

	// The request the two non-distributed frozen loops need, assembled from the cache slots they are
	// already holding. Several matrices routinely share one slot (the Jacobian and every derivative of
	// it), so the maps are deduplicated here and the gather index built once per distinct pattern.
	bool Problem::try_parallel_frozen_assembly(const std::vector<int> &slots, const oomph::Vector<double *> &value,
											   const oomph::Vector<double *> &residuals, unsigned long el_lo,
											   unsigned long el_hi_plus_one)
	{
		if (!parallel_assembly_possible()) return false;
		const unsigned n_matrix = (unsigned)value.size();
		if (slots.size() != n_matrix) return false;
		const unsigned long n_element = mesh_pt()->nelement();

		FrozenAssemblyRequest req;
		req.el_lo = el_lo;
		req.el_hi_plus_one = el_hi_plus_one;
		req.n_matrix = n_matrix;
		req.n_vector = (unsigned)residuals.size();
		req.map_of_matrix.resize(n_matrix);
		std::vector<int> distinct;
		for (unsigned m = 0; m < n_matrix; m++)
		{
			if (slots[m] < 0 || (size_t)slots[m] >= frozen_sparsity_cache.size()) return false;
			const FrozenSparsity &sp = frozen_sparsity_cache[slots[m]];
			if (sp.element_offset.size() < (size_t)n_element + 1) return false;
			size_t g = 0;
			while (g < distinct.size() && distinct[g] != slots[m]) g++;
			if (g == distinct.size()) distinct.push_back(slots[m]);
			req.map_of_matrix[m] = (unsigned)g;
			req.element_offset.push_back(sp.element_offset.data());
			req.scatter_slot.push_back(sp.scatter_slot.data());
			req.scatter_source.push_back(sp.scatter_source.data());
		}
		req.n_map = (unsigned)distinct.size();
		req.value.assign(value.begin(), value.end());
		req.residual.assign(residuals.begin(), residuals.end());
		req.verify = verify_frozen_sparsity;
		return parallel_assemble_frozen(req);
	}

	// The residual-only element loop. oomph-lib's own is a plain serial sweep over the elements
	// (problem.cc, Problem::get_residuals), and there is nothing to freeze about it - the residual has
	// no sparsity pattern - so the gather here indexes equation numbers directly. Everything except
	// the loop is left to oomph-lib: the threaded branch only runs on a single rank with threads
	// actually asked for, and every other case is the untouched base call.
	void Problem::get_residuals_by_elemental_assembly(oomph::DoubleVector &residuals)
	{
		bool serial_ranks = true;
#ifdef OOMPH_HAS_MPI
		if (Problem_has_been_distributed || (Communicator_pt && Communicator_pt->nproc() > 1)) serial_ranks = false;
#endif
		if (!serial_ranks || !parallel_assembly_possible())
		{
			oomph::Problem::get_residuals(residuals);
			return;
		}

		// Same distribution logic as the base implementation, so a caller that pre-built the vector
		// keeps its distribution.
		oomph::LinearAlgebraDistribution *dist_pt = 0;
		if (residuals.built())
			dist_pt = new oomph::LinearAlgebraDistribution(residuals.distribution_pt());
		else
			create_new_linear_algebra_distribution(dist_pt);
		residuals.build(dist_pt, 0.0);

		FrozenAssemblyRequest req;
		req.el_lo = 0;
		req.el_hi_plus_one = mesh_pt()->nelement();
		req.n_matrix = 0;
		req.n_vector = 1;
		req.n_map = 0;
		req.residual.push_back(residuals.values_pt());
		bool done = false;
		try
		{
			done = parallel_assemble_frozen(req);
		}
		catch (...)
		{
			delete dist_pt;
			throw;
		}
		delete dist_pt;
		if (!done)
		{
			// The plan declined (a value-dependent pattern, say). Redo it serially; the vector was only
			// zeroed, so nothing has to be undone.
			report_parallel_refusal("the element loop cannot be planned without a stable sparsity pattern "
									"(keep_structural_zeros is off, or the pattern depends on the dof values)");
			oomph::Problem::get_residuals(residuals);
		}
	}

#ifdef OOMPH_HAS_MPI
	// The distributed counterpart of try_parallel_frozen_assembly. DistributedFrozenSparsity::Mat
	// carries the same three arrays under the same names as FrozenSparsity, so only the targets and
	// the row map differ: values go into this rank's local_values, residuals into local_res indexed by
	// the row of the equation in sp.my_eqns.
	bool Problem::try_parallel_distributed_assembly(const DistributedFrozenSparsity &sp, unsigned n_matrix,
													unsigned n_vector, std::vector<std::vector<double>> &local_values,
													std::vector<std::vector<double>> &local_res, std::string &violation)
	{
		if (!parallel_assembly_possible()) return false;
		FrozenAssemblyRequest req;
		req.el_lo = sp.el_lo;
		req.el_hi_plus_one = sp.el_hi_plus_one;
		req.n_matrix = n_matrix;
		req.n_vector = n_vector;
		req.map_of_matrix.resize(n_matrix);
		for (unsigned m = 0; m < n_matrix; m++)
		{
			const DistributedFrozenSparsity::Mat &mt = sp.mats[m];
			if (mt.element_offset.size() < (size_t)mesh_pt()->nelement() + 1) return false;
			// No deduplication here: the distributed pattern is built per matrix, so two matrices never
			// share one Mat even when their patterns coincide.
			req.map_of_matrix[m] = m;
			req.element_offset.push_back(mt.element_offset.data());
			req.scatter_slot.push_back(mt.scatter_slot.data());
			req.scatter_source.push_back(mt.scatter_source.data());
			req.value.push_back(local_values[m].data());
		}
		req.n_map = n_matrix;
		for (unsigned v = 0; v < n_vector; v++) req.residual.push_back(local_res[v].data());
		req.res_row_map = sp.my_eqns.data();
		req.res_row_map_n = (unsigned)sp.my_eqns.size();
		req.verify = verify_frozen_sparsity;
		req.violation_out = &violation;
		return parallel_assemble_frozen(req);
	}
#endif

	bool Problem::parallel_assemble_frozen(const FrozenAssemblyRequest &req)
	{
#ifndef PYOOMPH_HAS_PARALLEL_ASSEMBLY
		(void)req;
		return false;
#else
		const int pslot = acquire_parallel_assembly_plan(req);
		if (pslot < 0) return false;
		const ParallelAssemblyPlan &plan = parallel_assembly_plans[pslot];
		const unsigned nchunk = plan.nchunk();
		if (!nchunk) return false;

		oomph::AssemblyHandler *const handler = this->assembly_handler_pt();
		const unsigned n_vector = req.n_vector, n_matrix = req.n_matrix;
		const int nt = (int)std::max(1u, num_threads);

		// PYOOMPH_REPORT_OMP_ASSEMBLY: where the time in a threaded assembly actually goes. The three
		// parts answer different questions - a slow pre-pass means the hanging sweep dominates, a
		// phase 2 comparable to phase 1 means the chunk does not fit in cache (see the budget above),
		// and a phase 1 that does not shrink with the thread count means the element kernel is
		// memory-bound rather than the loop being badly parallelised.
		static const bool report_phases = getenv("PYOOMPH_REPORT_OMP_ASSEMBLY") != NULL;
		double t_pre = 0.0, t_p1 = 0.0, t_p2 = 0.0, t0 = 0.0;
		// Per-thread BUSY time in phase 1, i.e. excluding the wait at the barrier. max/mean over the
		// threads is the load imbalance, and it is the only way to tell "the element kernel does not
		// scale" from "one thread did most of the work" - the two look identical in the phase totals.
		std::vector<double> busy(nt, 0.0);
		if (report_phases) t0 = pa_now();

		// Hanging values are pushed into the nodes ONCE, here, and the workers adopt the stamp.
		const unsigned long pass_id = prepare_hanging_values_for_parallel_assembly(plan);
		if (report_phases) { t_pre = pa_now() - t0; }

		// Scratch for one chunk, held on the Problem so that a Newton step does not reallocate it.
		// Never zeroed: an element's block is written in full by get_all_vectors_and_matrices() (which
		// initialises it), and the gather reads only positions inside a block written in this chunk.
		std::vector<std::vector<double>> &blocks = parallel_assembly_blocks;
		std::vector<std::vector<double>> &resbuf = parallel_assembly_resbuf;
		if (blocks.size() < n_matrix) blocks.resize(n_matrix);
		for (unsigned m = 0; m < n_matrix; m++)
			if (blocks[m].size() < (size_t)plan.max_block_size) blocks[m].resize((size_t)plan.max_block_size);
		if (resbuf.size() < n_vector) resbuf.resize(n_vector);
		for (unsigned v = 0; v < n_vector; v++)
			if (resbuf[v].size() < (size_t)plan.max_res_size) resbuf[v].resize((size_t)plan.max_res_size);

		std::atomic<bool> failed(false);
		std::string first_error;
		std::mutex error_mutex;
		auto record_error = [&](const std::string &what) {
			std::lock_guard<std::mutex> lock(error_mutex);
			if (first_error.empty()) first_error = what;
			failed.store(true);
		};
		// A verification failure under a collective caller: recorded, not thrown, and NOT a reason to
		// stop assembling - the exchange after the loop is what turns it into an error on every rank.
		auto record_violation = [&](const std::string &what) {
			std::lock_guard<std::mutex> lock(error_mutex);
			if (req.violation_out->empty()) *req.violation_out = what;
		};

		{
			// The workers call back into Python whenever an element uses a CustomMathExpression, and
			// they hold no GIL. Handing it over here lets the trampolines take it one at a time.
			GILReleaseScope gil;

#if defined(PYOOMPH_HAS_OPENMP)
#pragma omp parallel num_threads(nt)
			{
				AssemblyWorkerScope worker(pass_id);
				const int tid = omp_get_thread_num();
				const int nt_actual = omp_get_num_threads();
				oomph::Vector<oomph::Vector<double>> el_res(n_vector);
				oomph::Vector<oomph::DenseMatrix<double>> el_jac(n_matrix);

				for (unsigned c = 0; c < nchunk; c++)
				{
					const long chunk_len = plan.chunk_start[c + 1] - plan.chunk_start[c];
					// The dispatch granularity is derived from the chunk rather than fixed, and that
					// matters as soon as the elements are large: the chunk length comes from a budget on
					// nvar*nvar, so a domain with big elements gets a SHORT chunk, and a fixed grain of 8
					// would hand the whole of it to one or two threads while the rest idled. Aim for about
					// eight dispatches per thread - enough to even out elements of unequal cost without
					// paying for a shared counter per element.
					const int gran = (int)std::max(1L, std::min(8L, chunk_len / (8L * (long)nt_actual)));
					double tc = 0.0;
					if (report_phases && tid == 0) tc = pa_now();
					const double t_busy0 = (report_phases ? pa_now() : 0.0);
					// ---- phase 1: every element's block, in parallel, into its own slice
					// nowait plus the explicit barrier below rather than the implicit one: identical
					// semantics, but it leaves a place to read each thread's busy time before it waits.
#pragma omp for schedule(dynamic, gran) nowait
					for (long i = plan.chunk_start[c]; i < plan.chunk_start[c + 1]; i++)
					{
						if (failed.load(std::memory_order_relaxed)) continue; // cannot break out of an omp for
						try
						{
							oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(plan.elements[i]);
							const unsigned nv = (unsigned)plan.nvar[i];
							for (unsigned v = 0; v < n_vector; v++) el_res[v].resize(nv);
							for (unsigned m = 0; m < n_matrix; m++) el_jac[m].resize(nv);
							// With no matrix requested this is a residual-only sweep, and the DEFAULT
							// AssemblyHandler::get_all_vectors_and_matrices dereferences matrix[0]
							// unconditionally. ParallelResidualsHandler, the other handler that reaches this
							// with n_matrix==0, forwards to get_residuals anyway, so the two agree.
							if (n_matrix)
								handler->get_all_vectors_and_matrices(elem_pt, el_res, el_jac);
							else
								handler->get_residuals(elem_pt, el_res[0]);
							const size_t nblock = (size_t)nv * nv;
							for (unsigned m = 0; m < n_matrix; m++)
							{
								const double *flat = &el_jac[m](0, 0);
								std::copy(flat, flat + nblock, blocks[m].begin() + plan.block_base[i]);
								if (req.verify)
								{
									// Same guard as the serial path: the compact scatter cannot notice an entry
									// that fell outside the pattern, so count instead of trusting.
									const unsigned mm = req.map_of_matrix[m];
									(void)mm;
									const int *eoff = req.element_offset[m];
									const int *src = req.scatter_source[m];
									const int e = plan.elements[i];
									unsigned all = 0, taken = 0;
									for (size_t q = 0; q < nblock; q++) all += (flat[q] != 0.0);
									for (int k = eoff[e]; k < eoff[e + 1]; k++) taken += (flat[src[k]] != 0.0);
									if (all > taken)
									{
										// Name the offending positions, as the serial path does: which part of
										// the block escaped is what distinguishes an over-tight mask from a
										// genuinely wider matrix, and the two need opposite fixes.
										std::string where;
										unsigned shown = 0;
										std::vector<bool> covered(nblock, false);
										for (int k = eoff[e]; k < eoff[e + 1]; k++) covered[src[k]] = true;
										for (size_t q = 0; q < nblock && shown < 6; q++)
										{
											if (flat[q] == 0.0 || covered[q]) continue;
											where += (shown ? ", " : "") + std::string("(") +
													 std::to_string(q / nv) + "," + std::to_string(q % nv) +
													 ")=" + std::to_string(flat[q]);
											shown++;
										}
										const std::string msg =
											"A Jacobian entry appeared outside the symbolic sparsity pattern "
											"(matrix " + std::to_string(m) + ", element " + std::to_string(e) +
											", nvar " + std::to_string(nv) + "). Positions missing from the "
											"pattern: " + where + ". Refusing rather than truncating the matrix. "
											"Workaround: problem.use_frozen_sparsity=False.";
										if (req.violation_out) record_violation(msg); else throw_runtime_error(msg);
									}
								}
							}
							for (unsigned v = 0; v < n_vector; v++)
								std::copy(el_res[v].begin(), el_res[v].begin() + nv,
										  resbuf[v].begin() + plan.res_base[i]);
						}
						catch (const std::exception &err) { record_error(err.what()); }
						catch (...) { record_error("unknown exception during threaded element assembly"); }
					}
					if (report_phases) busy[tid] += pa_now() - t_busy0;
#pragma omp barrier
					// every block of this chunk is now written
					if (report_phases && tid == 0) { t_p1 += pa_now() - tc; tc = pa_now(); }

					// ---- phase 2: each thread owns a range of TARGET SLOTS, so nothing is shared
					if (!failed.load(std::memory_order_relaxed))
					{
						for (unsigned m = 0; m < n_matrix; m++)
						{
							const ParallelAssemblyPlan::Gather &gt = plan.mat_gather[req.map_of_matrix[m]];
							const long lo = gt.chunk_gather_start[c], hi = gt.chunk_gather_start[c + 1];
							long a, b;
							gather_range_for_thread(gt.slot.data(), lo, hi, tid, nt_actual, a, b);
							const int *gslot = gt.slot.data();
							const int *gsrc = gt.src.data();
							const double *src_block = blocks[m].data();
							double *val = req.value[m];
							long k = a;
							while (k < b)
							{
								const int s = gslot[k];
								// Seeded from the current value and summed left to right, which is the order
								// the serial loop's val[slot] += ... produces.
								double acc = val[s];
								while (k < b && gslot[k] == s) { acc += src_block[gsrc[k]]; k++; }
								val[s] = acc;
							}
						}
						{
							const ParallelAssemblyPlan::Gather &gt = plan.res_gather;
							const long lo = gt.chunk_gather_start[c], hi = gt.chunk_gather_start[c + 1];
							long a, b;
							gather_range_for_thread(gt.slot.data(), lo, hi, tid, nt_actual, a, b);
							const int *gslot = gt.slot.data();
							const int *gsrc = gt.src.data();
							for (unsigned v = 0; v < n_vector; v++)
							{
								const double *src_res = resbuf[v].data();
								double *out = req.residual[v];
								long k = a;
								while (k < b)
								{
									const int s = gslot[k];
									double acc = out[s];
									while (k < b && gslot[k] == s) { acc += src_res[gsrc[k]]; k++; }
									out[s] = acc;
								}
							}
						}
					}
#pragma omp barrier
					if (report_phases && tid == 0) t_p2 += pa_now() - tc;
				}
			}
#else  // PYOOMPH_HAS_GCD
			// The same two-phase chunk sweep as the OpenMP region above, but dispatch_apply IS the
			// parallel-for and its synchronous completion IS the barrier - no OpenMP runtime is linked,
			// so nothing collides with MKL's libiomp5 on an Intel mac. One libdispatch block per intended
			// thread; phase 1 hands out elements through an atomic counter, reproducing
			// schedule(dynamic, gran). The gather in phase 2 is byte-for-byte the OpenMP one.
			dispatch_queue_t pa_queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
			for (unsigned c = 0; c < nchunk; c++)
			{
				const long cstart = plan.chunk_start[c];
				const long cend = plan.chunk_start[c + 1];
				const long chunk_len = cend - cstart;
				const int gran = (int)std::max(1L, std::min(8L, chunk_len / (8L * (long)nt)));
				double tc = (report_phases ? pa_now() : 0.0);

				std::atomic<long> next(cstart);
				auto phase1 = [&](size_t widx)
				{
					const double t_busy0 = (report_phases ? pa_now() : 0.0);
					try
					{
						AssemblyWorkerScope worker(pass_id);
						oomph::Vector<oomph::Vector<double>> el_res(n_vector);
						oomph::Vector<oomph::DenseMatrix<double>> el_jac(n_matrix);
						for (;;)
						{
							if (failed.load(std::memory_order_relaxed)) break;
							const long i0 = next.fetch_add((long)gran, std::memory_order_relaxed);
							if (i0 >= cend) break;
							const long i1 = std::min(i0 + (long)gran, cend);
							for (long i = i0; i < i1; i++)
							{
								try
								{
									oomph::GeneralisedElement *elem_pt = mesh_pt()->element_pt(plan.elements[i]);
									const unsigned nv = (unsigned)plan.nvar[i];
									for (unsigned v = 0; v < n_vector; v++) el_res[v].resize(nv);
									for (unsigned m = 0; m < n_matrix; m++) el_jac[m].resize(nv);
									if (n_matrix)
										handler->get_all_vectors_and_matrices(elem_pt, el_res, el_jac);
									else
										handler->get_residuals(elem_pt, el_res[0]);
									const size_t nblock = (size_t)nv * nv;
									for (unsigned m = 0; m < n_matrix; m++)
									{
										const double *flat = &el_jac[m](0, 0);
										std::copy(flat, flat + nblock, blocks[m].begin() + plan.block_base[i]);
										if (req.verify)
										{
											const int *eoff = req.element_offset[m];
											const int *src = req.scatter_source[m];
											const int e = plan.elements[i];
											unsigned all = 0, taken = 0;
											for (size_t q = 0; q < nblock; q++) all += (flat[q] != 0.0);
											for (int k = eoff[e]; k < eoff[e + 1]; k++) taken += (flat[src[k]] != 0.0);
											if (all > taken)
											{
												std::string where;
												unsigned shown = 0;
												std::vector<bool> covered(nblock, false);
												for (int k = eoff[e]; k < eoff[e + 1]; k++) covered[src[k]] = true;
												for (size_t q = 0; q < nblock && shown < 6; q++)
												{
													if (flat[q] == 0.0 || covered[q]) continue;
													where += (shown ? ", " : "") + std::string("(") +
															 std::to_string(q / nv) + "," + std::to_string(q % nv) +
															 ")=" + std::to_string(flat[q]);
													shown++;
												}
												const std::string msg =
													"A Jacobian entry appeared outside the symbolic sparsity pattern "
													"(matrix " + std::to_string(m) + ", element " + std::to_string(e) +
													", nvar " + std::to_string(nv) + "). Positions missing from the "
													"pattern: " + where + ". Refusing rather than truncating the matrix. "
													"Workaround: problem.use_frozen_sparsity=False.";
												if (req.violation_out) record_violation(msg); else throw_runtime_error(msg);
											}
										}
									}
									for (unsigned v = 0; v < n_vector; v++)
										std::copy(el_res[v].begin(), el_res[v].begin() + nv,
												  resbuf[v].begin() + plan.res_base[i]);
								}
								catch (const std::exception &err) { record_error(err.what()); }
								catch (...) { record_error("unknown exception during threaded element assembly"); }
							}
						}
					}
					catch (const std::exception &err) { record_error(err.what()); }
					catch (...) { record_error("unknown exception during threaded element assembly"); }
					if (report_phases) busy[widx] += pa_now() - t_busy0;
				};
				dispatch_apply_f((size_t)nt, pa_queue, &phase1, &pa_apply_trampoline<decltype(phase1)>);
				if (report_phases) { t_p1 += pa_now() - tc; tc = pa_now(); }

				if (!failed.load(std::memory_order_relaxed))
				{
					auto phase2 = [&](size_t widx)
					{
						const int tid = (int)widx;
						const int nt_actual = nt;
						try
						{
							for (unsigned m = 0; m < n_matrix; m++)
							{
								const ParallelAssemblyPlan::Gather &gt = plan.mat_gather[req.map_of_matrix[m]];
								const long lo = gt.chunk_gather_start[c], hi = gt.chunk_gather_start[c + 1];
								long a, b;
								gather_range_for_thread(gt.slot.data(), lo, hi, tid, nt_actual, a, b);
								const int *gslot = gt.slot.data();
								const int *gsrc = gt.src.data();
								const double *src_block = blocks[m].data();
								double *val = req.value[m];
								long k = a;
								while (k < b)
								{
									const int s = gslot[k];
									double acc = val[s];
									while (k < b && gslot[k] == s) { acc += src_block[gsrc[k]]; k++; }
									val[s] = acc;
								}
							}
							{
								const ParallelAssemblyPlan::Gather &gt = plan.res_gather;
								const long lo = gt.chunk_gather_start[c], hi = gt.chunk_gather_start[c + 1];
								long a, b;
								gather_range_for_thread(gt.slot.data(), lo, hi, tid, nt_actual, a, b);
								const int *gslot = gt.slot.data();
								const int *gsrc = gt.src.data();
								for (unsigned v = 0; v < n_vector; v++)
								{
									const double *src_res = resbuf[v].data();
									double *out = req.residual[v];
									long k = a;
									while (k < b)
									{
										const int s = gslot[k];
										double acc = out[s];
										while (k < b && gslot[k] == s) { acc += src_res[gsrc[k]]; k++; }
										out[s] = acc;
									}
								}
							}
						}
						catch (...) { record_error("unknown exception during threaded gather"); }
					};
					dispatch_apply_f((size_t)nt, pa_queue, &phase2, &pa_apply_trampoline<decltype(phase2)>);
				}
				if (report_phases) t_p2 += pa_now() - tc;
			}
#endif
		}
		if (report_phases)
		{
			double bmax = 0.0, bsum = 0.0;
			for (int t = 0; t < nt; t++) { bmax = std::max(bmax, busy[t]); bsum += busy[t]; }
			const double bmean = (nt ? bsum / nt : 0.0);
			oomph::oomph_info << "pyoomph OMP assembly: " << nchunk << " chunks, " << nt << " threads, total "
							  << (pa_now() - t0) * 1000 << " ms = hang pre-pass " << t_pre * 1000
							  << " + phase 1 " << t_p1 * 1000 << " + phase 2 " << t_p2 * 1000
							  << " ms; phase-1 imbalance max/mean " << (bmean > 0 ? bmax / bmean : 1.0)
							  << " (busiest " << bmax * 1000 << " ms, mean " << bmean * 1000 << " ms)"
							  << std::endl;
		}

		// Rethrown here rather than from the worker: an exception must not leave an OpenMP region. The
		// prefix is what tells the reader that the element it names was assembled off the main thread,
		// and hence that --omp 1 is the first thing to try when the message is puzzling.
		if (failed.load()) throw_runtime_error("during threaded element assembly: " + first_error);
		parallel_assemblies_done++;
		return true;
#endif
	}

}
