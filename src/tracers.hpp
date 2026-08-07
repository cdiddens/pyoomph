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

// Passive tracer particles advected through a (possibly moving and deforming) mesh.
//
// ---------------------------------------------------------------------------------------------
// The formulation, because it is not the obvious one and the difference is the whole point
// ---------------------------------------------------------------------------------------------
//
// A particle sits at x_p(t) = X(s(t), t), where X is the mesh map and J = dX/ds. In general
//
//     ds/dt = J^+ (v - dX/dt|_s)                                     (J^+ = pseudo-inverse)
//
// and two cases fall out of that, which is why one piece of code covers both:
//
//  * BULK (codimension 0). J is square, so J J^+ = I and the whole thing collapses to
//    dx_p/dt = v. The mesh velocity cancels ANALYTICALLY. So this class integrates the physical
//    position and uses the local coordinate s purely as a chart for evaluating the field. Nothing
//    ever computes a mesh velocity, and a particle in a moving mesh with v = 0 does not move
//    because every stage derivative is identically zero - not because two terms cancel to within
//    rounding. The old implementation integrated s instead and subtracted a mesh_velocity() term
//    that was neither blended over the sub-step nor even emitted unless the mesh had position
//    dofs, so a mesh moved by macro elements dragged its tracers along with it.
//
//  * INTERFACE (codimension 1). J is (d-1) x d, P = J^+ J is the orthogonal projector onto the
//    tangent space, and
//
//        dx_p/dt = P v + (I - P) dX/dt|_s
//
//    i.e. the tangential part of the advection field plus the normal part of the interface's own
//    motion. That is exactly "advected tangentially, co-moving normally", with no explicit
//    normal/tangent algebra anywhere. After each sub-step the position is re-anchored onto the
//    interface by the same least-squares inversion, which pins the normal offset at machine zero.
//
// The configuration WITHIN a timestep is a Lagrange interpolation in time of the nodal positions
// between the stored history levels (see TracerTimeConfig). Since the shape functions do not
// depend on time, J and dX/dt of that interpolant are exact rather than approximated. The solver
// never defines intermediate configurations, so this is the best available statement of where the
// mesh was in the middle of a step - and it caps the achievable accuracy at the interpolation
// order regardless of how tight the integrator tolerance is set.
//
// ---------------------------------------------------------------------------------------------
// Bookkeeping
// ---------------------------------------------------------------------------------------------
//
// The PHYSICAL POSITION is the authoritative state. `elem` and `s` are derived, and are dropped
// whenever the mesh announces a new topology generation. Every particle carries a globally unique,
// never-recycled TracerId, which is what makes state files partition-independent and gathers
// deterministic under MPI.

#pragma once
#include <map>
#include <string>
#include <vector>

#include "mesh.hpp"
#include "refdomain.hpp"

namespace pyoomph
{
  class BulkElementBase;
  class MeshPointLocator;
  class TracerCollection;

  typedef unsigned long long TracerId;

  // Lagrange interpolation in time of the nodal positions, over one timestep.
  //
  // tau runs over [0,1] within the step: tau = 0 is history level 1 (where the particles were at
  // the end of the previous step) and tau = 1 is history level 0 (the configuration the Newton
  // solve just produced). Level 2, when usable, sits at tau = -dtprev/dt.
  class TracerTimeConfig
  {
  public:
    unsigned nlevel = 1;
    double dt = 0.0;                        // t(0) - t(1), the step being advected over
    double t_current = 0.0;                 // t(0), used to stamp which step a particle has finished
    double tau_of_level[3] = {1.0, 0.0, 0.0};
    double w[3] = {1.0, 0.0, 0.0};          // Lagrange weights at the current tau
    double dwdtau[3] = {0.0, 0.0, 0.0};     // and their derivatives w.r.t. tau

    // requested_order < 0 means "the best the stored history supports". The order is demoted
    // silently when it has to be: on the first step t(1) == t(2) and the quadratic basis is
    // singular, and a code that only registered two history levels cannot supply a third.
    static TracerTimeConfig from_mesh(Mesh *m, int requested_order, unsigned nlevel_available);
    void set_tau(double tau);
  };

  // One particle. Not a POD and not what crosses an MPI boundary - TracerCollection packs and
  // unpacks these into a flat, per-collection-uniform stride for that.
  class TracerParticle
  {
    friend class TracerCollection;

  protected:
    TracerId id = 0;
    int tag = 0;
    std::vector<double> x;       // authoritative physical position, nodal_dim entries
    std::vector<double> s;       // local coordinate in `elem`; meaningless when elem is NULL
    std::vector<double> payload; // path-integrated user scalars

    BulkElementBase *elem = nullptr; // derived, never serialised, dropped on a generation bump
    RefDomain refdomain = RefDomain::Unknown;

    double timefrac = 0.0;   // fraction of the current step already advected; only ever non-zero
                             // between MPI migration rounds
    double next_h = 0.25;    // sub-step size the controller ended the last step on
    // Time level 0 of the step this particle last completed. A particle handed over from another
    // domain mid-step has already finished the current step by the time the receiving domain's own
    // hook runs, and without this it would be advected a second time.
    double completed_step_time = -1e300;

    // Rolling position history for trails, as (t, x[0..d-1]) samples, oldest first once unwrapped.
    // Stored as a ring so that pruning is O(1) and the buffer never grows without bound.
    std::vector<double> hist;
    unsigned hist_n = 0, hist_head = 0;

  public:
    virtual ~TracerParticle() {}
    TracerId get_id() const { return id; }
    int get_tag() const { return tag; }
    const std::vector<double> &get_position() const { return x; }
  };

  // Where particles crossing a given mesh boundary should be handed over to.
  class TracerTransferInterfaceInfo
  {
  public:
    TracerCollection *other_collection = nullptr;
  };

  class TracerCollection
  {
  protected:
    Mesh *mesh = nullptr;
    std::string tracer_name;

    unsigned nodal_dim = 0, elem_dim = 0; // codimension = nodal_dim - elem_dim, must be 0 or 1
    unsigned n_payload = 0;

    // One generated-code entry per history level, -1 where the level was not registered.
    int code_index[3] = {-1, -1, -1};
    // n_payload entries per level, flattened as [level*n_payload + p].
    std::vector<int> payload_code_index;

    std::vector<TracerParticle *> tracers; // owned; dense, no free-slot recycling
    std::map<unsigned, TracerTransferInterfaceInfo> transfer_interfaces;

    // Periodic images: a particle that has run out of the mesh is offered its position plus each of
    // these shifts, and is taken back in at the first one that lands inside.
    //
    // Not keyed on the boundary it left through, on purpose. A shifted position that lands inside
    // the mesh IS the periodic image - a domain in which two different shifts both land inside
    // would have to be larger than its own period. So there is nothing to detect, which also makes
    // a particle leaving through a corner where two periodic directions meet fall out for free.
    std::vector<std::vector<double>> periodic_wraps;

    // Particles whose periodic image lies on nobody-knows-which process. Held between the local
    // pass and the collective reinjection round of advect_all, which owns them until then.
    std::vector<TracerParticle *> pending_reinject;

    TracerId next_id = 1;

    // Set by advect_one when the particle it was given was handed to another domain's collection,
    // which then owns it. The caller must not keep it.
    bool transferred_away = false;

    // Point locators over this mesh, one per time level. Cached because building one walks every
    // element, indexes its nodes into a kd-tree and fits an affine inverse per element.
    //
    // A locator freezes the NODAL POSITIONS it was built from, so caching it on the topology
    // generation alone was wrong on any moving mesh: the geometry moves without the topology
    // changing, and a locator from an earlier configuration then reports points near the moved
    // boundary as outside the mesh. `geometry_stale` is set at the points where the configuration
    // may have moved since the last build - see mark_geometry_stale() - and forces the rebuild.
    //
    // These INCLUDE halo elements. A particle is deliberately allowed to advect through the halo
    // layer, whose nodal positions and dof values are synchronised copies of the owner's, so that
    // it reaches the end of the step before ownership is reconsidered. Migrating mid-step would
    // otherwise need the receiving rank to place the particle in the time-interpolated
    // configuration, which no locator is built for. Ownership is then decided by is_halo() on the
    // element the particle ended in, at tau = 1, where the level-0 locator is exactly valid.
    MeshPointLocator *locator[2] = {nullptr, nullptr};
    unsigned long locator_generation = 0;
    bool has_locator_generation = false;
    bool geometry_stale = true;

    int mpi_nproc() const;
    int mpi_rank() const;
    bool is_distributed() const;
    // Hand every particle whose element is a halo to the rank that owns that element, and take in
    // the ones sent here. Returns the number of particles that moved, summed over all ranks.
    unsigned exchange_migrants();
    // Flatten / rebuild one particle for the wire and for state files.
    unsigned record_stride() const;
    void pack(const TracerParticle *p, double *out) const;
    TracerParticle *unpack(const double *in, TracerId id, int tag);

    // Per-step counters, reported by step_statistics().
    unsigned long stat_substeps = 0, stat_rejected = 0, stat_walks = 0;
    unsigned stat_global_locates = 0;
    unsigned stat_lost = 0, stat_migrated = 0, stat_transferred = 0;
    unsigned stat_wrapped = 0, stat_reinjected = 0;

    // A locator whose GEOMETRY is valid for the mesh's current configuration at `time_level`.
    MeshPointLocator *get_locator(unsigned time_level);
    // A locator to ask for node-element adjacency only. That answer is pure incidence, so any
    // locator of the current topology generation gives it - which is what keeps the element walk
    // from forcing a geometric rebuild on every timestep of a moving mesh.
    MeshPointLocator *get_adjacency_locator();
    void drop_locators();
    // Announce that the nodal positions may have moved since the locators were built.
    void mark_geometry_stale() { geometry_stale = true; }
    // True when the mesh has announced a new generation since the locators were built.
    bool generation_changed() const;

    void resolve_code_indices();

    // Invert (bulk) or least-squares project (interface) X(s, tau) = target, starting from p->s in
    // p->elem, walking to a neighbouring element if it leaves the reference domain. Returns false
    // if the point could not be placed in any element reachable by the walk - which the caller is
    // expected to treat as "the sub-step was too big", not as "the particle is gone".
    bool place_at(TracerParticle *p, const TracerTimeConfig &cfg, const double *target, double *x_on_elem);
    // Same, but starting from scratch through the global locator. Only valid at tau = 1, where the
    // locator's time level matches the configuration.
    bool place_globally(TracerParticle *p, unsigned time_level);

    // dy/dtau at (tau, y) for one particle, leaving p->elem/p->s at the located position.
    // Returns false if y could not be placed.
    bool eval_derivative(TracerParticle *p, TracerTimeConfig &cfg, double tau, const double *y,
                         double *dydtau, double *dpdtau);

    // Advect one particle from its current timefrac to 1. Returns false if it left the mesh.
    // `depth` bounds the chain of domain-to-domain handovers within one timestep.
    bool advect_one(TracerParticle *p, TracerTimeConfig &cfg, unsigned depth = 0);

    // Try every registered periodic shift and place `p` at the first image that lands in a non-halo
    // element of this process's part of the mesh. Leaves p->x untouched and returns false if none
    // does. Drops the position history on success: a trail is a path through the plotted
    // coordinates, and a wrapped one is not continuous there, so keeping the samples from before
    // the jump would draw a line straight back across the domain.
    bool place_periodic_image(TracerParticle *p);

    enum class WrapResult
    {
      NotPlaced,             // no shift put it anywhere, and nothing else will take it
      PlacedHere,            // it is back in this process's mesh, ready to finish its step
      ParkedForReinjection   // pending_reinject owns it now; the caller must forget it
    };
    // place_periodic_image, plus the decision of what to do when no image was local.
    WrapResult wrap_position(TracerParticle *p);

    // COLLECTIVE. Offer every process's parked particles to all of them, so that the one holding
    // the periodic image takes it. The migration exchange cannot do this: it routes a particle to
    // the owner of the HALO element it ended in, and the far end of a periodic domain is not a halo
    // of the near end - it is usually not in the sending process's mesh at all.
    unsigned exchange_reinjections();

    // Take over a particle that has just left another domain through a shared interface, place it
    // here and finish its timestep. Returns false (having NOT taken ownership) if the position does
    // not lie in this mesh.
    bool adopt(TracerParticle *p, unsigned depth);
    // Offer a particle that cannot continue here to the collections registered on the boundaries it
    // may have crossed. Returns the collection that took it, or null.
    TracerCollection *try_transfer(TracerParticle *p, unsigned depth);

    TracerParticle *make_and_place(const std::vector<double> &pos, int tag,
                                   const std::vector<double> &payload_init);

    // The two directions of the rolling history ring: unwrap it into chronological (t, x...)
    // samples, and rebuild it from such samples, keeping the newest ones that fit the capacity.
    std::vector<double> history_of(const TracerParticle *p) const;
    void set_history(TracerParticle *p, const double *samples, unsigned count);

    // Gather `local` (ncol doubles per local particle) from every process and return it sorted by
    // particle id, so the answer is the same everywhere and independent of the partitioning. Also
    // returns the sorted ids if asked.
    std::vector<double> gather_rows(const std::vector<double> &local, unsigned ncol,
                                    std::vector<long long> *ids_out = nullptr) const;

  public:
    // Tunables. Deliberately public: they are plain knobs with no invariants between them.
    double rtol = 1e-8;
    double atol = 1e-10;
    double history_window = 0.0;      // 0 disables the position history entirely
    unsigned history_capacity = 64;
    int time_interpolation_order = -1; // -1 = as good as the stored history allows
    int fixed_substeps = 0;            // > 0 forces uniform sub-steps, for order-of-convergence tests
    unsigned long max_substeps = 1000000;
    unsigned max_migration_rounds = 64;
    // How many times one particle may be wrapped within a single timestep. More than one only
    // happens if it crosses the whole periodic length in a step, which is a bound on the timestep
    // rather than something to accommodate; this is here so that a degenerate wrap cannot spin.
    unsigned max_periodic_wraps = 8;

    TracerCollection(const std::string &name) : tracer_name(name) {}
    virtual ~TracerCollection();

    virtual void set_mesh(Mesh *m);
    Mesh *get_mesh() const { return mesh; }
    unsigned get_coordinate_dimension() const { return nodal_dim; }
    unsigned get_codimension() const { return nodal_dim - elem_dim; }
    void set_num_payloads(unsigned n);
    unsigned get_num_payloads() const { return n_payload; }

    virtual void clear();
    // Adds one particle on THIS process only. Returns 0 and adds nothing if the point does not lie
    // in a non-halo element of this process's part of the mesh - which under MPI is the normal
    // outcome on all but one rank, so a caller that wants one particle per point must use
    // add_tracers_collective instead.
    TracerId add_tracer(const std::vector<double> &pos, int tag, const std::vector<double> &payload_init);

    // COLLECTIVE. Every process must pass the same candidate list. Each candidate ends up on
    // exactly one process - the lowest-numbered one that holds it in a non-halo element - and gets
    // an identity derived from its index in the list, so the resulting set of particles and their
    // ids do not depend on how the mesh is partitioned.
    //
    // Returns how many candidates lay outside the mesh on every process.
    // `ids`, when non-empty, supplies the identity of each candidate instead of deriving it from
    // the index. That is what restoring a state file needs: the identities are part of the file.
    unsigned add_tracers_collective(const std::vector<double> &pos, const std::vector<int> &tags,
                                    const std::vector<double> &payload_init,
                                    const std::vector<long long> &ids = std::vector<long long>());

    bool remove_tracer(TracerId id);
    unsigned nlocal() const { return (unsigned)tracers.size(); }
    // COLLECTIVE. Number of particles over all processes.
    unsigned long nglobal() const;

    // Local views, in the order the particles are stored (which is creation order until removals).
    std::vector<double> get_positions();
    std::vector<long long> get_ids() const;
    std::vector<int> get_tags() const;
    std::vector<double> get_payloads() const;
    std::vector<double> get_history_of(TracerId id) const;

    // COLLECTIVE. All processes' particles, id-sorted, identical on every process. This is the view
    // that does not depend on the partitioning, and the one state files and plots should use.
    std::vector<double> gather_positions() const;
    std::vector<long long> gather_ids() const;
    std::vector<int> gather_tags() const;
    std::vector<double> gather_payloads() const;

    // Re-derive every particle's element and local coordinate from its stored position, in the
    // configuration at `time_level`. Drops particles that no longer lie in the mesh.
    virtual void relocate_all(unsigned time_level);
    // Advect every particle through one accepted timestep.
    virtual void advect_all();

    std::string step_statistics() const;
    // How many particles had to be located from scratch during the last advection, i.e. how many
    // were holding an element pointer that the mesh had invalidated. Zero on a step where nothing
    // adapted or remeshed; equal to the particle count on a step where something did. Exposed
    // because "did the collection notice the adaptation" is otherwise invisible: a stale pointer
    // into a refined element's still-alive parent keeps producing plausible answers, and one into
    // an unrefined element's deleted son is undefined behaviour that may not crash.
    unsigned get_relocations_last_step() const { return stat_global_locates; }
    // The two halves of periodic re-injection during the last advection: particles whose image was
    // in this process's own mesh, and particles taken over from another process because the image
    // was not. Disjoint, so the number of wraps over the whole collection is the sum of both over
    // all processes. The second is zero serially, and a non-zero value is the only visible proof
    // that the collective round did the work - the positions cannot show it.
    unsigned get_wraps_last_step() const { return stat_wrapped; }
    unsigned get_reinjections_last_step() const { return stat_reinjected; }

    // `with_history` also serialises the rolling position history the trail plots read; it costs
    // the samples themselves in the file and makes `tagarr` three entries per particle instead of
    // two, so the caller has to keep it consistent with the state file format version.
    virtual void _save_state(std::vector<double> &posarr, std::vector<long long> &tagarr,
                             bool with_history = true);
    virtual void _load_state(const std::vector<double> &posarr, const std::vector<long long> &tagarr,
                             bool with_history = true);

    virtual void set_transfer_interface(unsigned boundary_index, TracerCollection *opp);

    // Declare that a particle leaving the mesh is the same particle re-entering it at its position
    // plus `shift` - i.e. that the domain is periodic by that vector. Registering the same shift
    // twice is a no-op, so attaching the boundary condition to both ends of a periodic pair, which
    // is the natural thing to write, costs nothing.
    virtual void add_periodic_wrap(const std::vector<double> &shift);
    virtual void clear_periodic_wraps();
    const std::vector<std::vector<double>> &get_periodic_wraps() const { return periodic_wraps; }
  };

}
