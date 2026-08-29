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

#include "refinement_coupling.hpp"
#include "elements.hpp"
#include "exception.hpp"
#include "mesh.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>

namespace pyoomph
{
  const double INTERFACE_COUPLING_KEY_SCALE = 1e8;

  namespace
  {
    // Can this pair be matched topologically? Both sides must carry a complete set of node identities,
    // and the connection must be coincident: a periodic/translated pair relates two DIFFERENT template
    // facets, which no topological identity can bridge, so those keep the position keys and their offset.
    bool pair_is_topological(const CoupledInterfacePair &p)
    {
      if (topological_interface_keys_disabled()) return false;
      for (double o : p.offset)
        if (o != 0.0) return false;
      if (!p.meshA || !p.meshB) return false;
      // Assign here rather than trusting a flag set at the last adaptation. Several paths create nodes
      // without going through adapt_finalise() -- refine_uniformly(), and the repair loop's own
      // refine_selected_elements() between rounds -- and a stale "complete" would then hand an unset id
      // to node_key(). Idempotent, and a no-op once every node has one.
      p.meshA->assign_interface_topological_ids();
      p.meshB->assign_interface_topological_ids();
      return p.meshA->has_complete_interface_topological_ids() &&
             p.meshB->has_complete_interface_topological_ids();
    }

    // The identity of an interface node: topological where available, quantised position otherwise.
    //
    // The position was the only key here until the space matrix was swept. It rests on the two sides'
    // interface VERTICES coinciding, and they do not: a C2 domain's interface is a quadratic curve
    // through three nodes while a C1 domain's is the chord between two of them, so a refinement promotes
    // an off-chord midside node to a vertex on one side and creates a chord midpoint on the other. The
    // two sides then refine the SAME facet and disagree about where its midpoint is, which this file
    // used to report as "the coarser side could not be refined any further".
    // See dev_docs/interface_refinement_coupling.md section 14.3 and pyoomph::Node::interface_topological_id.
    InterfacePosKey node_key(oomph::Node *n, const std::vector<double> &offset, bool topological)
    {
      if (topological)
      {
        pyoomph::Node *pn = static_cast<pyoomph::Node *>(n);
        // Guaranteed by pair_is_topological (both meshes report a complete set), so an unset id here
        // would mean a node reached the interface from outside those meshes.
        if (!pn->has_interface_topological_id()) throw_runtime_error("Interface node without a topological identity");
        const std::array<unsigned long long, 2> &id = pn->get_interface_topological_id();
        return InterfacePosKey{(long long)id[0], (long long)id[1], 0};
      }
      InterfacePosKey k = {0, 0, 0};
      const unsigned nd = n->ndim();
      for (unsigned d = 0; d < 3; d++)
      {
        double x = (d < nd ? n->x(d) : 0.0);
        if (d < offset.size()) x += offset[d];
        k[d] = (long long)std::llround(x * INTERFACE_COUPLING_KEY_SCALE);
      }
      return k;
    }

    // Mesh::get_boundary_index throws when the name is absent, but an absent side is a legitimate
    // state here (only dummy equations were created for it), so look it up without throwing.
    int find_boundary_index(Mesh *m, const std::string &bname)
    {
      const std::vector<std::string> names = m->get_boundary_names();
      for (unsigned i = 0; i < names.size(); i++)
        if (names[i] == bname) return (int)i;
      return -1;
    }

    // The communicator to reduce over. Both meshes live in the same Problem and hence the same
    // communicator; take it from whichever side is distributed. NULL means "purely serial, no
    // reduction needed" -- which is also the case in an MPI build that never called distribute().
    oomph::OomphCommunicator *coupling_communicator(const std::vector<CoupledInterfacePair> &pairs)
    {
      for (const CoupledInterfacePair &p : pairs)
      {
        oomph::Mesh *sides[2] = {dynamic_cast<oomph::Mesh *>(p.meshA), dynamic_cast<oomph::Mesh *>(p.meshB)};
        for (int s = 0; s < 2; s++)
          if (sides[s] && sides[s]->is_mesh_distributed() && sides[s]->communicator_pt() &&
              sides[s]->communicator_pt()->nproc() > 1)
            return sides[s]->communicator_pt();
      }
      return NULL;
    }

    // Replace the facet/vertex sets of `side` by their union across all processes. Without this a rank
    // would judge its own facets against only the part of the interface it happens to hold -- and the
    // two domains are partitioned INDEPENDENTLY (they share no nodes, so the partitioner sees two
    // disconnected components), so a rank routinely holds one side of a facet pair and not the other.
    void allgather_side(InterfaceSideFacets &side, oomph::OomphCommunicator *comm_pt)
    {
#ifdef OOMPH_HAS_MPI
      if (!comm_pt || comm_pt->nproc() < 2) return;
      MPI_Comm mc = comm_pt->mpi_comm();
      const int nproc = comm_pt->nproc();

      // Flatten as [nkeys, k0x,k0y,k0z, k1x,...] per facet.
      std::vector<long long> mine;
      for (const auto &f : side.local)
      {
        mine.push_back((long long)f.second.size());
        for (const InterfacePosKey &k : f.second)
        {
          mine.push_back(k[0]);
          mine.push_back(k[1]);
          mine.push_back(k[2]);
        }
      }
      int mycount = (int)mine.size();
      std::vector<int> counts(nproc, 0), displs(nproc, 0);
      MPI_Allgather(&mycount, 1, MPI_INT, &counts[0], 1, MPI_INT, mc);
      int total = 0;
      for (int i = 0; i < nproc; i++)
      {
        displs[i] = total;
        total += counts[i];
      }
      std::vector<long long> all((size_t)std::max(total, 1));
      MPI_Allgatherv(mycount ? &mine[0] : NULL, mycount, MPI_LONG_LONG,
                     &all[0], &counts[0], &displs[0], MPI_LONG_LONG, mc);

      side.facets.clear();
      side.vertices.clear();
      int i = 0;
      while (i < total)
      {
        const int n = (int)all[i++];
        if (n <= 0 || i + 3 * n > total) break; // malformed; cannot happen, but do not run off the end
        std::vector<InterfacePosKey> keys;
        keys.reserve(n);
        for (int j = 0; j < n; j++)
        {
          InterfacePosKey k = {all[i], all[i + 1], all[i + 2]};
          i += 3;
          keys.push_back(k);
          side.vertices.insert(k);
        }
        std::sort(keys.begin(), keys.end());
        side.facets.insert(keys);
      }
#else
      (void)side;
      (void)comm_pt;
#endif
    }

    unsigned global_sum(unsigned local, oomph::OomphCommunicator *comm_pt)
    {
#ifdef OOMPH_HAS_MPI
      if (comm_pt && comm_pt->nproc() > 1)
      {
        unsigned total = 0;
        MPI_Allreduce(&local, &total, 1, MPI_UNSIGNED, MPI_SUM, comm_pt->mpi_comm());
        return total;
      }
#else
      (void)comm_pt;
#endif
      return local;
    }

    // Is this rank's facet `keys` (of the side described by `mine`) coarser than the other side?
    //
    // Refinement is NESTED: when a side subdivides a facet, that facet's own corner nodes survive as
    // corners of the children. So the other side's vertex set contains the corners of every facet it
    // ever refined, as well as of its leaves. That gives an exact, geometry-free test -- no
    // point-in-facet predicates, no per-family refinement patterns, no level arithmetic (which would
    // be wrong anyway when the two domains start at different _initial_uniform_refinement_level):
    //
    //    keys in other.facets                 -> the two sides agree here
    //    else, every key of `keys` in other.vertices -> the other side subdivided THIS facet: too coarse
    //    else                                 -> this side is the finer one; the other side's own pass
    //                                            over the same facet pair will select its element
    //
    // The third branch also covers the genuinely incompatible case (a triangular face against a
    // quadrilateral one), where neither side can ever match: nothing is selected, the fixed point
    // terminates, and check_interface_conformity reports what is left.
    bool facet_is_too_coarse(const std::vector<InterfacePosKey> &keys, const InterfaceSideFacets &other)
    {
      if (other.facets.count(keys)) return false;
      for (const InterfacePosKey &k : keys)
        if (!other.vertices.count(k)) return false;
      return true;
    }

    // The pending adaptation decision for the element behind one boundary facet, as flag bits.
    int facet_flags(oomph::Mesh *om, TemplatedMeshBase *tm, unsigned elem_index)
    {
      oomph::RefineableElement *el = dynamic_cast<oomph::RefineableElement *>(om->element_pt(elem_index));
      if (!el) return 0;
      int f = 0;
      if (el->to_be_refined()) f |= IFACET_TO_BE_REFINED;
      oomph::Tree *father = (el->tree_pt() ? el->tree_pt()->father_pt() : NULL);
      if (father && father->object_pt())
      {
        oomph::RefineableElement *fel = dynamic_cast<oomph::RefineableElement *>(father->object_pt());
        if (fel && fel->sons_to_be_unrefined()) f |= IFACET_SONS_TO_BE_UNREFINED;
      }
      if (el->refinement_is_enabled() && tm && el->refinement_level() < tm->max_refinement_level())
        f |= IFACET_CAN_REFINE;
      return f;
    }

    // Collect one side AND the adaptation decision behind each of its facets, reduced across processes
    // so every rank sees the same picture of both sides. The two reductions differ on purpose:
    //   refine / can-refine are OR-ed  -- if any copy of an element wants (or is able) to refine, it does
    //   unrefine is AND-ed             -- a father is only merged away if every copy agrees to it
    // Halo and owner should already agree after synchronise_elemental_errors; these are the safe
    // directions if they somehow do not.
    void collect_side_with_flags(Mesh *m, const std::string &bname, const std::vector<double> &offset,
                                 InterfaceSideFacets &out, oomph::OomphCommunicator *comm_pt, bool topological)
    {
      collect_interface_side(m, bname, offset, out, topological);
      out.flags.clear();
      oomph::Mesh *om = dynamic_cast<oomph::Mesh *>(m);
      TemplatedMeshBase *tm = dynamic_cast<TemplatedMeshBase *>(m);
      if (om)
        for (const auto &f : out.local)
        {
          const int fl = facet_flags(om, tm, f.first.first);
          auto it = out.flags.find(f.second);
          if (it == out.flags.end())
            out.flags[f.second] = fl;
          else
            it->second = ((it->second | fl) & (IFACET_TO_BE_REFINED | IFACET_CAN_REFINE)) |
                         ((it->second & fl) & IFACET_SONS_TO_BE_UNREFINED);
        }

#ifdef OOMPH_HAS_MPI
      if (!comm_pt || comm_pt->nproc() < 2)
      {
        allgather_side(out, comm_pt);
        return;
      }
      MPI_Comm mc = comm_pt->mpi_comm();
      const int nproc = comm_pt->nproc();
      // [nkeys, keys..., flags] per facet
      std::vector<long long> mine;
      for (const auto &kv : out.flags)
      {
        mine.push_back((long long)kv.first.size());
        for (const InterfacePosKey &k : kv.first)
        {
          mine.push_back(k[0]);
          mine.push_back(k[1]);
          mine.push_back(k[2]);
        }
        mine.push_back((long long)kv.second);
      }
      int mycount = (int)mine.size();
      std::vector<int> counts(nproc, 0), displs(nproc, 0);
      MPI_Allgather(&mycount, 1, MPI_INT, &counts[0], 1, MPI_INT, mc);
      int total = 0;
      for (int i = 0; i < nproc; i++) { displs[i] = total; total += counts[i]; }
      std::vector<long long> all((size_t)std::max(total, 1));
      MPI_Allgatherv(mycount ? &mine[0] : NULL, mycount, MPI_LONG_LONG,
                     &all[0], &counts[0], &displs[0], MPI_LONG_LONG, mc);

      out.flags.clear();
      out.facets.clear();
      out.vertices.clear();
      int i = 0;
      while (i < total)
      {
        const int n = (int)all[i++];
        if (n <= 0 || i + 3 * n >= total + 1) break;
        std::vector<InterfacePosKey> keys;
        keys.reserve(n);
        for (int j = 0; j < n; j++)
        {
          InterfacePosKey k = {all[i], all[i + 1], all[i + 2]};
          i += 3;
          keys.push_back(k);
          out.vertices.insert(k);
        }
        const int fl = (int)all[i++];
        std::sort(keys.begin(), keys.end());
        out.facets.insert(keys);
        auto it = out.flags.find(keys);
        if (it == out.flags.end())
          out.flags[keys] = fl;
        else
          it->second = ((it->second | fl) & (IFACET_TO_BE_REFINED | IFACET_CAN_REFINE)) |
                       ((it->second & fl) & IFACET_SONS_TO_BE_UNREFINED);
      }
#else
      (void)comm_pt;
#endif
    }

    // --- The vertex-connected balance closure ------------------------------------------------------
    //
    // Conformity is a statement about FACETS, and so is everything above that restores it. That leaves
    // one blind spot, and it is not a small one: an element can touch a coupled interface at a single
    // VERTEX and carry no facet on it at all. Nothing above can see such an element -- it is not in
    // nboundary_element(), it contributes no key to either side's facet set -- so when the opposite
    // domain forces the interface to be refined, its facet-carrying neighbours follow and it does not.
    // The level jump across that shared vertex then grows without bound. Measured on an evaporating
    // droplet coupled to a gas domain, with a Z2 error estimator stated in the droplet alone: after
    // five adapts the gas carried level-0 elements sharing an interface vertex with level-4 ones, in a
    // refined band exactly one element thick.
    //
    // The error path already knows about this set of elements. Mesh::enlarge_elemental_error_max_
    // override_to_only_nodal_connected_elems spreads a boundary element's error override to precisely
    // the elements that touch the boundary only at a vertex, "to force refinement rather than leave a
    // 2:1 hang on the boundary". But it can only fire when the refinement is DRIVEN BY AN ERROR at
    // that interface. A refinement forced by the opposite domain never passes through an error at all
    // -- it is a flag reconciliation (harmonise_adapt_selection) and a facet repair (below) -- so
    // nothing spreads it, and the closure has to be stated here as well.
    //
    // Stated as the 2:1 rule at the vertex: an element sharing an interface vertex with a boundary
    // facet may not be more than one level coarser than the finest facet at that vertex. Two
    // properties make this safe to run inside the conformity fixed point:
    //
    //  * it never creates a facet. A corner element has no edge/face on the interface, so neither has
    //    any of its sons -- the son at the shared vertex is again corner-only. The closure therefore
    //    cannot cascade ALONG the interface, and cannot invalidate the facet sets the repair just
    //    agreed on. (It can split a facet of a DIFFERENT coupled interface the same element sits on;
    //    that is exactly why the two share one fixed point rather than running in sequence.)
    //  * it is monotone and bounded, by the finest facet level and by max_refinement_level().
    //
    // Unlike a conformity repair it is also not lossy: these elements have never been refined, so
    // their sons are interpolated from a father that still holds the current solution. There is no
    // unrefine/re-refine round trip to lose anything in, which is why the count is reported separately
    // from the repair count rather than added to it.
    bool vertex_balance_disabled()
    {
      // Negative-testing hatch, the C++ counterpart of PYOOMPH_DISABLE_INTERFACE_CONFORMITY: a test
      // that still passes with the closure switched off is not measuring the closure. Read on every
      // call rather than cached in a static, so a test can toggle it in-process and get the answer it
      // asked for rather than whichever value the first case in the worker happened to see.
      const char *v = getenv("PYOOMPH_DISABLE_INTERFACE_VERTEX_BALANCE");
      const std::string s(v ? v : "");
      return !(s == "" || s == "0" || s == "off");
    }

    // Globally MAX-reduce a "vertex key -> finest facet level" map, so every rank reaches the same
    // verdict about every element it holds. This is the reason to key on position rather than to union
    // selected elements after the fact (the pattern in TemplatedMeshBase3d::enforce_refinement_balance):
    // a rank can hold a coarse corner element whose fine facet neighbours all live elsewhere, and with
    // the reduced map it still sees the level it has to match, halo copy or not.
    void allreduce_level_map(std::map<InterfacePosKey, int> &m, oomph::OomphCommunicator *comm_pt)
    {
#ifdef OOMPH_HAS_MPI
      if (!comm_pt || comm_pt->nproc() < 2) return;
      MPI_Comm mc = comm_pt->mpi_comm();
      const int nproc = comm_pt->nproc();
      std::vector<long long> mine;
      mine.reserve(4 * m.size());
      for (const auto &kv : m)
      {
        mine.push_back(kv.first[0]);
        mine.push_back(kv.first[1]);
        mine.push_back(kv.first[2]);
        mine.push_back((long long)kv.second);
      }
      int mycount = (int)mine.size();
      std::vector<int> counts(nproc, 0), displs(nproc, 0);
      MPI_Allgather(&mycount, 1, MPI_INT, &counts[0], 1, MPI_INT, mc);
      int total = 0;
      for (int i = 0; i < nproc; i++) { displs[i] = total; total += counts[i]; }
      std::vector<long long> all((size_t)std::max(total, 1));
      MPI_Allgatherv(mycount ? &mine[0] : NULL, mycount, MPI_LONG_LONG,
                     &all[0], &counts[0], &displs[0], MPI_LONG_LONG, mc);
      m.clear();
      for (int i = 0; i + 3 < total; i += 4)
      {
        InterfacePosKey k = {all[i], all[i + 1], all[i + 2]};
        const int lvl = (int)all[i + 3];
        auto it = m.find(k);
        if (it == m.end()) m[k] = lvl;
        else it->second = std::max(it->second, lvl);
      }
#else
      (void)m;
      (void)comm_pt;
#endif
    }

    // Append to `out` the indices of this mesh's elements that touch boundary `bname` at a vertex only
    // and sit more than one level below the finest boundary facet at that vertex.
    void select_vertex_connected_too_coarse(TemplatedMeshBase *tm, const std::string &bname,
                                            oomph::OomphCommunicator *comm_pt, std::vector<unsigned> &out,
                                            bool topological)
    {
      if (!tm || vertex_balance_disabled()) return;
      oomph::Mesh *om = dynamic_cast<oomph::Mesh *>(tm);
      if (!om) return;
      const int bindi = find_boundary_index(tm, bname);
      if (bindi < 0) return; // a side may legitimately be absent
      const unsigned bind = (unsigned)bindi;

      const std::vector<double> no_offset; // one mesh, so no side-A/side-B translation applies here
      std::map<InterfacePosKey, int> finest;                    // interface vertex -> finest facet level
      std::set<oomph::GeneralisedElement *> carries_facet;       // elements the facet machinery already sees
      const unsigned nbe = om->nboundary_element(bind);
      for (unsigned i = 0; i < nbe; i++)
      {
        BulkElementBase *el = dynamic_cast<BulkElementBase *>(om->boundary_element_pt(bind, i));
        if (!el) continue;
        carries_facet.insert(el);
        oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(el);
        const int lvl = (re ? (int)re->refinement_level() : 0);
        std::vector<pyoomph::Node *> verts = el->get_vertex_nodes_of_face(om->face_index_at_boundary(bind, i));
        for (unsigned v = 0; v < verts.size(); v++)
        {
          const InterfacePosKey k = node_key(verts[v], no_offset, topological);
          auto it = finest.find(k);
          if (it == finest.end()) finest[k] = lvl;
          else it->second = std::max(it->second, lvl);
        }
      }
      allreduce_level_map(finest, comm_pt);
      if (finest.empty()) return;

      for (unsigned ie = 0; ie < om->nelement(); ie++)
      {
        BulkElementBase *el = dynamic_cast<BulkElementBase *>(om->element_pt(ie));
        // An element that carries a facet of this interface is the facet repair's business, not this
        // one's -- even if it ALSO touches the interface at some unrelated vertex. Same exclusion as
        // enlarge_elemental_error_max_override_to_only_nodal_connected_elems makes.
        if (!el || carries_facet.count(el)) continue;
        oomph::RefineableElement *re = dynamic_cast<oomph::RefineableElement *>(el);
        if (!re || !re->refinement_is_enabled()) continue;
        const int lvl = (int)re->refinement_level();
        if (lvl >= (int)tm->max_refinement_level()) continue; // cannot go finer; the jump has to stand
        for (unsigned v = 0; v < el->nvertex_node(); v++)
        {
          auto it = finest.find(node_key(el->vertex_node_pt(v), no_offset, topological));
          if (it != finest.end() && lvl < it->second - 1)
          {
            out.push_back(ie);
            break;
          }
        }
      }
    }

    // A topological key is a 128-bit digest, not a position, so it has to be printed as one -- rendering
    // it through the coordinate scale would put plausible-looking but meaningless numbers in an error
    // message that a reader is meant to act on.
    std::string key_to_string(const InterfacePosKey &k, bool topological)
    {
      std::ostringstream oss;
      if (topological)
      {
        oss << "#" << std::hex << (unsigned long long)k[0] << ":" << (unsigned long long)k[1] << std::dec;
        return oss.str();
      }
      oss << "(";
      for (int d = 0; d < 3; d++)
        oss << (double)k[d] / INTERFACE_COUPLING_KEY_SCALE << (d < 2 ? "," : "");
      oss << ")";
      return oss.str();
    }

    std::string facet_to_string(const std::vector<InterfacePosKey> &keys, bool topological)
    {
      std::ostringstream oss;
      for (unsigned i = 0; i < keys.size(); i++) oss << (i ? " " : "") << key_to_string(keys[i], topological);
      return oss.str();
    }

    // Facets whose counterpart does not exist on the other side AT ALL, globally reduced. Deliberately
    // not the same question as check_interface_conformity's full count: this one excludes the
    // "counterpart exists but not on this process" case, which is a halo-coverage problem and not
    // something refinement can repair.
    unsigned count_facet_mismatch(const std::vector<CoupledInterfacePair> &pairs,
                                  oomph::OomphCommunicator *comm_pt, std::string &detail)
    {
      unsigned n = 0;
      std::ostringstream oss;
      unsigned reported = 0;
      const unsigned MAX_REPORTED = 6;
      for (const CoupledInterfacePair &p : pairs)
      {
        InterfaceSideFacets A, B;
        const bool topo = pair_is_topological(p);
        collect_interface_side(p.meshA, p.bnameA, p.offset, A, topo);
        collect_interface_side(p.meshB, p.bnameB, std::vector<double>(), B, topo);
        allgather_side(A, comm_pt);
        allgather_side(B, comm_pt);
        if (A.facets.empty() && B.facets.empty()) continue;
        const InterfaceSideFacets *sides[2] = {&A, &B};
        const InterfaceSideFacets *others[2] = {&B, &A};
        const char *labels[2] = {"A", "B"};
        for (int s = 0; s < 2; s++)
          for (const auto &kv : sides[s]->facets)
            if (!others[s]->facets.count(kv))
            {
              n++;
              if (reported < MAX_REPORTED)
              {
                reported++;
                oss << "  side " << labels[s] << " facet " << facet_to_string(kv, topo)
                    << (facet_is_too_coarse(kv, *others[s]) ? "  [too coarse -- could not be refined]"
                                                            : "  [too fine]")
                    << "\n";
              }
            }
      }
      detail = oss.str();
      // Already identical on every rank (the facet sets are globally reduced), but reduce anyway so the
      // verdict -- and any throw built on it -- is unanimous rather than merely expected to be.
      unsigned total = n;
#ifdef OOMPH_HAS_MPI
      if (comm_pt && comm_pt->nproc() > 1)
        MPI_Allreduce(&n, &total, 1, MPI_UNSIGNED, MPI_MAX, comm_pt->mpi_comm());
#endif
      return total;
    }

  }

  void collect_interface_side(Mesh *m, const std::string &bname, const std::vector<double> &offset,
                              InterfaceSideFacets &out, bool topological)
  {
    out.facets.clear();
    out.vertices.clear();
    out.local.clear();
    if (!m) return;
    oomph::Mesh *om = dynamic_cast<oomph::Mesh *>(m);
    if (!om) return;
    const int bind = find_boundary_index(m, bname);
    if (bind < 0) return;

    // refine_selected_elements() addresses elements by their index in element_pt(), while the boundary
    // lookup hands out pointers; map once rather than searching per facet.
    std::map<oomph::GeneralisedElement *, unsigned> index_of;
    for (unsigned e = 0; e < om->nelement(); e++) index_of[om->element_pt(e)] = e;

    const unsigned nbe = om->nboundary_element((unsigned)bind);
    for (unsigned i = 0; i < nbe; i++)
    {
      BulkElementBase *el = dynamic_cast<BulkElementBase *>(om->boundary_element_pt((unsigned)bind, i));
      if (!el) continue;
      const int face = om->face_index_at_boundary((unsigned)bind, i);
      std::vector<pyoomph::Node *> verts = el->get_vertex_nodes_of_face(face);
      if (verts.size() < 2) continue; // 0d "point" faces carry no vertex set to match on
      std::vector<InterfacePosKey> keys;
      keys.reserve(verts.size());
      for (unsigned v = 0; v < verts.size(); v++) keys.push_back(node_key(verts[v], offset, topological));
      std::sort(keys.begin(), keys.end());
      auto found = index_of.find(el);
      if (found == index_of.end()) continue; // not an active element of this mesh (should not happen)
      out.local.push_back(std::make_pair(std::make_pair(found->second, face), keys));
      out.facets.insert(keys);
      for (unsigned v = 0; v < keys.size(); v++) out.vertices.insert(keys[v]);
    }
  }

  unsigned check_interface_conformity(const std::vector<CoupledInterfacePair> &pairs,
                                      const std::string &when, int mode)
  {
    if (pairs.empty()) return 0;
    oomph::OomphCommunicator *comm_pt = coupling_communicator(pairs);

    unsigned n_bad_local = 0;  // globally-reduced facet sets disagree: the meshes were refined differently
    unsigned n_unreachable = 0; // they agree, but this rank does not HOLD the partner it would be paired to
    std::ostringstream msg;
    const unsigned MAX_REPORTED = 8; // per side of a pair; enough to see the pattern, not a flood

    for (const CoupledInterfacePair &p : pairs)
    {
      InterfaceSideFacets A, B, A_local, B_local;
      // Both sides are keyed in side-A coordinates: the offset moves A onto B, so A carries it.
      const bool topo = pair_is_topological(p);
      collect_interface_side(p.meshA, p.bnameA, p.offset, A, topo);
      collect_interface_side(p.meshB, p.bnameB, std::vector<double>(), B, topo);
      A_local = A; // keep the rank-local facet sets before they are replaced by the global union
      B_local = B;
      allgather_side(A, comm_pt);
      allgather_side(B, comm_pt);
      if (A.facets.empty() && B.facets.empty()) continue;

      const InterfaceSideFacets *sides[2] = {&A, &B};
      const InterfaceSideFacets *others[2] = {&B, &A};
      const InterfaceSideFacets *loc[2] = {&A_local, &B_local};
      const InterfaceSideFacets *loc_other[2] = {&B_local, &A_local};
      const char *labels[2] = {"A", "B"};
      for (int s = 0; s < 2; s++)
      {
        unsigned reported = 0;
        for (const auto &kv : sides[s]->facets)
        {
          if (others[s]->facets.count(kv)) continue;
          n_bad_local++;
          if (mode && reported < MAX_REPORTED)
          {
            reported++;
            msg << "  side " << labels[s] << " facet not present on the other side: "
                << facet_to_string(kv, topo) << (facet_is_too_coarse(kv, *others[s]) ? "  [too coarse]" : "  [too fine]")
                << "\n";
          }
        }
        // The globally-conforming case is not yet enough for connect_interface_elements_by_kdtree, which
        // is RANK-LOCAL: it pairs the interface elements THIS rank holds. The two domains are partitioned
        // independently (they share no nodes, so the partitioner sees two disconnected components), so a
        // rank can hold one side of a facet pair and not the other -- a halo-coverage problem, not a
        // refinement one, and it needs a different fix. Counted and reported separately for exactly that
        // reason: the two failure modes look identical from the matcher's error message.
        //
        // Only meaningful when distributed. Serially the local sets ARE the global ones, so this would
        // just count the same mismatches a second time and report them under a misleading heading.
        if (!comm_pt || comm_pt->nproc() < 2) continue;
        unsigned reported_unreachable = 0;
        for (const auto &f : loc[s]->local)
        {
          if (loc_other[s]->facets.count(f.second)) continue;
          n_unreachable++;
          if (mode && reported_unreachable < MAX_REPORTED)
          {
            reported_unreachable++;
            msg << "  side " << labels[s] << " facet held here but its partner is NOT on this process: "
                << facet_to_string(f.second, topo) << "\n";
          }
        }
      }
    }

    // The facet sets are already globally reduced, so every rank counted the same thing; the sum is
    // only meaningful once. Reduce with MAX rather than SUM for exactly that reason, and so a throwing
    // verdict is unanimous -- an asymmetric throw would leave the other ranks blocked in the next
    // collective, which is the failure mode this check exists to prevent.
    unsigned n_bad = n_bad_local;
    unsigned n_unreach_total = n_unreachable;
#ifdef OOMPH_HAS_MPI
    if (comm_pt && comm_pt->nproc() > 1)
    {
      MPI_Allreduce(&n_bad_local, &n_bad, 1, MPI_UNSIGNED, MPI_MAX, comm_pt->mpi_comm());
      // Unreachable facets ARE per-rank, so sum them -- but still collectively, so the verdict below
      // is unanimous and a throwing check fails the whole job rather than one rank while the others
      // block in the next collective.
      MPI_Allreduce(&n_unreachable, &n_unreach_total, 1, MPI_UNSIGNED, MPI_SUM, comm_pt->mpi_comm());
    }
#endif

    if ((n_bad || n_unreach_total) && mode)
    {
      std::ostringstream full;
      full << "Interface conformity violated (" << when << "): ";
      if (n_bad)
        full << n_bad << " boundary facet(s) of a coupled interface have no counterpart on the opposite "
                         "side -- the two domains were refined differently along a shared interface.\n";
      if (n_unreach_total)
        full << n_unreach_total << " boundary facet(s) have a counterpart, but not on the process that "
                                   "holds them -- the halo layer does not cover the opposite domain.\n";
      full << "Either way the opposite-element matcher cannot pair them up.\n"
           << msg.str();
      if (mode > 1) throw_runtime_error(full.str());
      std::cout << "pyoomph WARNING: " << full.str() << std::flush;
    }
    return n_bad + n_unreach_total;
  }

  unsigned harmonise_adapt_selection(const std::vector<CoupledInterfacePair> &pairs, unsigned max_rounds)
  {
    if (pairs.empty()) return 0;
    oomph::OomphCommunicator *comm_pt = coupling_communicator(pairs);
    unsigned total_changes = 0;

    // Unrefinement first, then refinement -- and to separate fixed points, because the two have
    // opposite monotonicity. Deselecting an unrefinement never creates a new refinement selection and
    // selecting a refinement never creates a new unrefinement selection, so run in this order they
    // cannot chase each other.
    for (int phase = 0; phase < 2; phase++)
    {
      for (unsigned round = 0; round < max_rounds; round++)
      {
        unsigned changes = 0;
        for (const CoupledInterfacePair &p : pairs)
        {
          InterfaceSideFacets A, B;
          const bool topo = pair_is_topological(p);
          collect_side_with_flags(p.meshA, p.bnameA, p.offset, A, comm_pt, topo);
          collect_side_with_flags(p.meshB, p.bnameB, std::vector<double>(), B, comm_pt, topo);
          if (A.facets.empty() || B.facets.empty()) continue;

          oomph::Mesh *ms[2] = {dynamic_cast<oomph::Mesh *>(p.meshA), dynamic_cast<oomph::Mesh *>(p.meshB)};
          TemplatedMeshBase *tms[2] = {dynamic_cast<TemplatedMeshBase *>(p.meshA),
                                       dynamic_cast<TemplatedMeshBase *>(p.meshB)};
          InterfaceSideFacets *sides[2] = {&A, &B};
          const InterfaceSideFacets *others[2] = {&B, &A};
          for (int s = 0; s < 2; s++)
          {
            if (!ms[s] || !tms[s]) continue;
            for (const auto &f : sides[s]->local)
            {
              auto oit = others[s]->flags.find(f.second);
              if (oit == others[s]->flags.end()) continue; // no partner: already non-conforming, not our job
              const int other = oit->second;
              oomph::RefineableElement *el =
                  dynamic_cast<oomph::RefineableElement *>(ms[s]->element_pt(f.first.first));
              if (!el || !el->tree_pt()) continue;

              if (phase == 0)
              {
                // My father is about to be merged away but my partner's is not. Deselect mine: an
                // unrefinement cannot be manufactured on the other side, since that needs unanimity
                // among sons this code does not control.
                oomph::Tree *father = el->tree_pt()->father_pt();
                if (!father || !father->object_pt()) continue;
                oomph::RefineableElement *fel = dynamic_cast<oomph::RefineableElement *>(father->object_pt());
                if (!fel || !fel->sons_to_be_unrefined()) continue;
                if (other & IFACET_SONS_TO_BE_UNREFINED) continue;
                fel->deselect_sons_for_unrefinement();
                changes++;
              }
              else
              {
                const bool mine_refines = el->to_be_refined();
                const bool i_can = el->refinement_is_enabled() &&
                                   el->refinement_level() < tms[s]->max_refinement_level();
                if ((other & IFACET_TO_BE_REFINED) && !mine_refines && i_can)
                {
                  el->select_for_refinement();
                  changes++;
                }
                else if (mine_refines && !(other & IFACET_TO_BE_REFINED) && !(other & IFACET_CAN_REFINE))
                {
                  // The other side is capped (at its max_refinement_level, or refinement disabled), so
                  // it can never follow. Refining here would create a mismatch nothing downstream could
                  // repair, so hold instead. The other side reaches the mirror-image conclusion from the
                  // same globally-reduced flags, so the two agree without talking about it.
                  el->deselect_for_refinement();
                  changes++;
                }
              }
            }
          }
        }
        const unsigned global_changes = global_sum(changes, comm_pt);
        total_changes += global_changes;
        if (!global_changes) break;
        if (round + 1 == max_rounds)
          throw_runtime_error("harmonise_adapt_selection did not reach a fixed point in " +
                              std::to_string(max_rounds) + " rounds (phase " + std::to_string(phase) + ")");
      }
    }
    return total_changes;
  }

  unsigned enforce_interface_conformity(const std::vector<CoupledInterfacePair> &pairs, unsigned max_rounds,
                                        unsigned *n_vertex_balance)
  {
    if (n_vertex_balance) *n_vertex_balance = 0;
    if (pairs.empty()) return 0;
    oomph::OomphCommunicator *comm_pt = coupling_communicator(pairs);

    // Every mesh that takes part, deduplicated and in a deterministic order: the refinement calls
    // below are collective on a distributed mesh, so all ranks must visit them in the same sequence.
    std::vector<TemplatedMeshBase *> meshes;
    for (const CoupledInterfacePair &p : pairs)
    {
      TemplatedMeshBase *sides[2] = {dynamic_cast<TemplatedMeshBase *>(p.meshA),
                                     dynamic_cast<TemplatedMeshBase *>(p.meshB)};
      for (int s = 0; s < 2; s++)
        if (sides[s] && std::find(meshes.begin(), meshes.end(), sides[s]) == meshes.end())
          meshes.push_back(sides[s]);
    }

    unsigned total_refined = 0;   // rank-local: only ever used through global_refined below
    unsigned total_balanced = 0;  // ditto, for the vertex-connected closure
    unsigned global_refined = 0;
    bool converged = false;
    for (unsigned round = 0; round < max_rounds; round++)
    {
      // Each mesh's own 2:1 balancing refines further elements of its own accord, some of them at a
      // coupled interface -- which breaks conformity again. The two therefore have to converge
      // TOGETHER, in one fixed point, rather than one after the other.
      for (unsigned i = 0; i < meshes.size(); i++)
        if (meshes[i]->refinement_possible()) meshes[i]->enforce_refinement_balance();

      // Kept apart until both are deduplicated, so the two counts below are exact rather than an
      // apportionment of one merged list: a repair means information was lost and is worth reporting,
      // a vertex-balance refinement does not (see the closure).
      std::vector<std::vector<unsigned>> to_refine(meshes.size()), to_balance(meshes.size());
      for (const CoupledInterfacePair &p : pairs)
      {
        InterfaceSideFacets A, B;
        const bool topo = pair_is_topological(p);
        collect_interface_side(p.meshA, p.bnameA, p.offset, A, topo);
        collect_interface_side(p.meshB, p.bnameB, std::vector<double>(), B, topo);
        allgather_side(A, comm_pt);
        allgather_side(B, comm_pt);
        if (A.facets.empty() || B.facets.empty()) continue;

        TemplatedMeshBase *ms[2] = {dynamic_cast<TemplatedMeshBase *>(p.meshA),
                                    dynamic_cast<TemplatedMeshBase *>(p.meshB)};
        const InterfaceSideFacets *sides[2] = {&A, &B};
        const InterfaceSideFacets *others[2] = {&B, &A};
        for (int s = 0; s < 2; s++)
        {
          if (!ms[s] || !ms[s]->refinement_possible()) continue;
          const int mi = (int)(std::find(meshes.begin(), meshes.end(), ms[s]) - meshes.begin());
          if (mi < 0 || mi >= (int)meshes.size()) continue; // cannot happen: ms[s] was put in `meshes` above
          for (const auto &f : sides[s]->local)
          {
            if (!facet_is_too_coarse(f.second, *others[s])) continue;
            oomph::RefineableElement *re =
                dynamic_cast<oomph::RefineableElement *>(dynamic_cast<oomph::Mesh *>(ms[s])->element_pt(f.first.first));
            if (!re || !re->refinement_is_enabled()) continue;
            if (re->refinement_level() >= ms[s]->max_refinement_level()) continue; // cannot go finer
            to_refine[mi].push_back(f.first.first);
          }
          // ...and, in the same round, the elements that touch this interface at a vertex only. They
          // carry no facet, so the loop above is structurally blind to them; see the closure itself.
          select_vertex_connected_too_coarse(ms[s], (s == 0 ? p.bnameA : p.bnameB), comm_pt, to_balance[mi],
                                             pair_is_topological(p));
        }
      }

      // An element can carry more than one facet of the same interface (a corner element), and more
      // than one coupled interface can select it; refine_selected_elements would then split it twice.
      unsigned selected_local = 0;
      for (unsigned i = 0; i < meshes.size(); i++)
      {
        std::sort(to_refine[i].begin(), to_refine[i].end());
        to_refine[i].erase(std::unique(to_refine[i].begin(), to_refine[i].end()), to_refine[i].end());
        std::sort(to_balance[i].begin(), to_balance[i].end());
        to_balance[i].erase(std::unique(to_balance[i].begin(), to_balance[i].end()), to_balance[i].end());
        // An element selected by both is a repair; drop it from the balance list so it is counted once.
        std::vector<unsigned> only_balance;
        std::set_difference(to_balance[i].begin(), to_balance[i].end(),
                            to_refine[i].begin(), to_refine[i].end(), std::back_inserter(only_balance));
        to_balance[i].swap(only_balance);
        selected_local += (unsigned)(to_refine[i].size() + to_balance[i].size());
      }
      if (!global_sum(selected_local, comm_pt)) { converged = true; break; }

      for (unsigned i = 0; i < meshes.size(); i++)
      {
        // refine_selected_elements() ends in a collective adapt_mesh() on a distributed mesh, so every
        // rank has to enter it for every mesh -- including ranks with nothing of their own to refine.
        const unsigned n_here = (unsigned)(to_refine[i].size() + to_balance[i].size());
        if (!global_sum(n_here, comm_pt)) continue;
        oomph::Vector<unsigned> sel(n_here);
        for (unsigned k = 0; k < to_refine[i].size(); k++) sel[k] = to_refine[i][k];
        for (unsigned k = 0; k < to_balance[i].size(); k++) sel[to_refine[i].size() + k] = to_balance[i][k];
        total_refined += (unsigned)to_refine[i].size();
        total_balanced += (unsigned)to_balance[i].size();
        meshes[i]->refine_selected_elements(sel);
      }
      global_refined += global_sum(selected_local, comm_pt);
    }

    if (!converged)
    {
      // Every rank drove the loop off the same globally-summed selection count, so every rank arrives
      // here together and the throw is unanimous -- an asymmetric one would leave the others blocked in
      // the next collective.
      throw_runtime_error(
          "enforce_interface_conformity did not converge in " + std::to_string(max_rounds) +
          " rounds. This usually means the two sides of a coupled interface cannot be brought into "
          "correspondence at all -- e.g. incompatible facet shapes, or one side capped by a lower "
          "max_refinement_level than the other needs.");
    }

    // The extra refinement above bypassed the per-adapt hanging-node pass, so re-derive it. It is a
    // full, generative re-derivation (see TemplatedMeshBase3d::post_adapt_setup_hanging_nodes), not an
    // incremental one, so running it again is exactly what is wanted here; 2d is a no-op.
    //
    // Gated on the GLOBAL count, not this rank's: a rank that happened to refine nothing itself must
    // still re-derive its hanging nodes, or its (halo) copies of elements another rank refined keep a
    // stale hang scheme and the ranks disagree about the equation numbering.
    if (global_refined)
      for (unsigned i = 0; i < meshes.size(); i++)
        if (meshes[i]->refinement_possible()) meshes[i]->post_adapt_setup_hanging_nodes();

    // The loop stops as soon as it has nothing left to SELECT, which is not the same as having
    // succeeded: an element it wanted to refine may have been at max_refinement_level, or the two
    // sides' facets may not be in bijection at all. Left alone, that lands in
    // connect_interface_elements_by_kdtree a moment later as "Cannot locate opposite node at
    // x=(...)", which says nothing about the cause. Diagnose it here instead, while the reason is
    // still in scope.
    std::string detail;
    if (count_facet_mismatch(pairs, comm_pt, detail))
    {
      std::ostringstream msg;
      msg << "Cannot make the two sides of a coupled interface match.\n"
          << "The coarser side could not be refined any further, so no amount of refinement will "
             "bring them into correspondence. The usual cause is that the two domains have different "
             "max_refinement_level and the shallower one cannot follow the deeper one; the other is "
             "that their facets are not in bijection to begin with (e.g. a triangular face meeting a "
             "quadrilateral one), which no refinement can fix either.\n"
          << detail;
      throw_runtime_error(msg.str());
    }

    if (n_vertex_balance) *n_vertex_balance = total_balanced;
    return total_refined;
  }
}
