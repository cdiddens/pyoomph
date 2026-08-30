from __future__ import annotations
#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  @author Maxim de Wildt <m.dewildt@utwente.nl>
#
#  @section LICENSE
#
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC
#  Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  The main author may be contacted at c.diddens@utwente.nl
#
# ========================================================================

"""Merging the per-rank mesh data of a distributed mesh into one global mesh on rank 0.

Each rank exports its own partition through :py:meth:`Mesh.to_numpy` (see
:py:class:`~pyoomph.meshes.meshdatacache.MeshDataCacheEntry`). Elements are owned by exactly one rank,
so concatenating the non-halo ones gives every element exactly once. Nodes on the partition interface,
however, exist on several ranks at once and must be identified with each other, or the merged mesh
falls apart into disconnected pieces along the partition boundaries - which is precisely what anything
topological (interface line segments, boundary identification for remeshing) would get wrong.

The identification is exact, not geometric: oomph-lib's shared node scheme
(``Mesh::setup_shared_node_scheme``) is "a unique correspondence between all nodes on the halo/haloed
elements between two processors", built in a matched order on both sides, and
``Mesh.get_shared_node_numpy_indices`` exposes it in the row numbering to_numpy uses.

Two ways to trigger a merge, both collective:

* every rank calls ``get_cached_mesh_data(..., global_mesh=True)`` with the same arguments, or
* rank 0 calls it alone from inside :py:func:`run_with_global_mesh_data`, while the other ranks serve
  its requests. This is what the plotters use: plot code is written rank-agnostically and only rank 0
  should draw, so the other ranks cannot be expected to reach the same requests by themselves.
"""

from typing import TYPE_CHECKING
from ..typings import *
import numpy

from ..generic.mpi import get_mpi_nproc, get_mpi_rank, get_mpi_world_comm, mpi_share_root_failure
from .meshdatacache import MeshDataCacheEntry, MeshDataCacheKey
from .mesh import AnySpatialMesh

if TYPE_CHECKING:
    from ..generic.problem import Problem

#: Coordinates of nodes that were identified with each other may differ by at most this, relative to
#: the mesh extent. They are the same node, so they should agree bit for bit; anything above this
#: means the correspondence itself is wrong and the merged mesh would be scrambled in a way that is
#: very hard to notice downstream.
MERGE_COORDINATE_TOLERANCE = 1e-9


def needs_merging(msh: AnySpatialMesh) -> bool:
    """Whether global data for this mesh actually has to be gathered.

    False for a serial run and equally for ``mpirun`` without ``--distribute``, where every rank holds
    the complete mesh and the local data already *is* the global data."""
    return bool(msh.is_mesh_distributed()) and get_mpi_nproc() > 1


class _UnionFind:
    """Plain union-find over the (rank, local node row) pairs, keyed by their flat index.

    Union always keeps the smaller index as the root, so a class' root is its smallest member, i.e.
    the copy on the lowest-numbered rank. That makes the global node order deterministic and, for a
    single rank, identical to the serial one."""

    def __init__(self, n: int):
        self.parent = numpy.arange(n, dtype=numpy.int64)

    def find(self, i: int) -> int:
        p = self.parent
        root = i
        while p[root] != root:
            root = int(p[root])
        while p[i] != root:  # path compression
            p[i], i = root, int(p[i])
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def _local_payload(msh: AnySpatialMesh, key: MeshDataCacheKey) -> dict[str, Any] | None:
    """Everything this rank contributes to the merge, or None if it holds no part of this mesh."""
    if msh.nelement() == 0:
        # to_numpy cannot even describe the fields without an element to ask
        return None
    problem = msh.get_problem()
    local_kwargs = key.as_kwargs()
    local_kwargs["global_mesh"] = False
    local = problem.get_cached_mesh_data(msh, **local_kwargs)
    assert local is not None

    # Local expressions have to be evaluated here, while all ranks are together: get_data() would ask
    # the local mesh for them, and for eigen data it even reassigns the dofs, which is collective.
    local_exprs: dict[str, NPFloatArray] = {}
    for name in local.local_expr_indices.keys():
        if name in local.nodal_field_inds.keys():
            continue  # get_data resolves nodal fields first, so this name never reaches the local expressions
        data = local.get_data(name)
        if data is not None:
            local_exprs[name] = numpy.asarray(data)

    nproc = get_mpi_nproc()
    myrank = get_mpi_rank()
    if key.discontinuous:
        # Node rows are per-element node copies here, so nothing is shared between rows in the first
        # place - and the shared scheme is expressed in the continuous node numbering, which does not
        # apply to them at all.
        shared: dict[int, NPAnyIntArray] = {}
    else:
        shared = {p: numpy.asarray(msh.get_shared_node_numpy_indices(p), dtype=numpy.int64)
                  for p in range(nproc) if p != myrank}

    return {"nodal_values": local.nodal_values, "elem_indices": local.elem_indices,
            "elem_types": local.elem_types, "D0_data": local.D0_data, "DL_data": local.DL_data,
            "nodal_field_inds": local.nodal_field_inds, "elemental_field_inds": local.elemental_field_inds,
            "local_expr_indices": local.local_expr_indices, "vector_fields": local.vector_fields,
            "tensor_fields": local.tensor_fields,
            "merged_eigendata": local.merged_eigendata, "nodal_local_exprs": local_exprs,
            "shared": shared}


def _check_metadata(payloads: list[dict[str, Any] | None]) -> dict[str, Any]:
    """The field layout describes the equations, not the partition, so it must agree everywhere.

    If it did not, the merged columns would mean different things in different row ranges - a
    corruption that no amount of downstream plotting would reveal."""
    present = [(r, p) for r, p in enumerate(payloads) if p is not None]
    ref_rank, ref = present[0]
    for r, p in present[1:]:
        for what in ("nodal_field_inds", "elemental_field_inds", "local_expr_indices"):
            if p[what] != ref[what]:
                raise RuntimeError("Cannot merge the mesh data: rank " + str(r) + " and rank " + str(ref_rank) +
                                   " disagree on " + what + ": " + str(p[what]) + " vs. " + str(ref[what]))
    return ref


def _merge_on_root(payloads: list[dict[str, Any] | None], msh: AnySpatialMesh, key: MeshDataCacheKey) -> MeshDataCacheEntry | None:
    have = {r: p for r, p in enumerate(payloads) if p is not None}  # rank -> what that rank contributed
    present = list(have)
    if not present:
        return None
    ref = _check_metadata(payloads)

    # --- node identification ------------------------------------------------------------------
    nnodes = [0 if p is None else p["nodal_values"].shape[0] for p in payloads]
    offsets = numpy.concatenate(([0], numpy.cumsum(nnodes))).astype(numpy.int64)
    uf = _UnionFind(int(offsets[-1]))
    for r in present:
        for p in present:
            if p <= r:
                continue
            a = have[r]["shared"].get(p)
            b = have[p]["shared"].get(r)
            if a is None or b is None or len(a) == 0 or len(b) == 0:
                continue
            if len(a) != len(b):
                raise RuntimeError("Shared node scheme mismatch between rank " + str(r) + " (" + str(len(a)) +
                                   " entries) and rank " + str(p) + " (" + str(len(b)) + " entries). The two sides of "
                                   "the scheme must describe the same nodes in the same order")
            both = (a >= 0) & (b >= 0)  # -1 marks a bulk node that an interface mesh does not contain
            for ia, ib in zip(a[both], b[both]):
                uf.union(int(offsets[r] + ia), int(offsets[p] + ib))

    roots = numpy.array([uf.find(i) for i in range(int(offsets[-1]))], dtype=numpy.int64)
    representatives = numpy.flatnonzero(roots == numpy.arange(len(roots)))  # ascending, i.e. lowest rank first
    global_of_root = numpy.full(len(roots), -1, dtype=numpy.int64)
    global_of_root[representatives] = numpy.arange(len(representatives))
    uid_to_global = global_of_root[roots]

    all_nodal = numpy.concatenate([have[r]["nodal_values"] for r in present], axis=0)
    nodal_values = all_nodal[representatives]

    # The identification is exact, so copies of one node must carry the same coordinates. If they do
    # not, the correspondence is off and everything downstream would be quietly wrong.
    coord_cols = [ref["nodal_field_inds"][c] for c in ("coordinate_x", "coordinate_y", "coordinate_z")
                  if c in ref["nodal_field_inds"]]
    if coord_cols:
        coords = all_nodal[:, coord_cols]
        deviation = numpy.abs(coords - nodal_values[uid_to_global][:, coord_cols])
        extent = float(numpy.amax(numpy.abs(coords))) if len(coords) else 1.0
        if len(deviation) and float(numpy.amax(deviation)) > MERGE_COORDINATE_TOLERANCE * max(extent, 1.0):
            raise RuntimeError("Nodes identified across processes disagree on their position by up to " +
                               str(float(numpy.amax(deviation))) + ". The shared node correspondence used to merge "
                               "the distributed mesh data does not describe the same nodes on both sides")

    # --- elements -----------------------------------------------------------------------------
    width = max(have[r]["elem_indices"].shape[1] if have[r]["elem_indices"].size else 0 for r in present)
    elem_rows: list[NPAnyIntArray] = []
    for r in present:
        ei = have[r]["elem_indices"]
        if ei.size == 0:
            continue
        mapped = numpy.zeros((ei.shape[0], width), dtype=numpy.int64)
        # Rows are padded to the widest element type present, and the padding is whatever was in the
        # buffer - only the first entries (as many as elem_types implies) are ever read. So only
        # entries that are valid local node indices are translated; the rest stay 0.
        valid = (ei >= 0) & (ei < nnodes[r])
        mapped[:, :ei.shape[1]] = numpy.where(valid, uid_to_global[numpy.clip(ei, 0, nnodes[r] - 1) + offsets[r]], 0)
        elem_rows.append(mapped)
    elem_indices = numpy.concatenate(elem_rows, axis=0) if elem_rows else numpy.zeros((0, width), dtype=numpy.int64)
    elem_types = numpy.concatenate([have[r]["elem_types"] for r in present], axis=0)

    def cat_elemental(what: str) -> NPFloatArray:
        return numpy.concatenate([have[r][what] for r in present], axis=0)

    def map_nodal(arrays: list[NPFloatArray]) -> NPFloatArray:
        return numpy.concatenate(arrays, axis=0)[representatives]

    if key.discontinuous:
        # Node rows are the per-element node blocks, so they are elemental data in disguise: no node is
        # shared between rows and the identification above does not apply to them.
        D0_data = map_nodal([have[r]["D0_data"] for r in present])
        DL_data = map_nodal([have[r]["DL_data"] for r in present])
    else:
        D0_data = cat_elemental("D0_data")
        DL_data = cat_elemental("DL_data")

    merged_eigendata: dict[int, dict[str, Any]] = {}
    for ev in have[present[0]]["merged_eigendata"].keys():
        entry: dict[str, Any] = {}
        for what in ("nodal_values", "DL_data", "D0_data"):
            parts = [have[r]["merged_eigendata"][ev][what] for r in present]
            if what == "nodal_values" or key.discontinuous:
                entry[what] = tuple(map_nodal([p[i] for p in parts]) for i in (0, 1))
            else:
                entry[what] = tuple(numpy.concatenate([p[i] for p in parts], axis=0) for i in (0, 1))
        merged_eigendata[ev] = entry

    nodal_local_exprs = {name: map_nodal([have[r]["nodal_local_exprs"][name] for r in present])
                         for name in ref["nodal_local_exprs"].keys()}

    return MeshDataCacheEntry.from_arrays(msh, key, nodal_values=nodal_values, elem_indices=elem_indices,
                                          elem_types=elem_types, nodal_field_inds=ref["nodal_field_inds"],
                                          D0_data=D0_data, DL_data=DL_data,
                                          elemental_field_inds=ref["elemental_field_inds"],
                                          merged_eigendata=merged_eigendata, nodal_local_exprs=nodal_local_exprs,
                                          local_expr_indices=ref["local_expr_indices"],
                                          vector_fields=ref["vector_fields"],
                                          tensor_fields=ref["tensor_fields"])


def merge_global_mesh_data(msh: AnySpatialMesh, key: MeshDataCacheKey) -> MeshDataCacheEntry | None:
    """Gathers this mesh's data from all ranks and merges it into one entry on rank 0 (None elsewhere).

    Collective: every rank must call it, either directly or through the request scope below."""
    if not needs_merging(msh):
        raise RuntimeError("merge_global_mesh_data called for a mesh that is not distributed")
    comm = get_mpi_world_comm()
    assert comm is not None  # needs_merging implies more than one process, i.e. mpi4py is there
    if _request_scope_depth > 0 and get_mpi_rank() == 0:
        _broadcast_request(msh, key)
    payloads = comm.gather(_local_payload(msh, key), root=0)
    if get_mpi_rank() != 0:
        return None
    assert payloads is not None
    return _merge_on_root(payloads, msh, key)


def merge_perturbed_global_mesh_data(msh: AnySpatialMesh, key: MeshDataCacheKey,
                                     eigenvector_index: int, factor: complex) -> MeshDataCacheEntry | None:
    """The merged data of the state ``dofs + Re(factor * eigenvector)``, without keeping that state.

    This is what an eigendynamics animation needs (see ``Problem.create_eigendynamics_animation``): a
    frame is the base state plus a complex multiple of an eigenvector, and a mirrored half of the same
    frame is the same thing with a different factor. Perturbing the dofs around the extraction is easy
    serially, but under ``run_with_global_mesh_data`` the plot code runs on rank 0 alone, so a
    perturbation applied there would be merged with everybody else's BASE state and the animation would
    silently show a mixture of the two.

    So the perturbation is part of the request instead: rank 0 announces which eigenvector and which
    factor, and every rank does the same perturb-extract-restore. Only two scalars travel, because an
    eigenvector is replicated full length on every rank anyway (see
    :py:meth:`pyoomph.solvers.petsc.SlepcEigenSolver._vector_to_global_array`) and
    ``get_current_dofs``/``set_current_dofs`` are global and collective.

    The request is ATOMIC on purpose: the restore is in a ``finally`` and ``_merge_on_root`` -- the only
    part that can fail on rank 0 alone -- runs after it, so no rank can be left holding a perturbed
    state, whatever goes wrong. That is also why this cannot simply wrap
    :py:func:`merge_global_mesh_data`.

    Collective, like :py:func:`merge_global_mesh_data`, and the result is NOT cached anywhere: the
    cache key has no room for the factor, and the two mirror halves of one frame would otherwise
    collide with each other and with the next frame.
    """
    # Everything that can fail for a reason rank 0 can see has to fail BEFORE the first collective.
    # Afterwards rank 0 would unwind while the others wait in a bcast or gather that never comes.
    if not needs_merging(msh):
        raise RuntimeError("merge_perturbed_global_mesh_data called for a mesh that is not distributed")
    if key.with_halos:
        raise RuntimeError("A perturbed global merge cannot be combined with with_halos=True")
    if key.operator is not None:
        # This path bypasses MeshDataCacheStorage.get_data, so its guards do not apply here.
        raise NotImplementedError("Mesh data operators are not supported together with a perturbed global merge")
    problem = msh.get_problem()
    eigenvectors = problem.get_last_eigenvectors()
    if eigenvectors is None or len(eigenvectors) <= eigenvector_index:
        raise RuntimeError("No eigenvector at index " + str(eigenvector_index) + " to perturb the state with")
    msh._setup_output_scales()  # MeshDataCache.get_data does this before merging; this call bypasses it
    comm = get_mpi_world_comm()
    assert comm is not None  # needs_merging implies more than one process, i.e. mpi4py is there
    if _request_scope_depth > 0 and get_mpi_rank() == 0:
        _broadcast_request(msh, key, perturbation=(int(eigenvector_index), complex(factor)))
    # Nothing rank-dependent may be added between the broadcast above and the gather below.
    olddofs, _ = problem.get_current_dofs()
    # Before: _local_payload reads the ORDINARY (global_mesh=False) cache slot, which the plotter's
    # field probes have already filled with base-state data. A hit there would feed the merge the
    # unperturbed state on every rank.
    problem.invalidate_cached_mesh_data()
    try:
        problem.set_current_dofs(numpy.asarray(olddofs) + numpy.real(factor * eigenvectors[eigenvector_index]))
        payloads = comm.gather(_local_payload(msh, key), root=0)
    finally:
        problem.set_current_dofs(olddofs)
        # After: the perturbed values are sitting in slots keyed as if they were the base state.
        problem.invalidate_cached_mesh_data()
    if get_mpi_rank() != 0:
        return None
    assert payloads is not None
    return _merge_on_root(payloads, msh, key)


class GatheredTracers:
    """Every process's tracer particles, in the API the plot code already uses for a local collection.

    ``get_positions``/``get_tags``/``get_ids``/``get_history`` behave as
    :py:class:`TracerCollection`'s do, so a consumer does not have to know which of the two it holds.
    The data is sorted by particle identity and therefore identical on every process and independent of
    the partitioning.

    ``get_dead_ids`` is always empty: a dead particle -- one that has left the domain but whose trail is
    still fading -- is deliberately not gathered (dev_docs/tracers.md), so on a distributed mesh those
    last trails are the one thing a plot cannot show.
    """

    def __init__(self, positions, tags, ids, histories):
        self._positions, self._tags, self._ids, self._histories = positions, tags, ids, histories

    def get_positions(self): return self._positions

    def get_tags(self): return self._tags

    def get_ids(self): return self._ids

    def get_dead_ids(self): return []

    def get_history(self, tid: int):
        return self._histories.get(int(tid))


def gather_global_tracers(msh: AnySpatialMesh, tracer_name: str, with_history: bool = False) -> GatheredTracers:
    """All particles of one tracer collection, gathered from every process.

    Collective, and answered identically everywhere -- the underlying gathers are ``MPI_Allgatherv``
    plus a sort by identity. Like the merges above it can be reached either by every rank calling it,
    or by rank 0 asking inside a :py:func:`run_with_global_mesh_data` scope while the others serve.

    A tracer collection is rank-local: a particle belongs to the process whose element it sits in, and
    it migrates as it advects. Drawing ``get_positions()`` on the plotting rank therefore shows that
    rank's partition of the particles and nothing else, which is what this replaces.
    """
    col = msh.get_tracers(tracer_name)
    if col is None:
        raise RuntimeError("There is no tracer collection '" + tracer_name + "' on mesh " + msh.get_full_name())
    if needs_merging(msh) and _request_scope_depth > 0 and get_mpi_rank() == 0:
        _broadcast_scope_request("tracers", _scope_name_of(msh), msh.get_full_name(),
                                 (tracer_name, bool(with_history)))
    dim = int(col.get_coordinate_dimension())
    if not with_history:
        positions = numpy.asarray(col.gather_positions(), dtype=float).reshape(-1, dim)
        return GatheredTracers(positions, numpy.asarray(col.gather_tags()),
                               numpy.asarray(col.gather_ids()), {})
    # The trails need the position history, and the only gather that carries it is the checkpoint one.
    # Its layout: the id array is (id, tag, nsamples) per particle, and the position array is one row
    # of (coordinates, payloads) per particle followed by all the history samples, each (time, coords).
    flat, idtags = col._save_state(True)
    flat = numpy.asarray(flat, dtype=float)
    idtags = numpy.asarray(idtags).reshape(-1, 3)
    n = int(idtags.shape[0])
    if n == 0:
        return GatheredTracers(numpy.zeros((0, dim)), numpy.zeros(0, dtype=int), numpy.zeros(0, dtype=int), {})
    hstride = 1 + dim
    counts = idtags[:, 2].astype(int)
    # The row stride is coordinates plus payloads, and the number of payloads is not exposed - derive
    # it from what is left once the history block is accounted for.
    stride = int((len(flat) - int(counts.sum()) * hstride) // n)
    positions = flat[:n * stride].reshape(n, stride)[:, :dim]
    histories: dict[int, Any] = {}
    offset = n * stride
    for i in range(n):
        k = int(counts[i])
        if k:
            histories[int(idtags[i, 0])] = flat[offset:offset + k * hstride].reshape(k, hstride)
        offset += k * hstride
    return GatheredTracers(positions, idtags[:, 1], idtags[:, 0], histories)


# --- letting rank 0 ask on its own --------------------------------------------------------------
#
# Plot code is written without any notion of rank and only rank 0 should draw, so the other ranks
# never reach the same get_cached_mesh_data calls. Instead they wait here and replay whatever rank 0
# asks for - the same call on every rank, which is what makes the gather (and anything collective
# inside the local extraction, such as reassigning the dofs for an eigenvector) line up.

_request_scope_depth = 0
_scope_problems: dict[str, "Problem"] = {}


def _scope_name_of(msh: AnySpatialMesh) -> str:
    """The name this mesh's problem goes by in the surrounding request scope."""
    problem = msh.get_problem()
    for name, p in _scope_problems.items():
        if p is problem:
            return name
    raise RuntimeError("Global mesh data was requested for a mesh of a problem that the surrounding "
                       "run_with_global_mesh_data call does not know about, so the other ranks cannot "
                       "resolve the request")


def _broadcast_scope_request(kind: str, problem_name: str, mesh_name: str, payload: Any) -> None:
    """Announce one request to the ranks in the serve loop. Every request is (kind, problem, mesh, payload)."""
    comm = get_mpi_world_comm()
    assert comm is not None
    comm.bcast((kind, problem_name, mesh_name, payload), root=0)


def _broadcast_request(msh: AnySpatialMesh, key: MeshDataCacheKey, *,
                       perturbation: "tuple[int, complex] | None" = None) -> None:
    name = _scope_name_of(msh)
    kwargs = key.as_kwargs()
    kwargs.pop("operator")  # refused for global data anyway, and would not survive the trip
    payload: Any = kwargs if perturbation is None else (kwargs, perturbation)
    _broadcast_scope_request("merge" if perturbation is None else "merge_perturbed",
                             name, msh.get_full_name(), payload)


def _serve_global_mesh_data_requests() -> None:
    """Answer rank 0's merge requests until it signals that it is done. Never runs on rank 0."""
    global _request_scope_depth
    comm = get_mpi_world_comm()
    assert comm is not None
    _request_scope_depth += 1  # so the merges below do not try to broadcast requests themselves
    try:
        while True:
            request = comm.bcast(None, root=0)
            if request is None:
                return
            kind, problem_name, mesh_name, payload = request
            mesh = _scope_problems[problem_name].get_mesh(mesh_name)
            if kind == "merge":
                merge_global_mesh_data(mesh, MeshDataCacheKey.create(**payload))
            elif kind == "merge_perturbed":
                kwargs, (index, factor) = payload
                merge_perturbed_global_mesh_data(mesh, MeshDataCacheKey.create(**kwargs), index, factor)
            elif kind == "tracers":
                gather_global_tracers(mesh, payload[0], with_history=payload[1])
            else:
                raise RuntimeError("Unknown global mesh data request '" + str(kind) + "'")
    finally:
        _request_scope_depth -= 1


def run_with_global_mesh_data(problems: Mapping[str, "Problem | None"], func: Callable[[], None], context: str = "") -> None:
    """Run ``func`` on rank 0 (on every rank in a serial run) while the others serve its merge requests.

    Inside ``func``, rank 0 may ask for ``global_mesh=True`` data on its own; each request is announced
    and replayed by the other ranks. Outside of it, every rank has to reach the same request by itself.

    Modelled on :py:func:`pyoomph.generic.mpi.run_on_rank_zero`, and for the same reason: whatever rank
    0 does here has to end for all ranks or for none, or a failure while the others wait turns into a
    job that never returns."""
    global _request_scope_depth, _scope_problems
    if get_mpi_nproc() <= 1:
        func()
        return
    _scope_problems = {name: p for name, p in problems.items() if p is not None}
    error: BaseException | None = None
    if get_mpi_rank() == 0:
        _request_scope_depth += 1
        try:
            func()
        except BaseException as e:
            error = e
        finally:
            _request_scope_depth -= 1
            comm = get_mpi_world_comm()
            assert comm is not None
            # Also on failure: the other ranks are in the serve loop and would wait for a request that
            # is never coming if rank 0 unwound past this point in silence.
            comm.bcast(None, root=0)
    else:
        _serve_global_mesh_data_requests()
    _scope_problems = {}
    # Collective and rooted at 0, so it doubles as the barrier this block would otherwise need
    mpi_share_root_failure(error, context=context or "gathering the global mesh data")


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
