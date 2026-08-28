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

"""The mesh part of a state file, addressed structurally instead of by position.

A state file must not contain anything rank-local, or it can only be read back by the very run that
wrote it. Everything here is therefore addressed by keys a process can compute from the mesh structure
alone and that a serial run computes identically:

* an **element** by ``(index of its root in the undistributed base mesh, path through the refinement
  tree)`` - see ``Mesh.assign_global_base_element_indices`` and ``Mesh.get_element_structural_keys``;
* a **node** by the smallest ``(element key, local node index)`` among the elements holding it;
* the **refinement** of a root by the shape of its tree (preorder son counts) rather than by
  oomph-lib's level-wise element numbers, which are relative to whatever the local mesh contains.

The records are then sorted by key before being written, so the file depends neither on the number of
processes nor on how METIS happened to cut the mesh: serial and distributed runs produce the same
file, and either can read the other's.

Writing gathers to rank 0 (``DumpFile`` is a sequential stream, so no two processes can write into it
at once). Reading is done by every process independently, which needs no communication at all and
fills the halo nodes on the way, since they are in the file like any other node.

See dev_docs/distributed_state_files.md.
"""

from ..typings import *
import numpy

from ..generic.mpi import get_mpi_nproc, get_mpi_rank, get_mpi_world_comm

if TYPE_CHECKING:
    from .mesh import AnySpatialMesh, BulkTemplateMesh, InterfaceMesh
    from ..output.states import DumpFile


class StateFileInconsistency(RuntimeError):
    """Raised when the processes disagree about something they must agree on for the file to make sense."""


def _mesh_is_distributed(mesh: "AnySpatialMesh") -> bool:
    return bool(mesh.is_mesh_distributed()) and get_mpi_nproc() > 1


def _row_view(a: NPAnyIntArray) -> NPAnyArray:
    """View a 2D int array as 1D, so that sorting and searching compare rows lexicographically."""
    a = numpy.ascontiguousarray(a)
    return a.view([("f" + str(i), a.dtype) for i in range(a.shape[1])]).ravel()  # type:ignore


def _gather_blocks(*arrays: NPAnyArray) -> list[NPAnyArray] | None:
    """Concatenate the given arrays across all ranks (rank order) onto rank 0, None elsewhere."""
    comm = get_mpi_world_comm()
    assert comm is not None
    gathered = comm.gather([numpy.asarray(a) for a in arrays], root=0)
    if get_mpi_rank() != 0:
        return None
    assert gathered is not None
    return [numpy.concatenate([g[i] for g in gathered], axis=0) for i in range(len(arrays))]


def _block_gather(data: NPFloatArray, lengths: NPAnyIntArray, take: NPAnyIntArray) -> tuple[NPFloatArray, NPAnyIntArray]:
    """Pick the variable-length blocks `take` out of (data, lengths), keeping their order."""
    offsets = numpy.concatenate(([0], numpy.cumsum(lengths))).astype(numpy.int64)
    lens = numpy.asarray(lengths, dtype=numpy.int64)[take]
    if len(lens) == 0:
        return numpy.zeros((0,), dtype=numpy.float64), lens.astype(numpy.int32)
    starts = offsets[take]
    # index gymnastics instead of a Python loop: repeat each block's start, then add a running offset
    out_starts = numpy.concatenate(([0], numpy.cumsum(lens)[:-1]))
    idx = numpy.repeat(starts - out_starts, lens) + numpy.arange(int(lens.sum()))
    return numpy.asarray(data, dtype=numpy.float64)[idx], lens.astype(numpy.int32)


def _node_keys(mesh: "AnySpatialMesh", elem_keys: NPAnyIntArray, elem_nodes: NPAnyIntArray) -> NPAnyIntArray:
    """For every node of the mesh, the smallest (root, path, local index) among its elements."""
    nnode = mesh.nnode()
    nelem, stride = elem_nodes.shape
    node_of = elem_nodes.reshape(-1)
    valid = node_of >= 0
    node_of = node_of[valid]
    root_of = numpy.repeat(elem_keys[:, 0], stride)[valid]
    path_of = numpy.repeat(elem_keys[:, 1], stride)[valid]
    local_of = numpy.tile(numpy.arange(stride, dtype=numpy.int64), nelem)[valid]
    order = numpy.lexsort((local_of, path_of, root_of, node_of))
    node_sorted = node_of[order]
    first = numpy.concatenate(([True], node_sorted[1:] != node_sorted[:-1]))
    sel = order[first]
    keys = numpy.zeros((nnode, 3), dtype=numpy.int64)
    keys[node_of[sel]] = numpy.column_stack((root_of[sel], path_of[sel], local_of[sel]))
    return keys


def _lexicographically_smaller(b: NPAnyIntArray, a: NPAnyIntArray) -> NPBoolArray:
    """Row-wise b < a, comparing the columns in order."""
    return ((b[:, 0] < a[:, 0]) |
            ((b[:, 0] == a[:, 0]) & ((b[:, 1] < a[:, 1]) |
                                     ((b[:, 1] == a[:, 1]) & (b[:, 2] < a[:, 2])))))


def _reconcile_node_keys(mesh: "AnySpatialMesh", keys: NPAnyIntArray) -> NPAnyIntArray:
    """Reduce every shared node's key to the smallest one any process computed for it.

    A node's key is the smallest address among the elements holding it, and a process can only look at
    the elements it has. That is not the whole star: the halo layer covers the face neighbours of the
    owned elements, but an element touching a node only diagonally can be missing, and then this
    process' minimum is larger than the one a serial run computes - the record would be written under
    one key and looked up under another.

    So the candidates are exchanged along oomph-lib's shared node scheme (the same index-matched
    correspondence the mesh data merge uses) and reduced to the minimum. Every element of the star is
    owned by *some* process, and that process holds the node, so the minimum over the processes is the
    minimum over the whole star - which is exactly what a serial run gets. Repeated until nothing
    changes anywhere, so that nodes shared by more than two processes converge as well."""
    if not _mesh_is_distributed(mesh):
        return keys
    comm = get_mpi_world_comm()
    assert comm is not None
    rank, nproc = get_mpi_rank(), get_mpi_nproc()
    shared = {p: numpy.asarray(mesh.get_shared_node_numpy_indices(p), dtype=numpy.int64)
              for p in range(nproc) if p != rank}
    absent = numpy.iinfo(numpy.int64).max
    for _round in range(nproc):
        payload: list[Any] = []
        for p in range(nproc):
            if p == rank:
                payload.append(None)
                continue
            idx = shared[p]
            mine = numpy.full((len(idx), 3), absent, dtype=numpy.int64)
            here = idx >= 0
            mine[here] = keys[idx[here]]
            payload.append(mine)
        received = comm.alltoall(payload)
        changed = False
        for p in range(nproc):
            if p == rank or received[p] is None or len(received[p]) == 0:
                continue
            idx = shared[p]
            if len(received[p]) != len(idx):
                raise StateFileInconsistency(
                    "Shared node scheme mismatch between rank " + str(rank) + " and rank " + str(p) +
                    " while reconciling the state file keys")
            here = idx >= 0
            theirs = numpy.asarray(received[p])[here]
            mine = keys[idx[here]]
            better = _lexicographically_smaller(theirs, mine)
            if numpy.any(better):
                target = idx[here][better]
                keys[target] = theirs[better]
                changed = True
        from mpi4py import MPI  # type:ignore # only reached on a distributed mesh, so mpi4py is there
        if not comm.allreduce(bool(changed), op=MPI.LOR):
            break
    return keys


def _owned_elements(mesh: "AnySpatialMesh") -> NPBoolArray:
    """Which elements this process owns. Halo copies belong to their owner, which writes them."""
    if not _mesh_is_distributed(mesh):
        return numpy.ones((mesh.nelement(),), dtype=bool)
    return numpy.array([mesh.element_pt(i).non_halo_proc_ID() < 0 for i in range(mesh.nelement())], dtype=bool)


def _local_contribution(mesh: "AnySpatialMesh") -> dict[str, NPAnyArray]:
    """Everything this process contributes to the file, already reduced to what it owns."""
    nelem = mesh.nelement()
    elem_keys = numpy.asarray(mesh.get_element_structural_keys(), dtype=numpy.int64).reshape(nelem, 2)
    flat, stride = mesh.get_element_node_indices()
    elem_nodes = numpy.asarray(flat, dtype=numpy.int64).reshape(nelem, int(stride)) if nelem else numpy.zeros((0, 1), dtype=numpy.int64)
    if numpy.any(elem_keys[:, 0] < 0):
        # Naming the mesh and the count: which mesh it is decides where the missing call belongs, and
        # "all of them" versus "a handful" separates a mesh that was never numbered from one whose
        # elements lost their tree root.
        bad = int(numpy.count_nonzero(elem_keys[:, 0] < 0))
        try:
            where = mesh.get_full_name()
        except Exception:
            where = repr(mesh)
        raise StateFileInconsistency(
            "The mesh '" + where + "' has " + str(bad) + " of " + str(nelem) + " elements without a global base "
            "index. assign_global_base_element_indices() must run before the problem is distributed - a state "
            "file written from rank-local element numbers could only be read back by the very run that wrote it")

    node_keys = _reconcile_node_keys(mesh, _node_keys(mesh, elem_keys, elem_nodes))
    owned = _owned_elements(mesh)

    # Nodes we speak for: those held by an element we own. After the reconciliation above their keys
    # are the ones a serial run computes, so the records of the different processes line up and the
    # duplicates collapse in _dedup.
    used = numpy.unique(elem_nodes[owned])
    used = used[used >= 0]

    node_data, node_lens = mesh.save_nodal_state()
    elem_data, elem_lens = mesh.save_elemental_state()
    my_node_data, my_node_lens = _block_gather(node_data, node_lens, used)
    my_elem_data, my_elem_lens = _block_gather(elem_data, elem_lens, numpy.flatnonzero(owned))

    roots, sig_lens, sig_data = mesh.get_all_refinement_signatures()
    roots = numpy.asarray(roots, dtype=numpy.int64)
    sig_lens = numpy.asarray(sig_lens, dtype=numpy.int32)
    sig_data = numpy.asarray(sig_data, dtype=numpy.int32)

    return {"roots": roots, "sig_lens": sig_lens, "sig_data": sig_data,
            "node_keys": node_keys[used], "node_lens": my_node_lens, "node_data": my_node_data,
            "elem_keys": elem_keys[owned], "elem_lens": my_elem_lens, "elem_data": my_elem_data}


def _dedup(keys: NPAnyIntArray, lengths: NPAnyIntArray, data: NPFloatArray, what: str,
           check: bool) -> tuple[NPAnyIntArray, NPAnyIntArray, NPFloatArray]:
    """Sort records by key and drop duplicates, checking that duplicates actually agree."""
    if len(keys) == 0:
        return keys, lengths, data
    order = numpy.argsort(_row_view(keys), kind="stable")
    keys = keys[order]
    data, lengths = _block_gather(data, lengths, order)  # blocks follow their keys
    view = _row_view(keys)
    unique = numpy.concatenate(([True], view[1:] != view[:-1]))
    if check and not numpy.all(unique):
        _check_duplicates_agree(keys, lengths, data, unique, what)
    take = numpy.flatnonzero(unique)
    data, lengths = _block_gather(data, lengths, take)
    return keys[take], lengths, data


def _check_duplicates_agree(keys: NPAnyIntArray, lengths: NPAnyIntArray, data: NPFloatArray,
                            unique: NPBoolArray, what: str) -> None:
    """Two processes reporting the same record must report the same values.

    They are the same node or element, so the only way the values can differ is that the halo copies
    were not in sync when the state was written - which would otherwise end up in the file as
    whichever process happened to come first."""
    offsets = numpy.concatenate(([0], numpy.cumsum(numpy.asarray(lengths, dtype=numpy.int64))))
    group = numpy.cumsum(unique) - 1
    for g in numpy.flatnonzero(numpy.bincount(group) > 1):
        members = numpy.flatnonzero(group == g)
        ref = data[offsets[members[0]]:offsets[members[0] + 1]]
        for m in members[1:]:
            other = data[offsets[m]:offsets[m + 1]]
            if len(other) != len(ref) or not numpy.array_equal(other, ref):
                # How far apart, not just "different". A handful of entries off by round-off is a halo
                # that drifted; a value that is simply unrelated means the two records are not the same
                # entity at all and the KEY is wrong - which is what it turned out to be when
                # Problem::distribute() re-rooted the refinement tree and 240 of 256 addresses
                # collided. The message used to name the halo as the cause and sent the reader the
                # wrong way, so it now reports what was measured and leaves the diagnosis open.
                if len(other) == len(ref):
                    diff = numpy.abs(numpy.asarray(other) - numpy.asarray(ref))
                    worst = int(numpy.argmax(diff))
                    detail = (": %d of %d entries differ, worst |a-b|=%.3e at index %d (%r vs %r)" %
                              (int((diff > 0).sum()), len(diff), float(diff[worst]), worst,
                               ref[worst], other[worst]))
                else:
                    detail = ": one reports %d entries, the other %d" % (len(ref), len(other))
                raise StateFileInconsistency(
                    "Two processes report different values for the same " + what + " " + str(keys[members[0]]) +
                    " while writing the state file" + detail + ". Either their halo copies were out of sync, or "
                    "the two records are different " + what + "s that were given the same key")


def _sorted_records(local: dict[str, NPAnyArray], distributed: bool, check: bool) -> dict[str, NPAnyArray] | None:
    """Gather every process' contribution and reduce it to one sorted, duplicate-free set."""
    if not distributed:
        merged = dict(local)
    else:
        gathered = _gather_blocks(local["roots"], local["sig_lens"], local["sig_data"],
                                  local["node_keys"], local["node_lens"], local["node_data"],
                                  local["elem_keys"], local["elem_lens"], local["elem_data"])
        if gathered is None:
            return None
        merged = dict(zip(["roots", "sig_lens", "sig_data", "node_keys", "node_lens", "node_data",
                           "elem_keys", "elem_lens", "elem_data"], gathered))

    # Refinement signatures: one per root, and the processes sharing a root must describe the same tree
    roots, sig_lens, sig_data = merged["roots"], merged["sig_lens"], merged["sig_data"]
    sig_by_root: dict[int, NPAnyIntArray] = {}
    offs = numpy.concatenate(([0], numpy.cumsum(numpy.asarray(sig_lens, dtype=numpy.int64))))
    for i, r in enumerate(roots):
        sig = sig_data[offs[i]:offs[i + 1]]
        previous = sig_by_root.get(int(r))
        if previous is None:
            sig_by_root[int(r)] = sig
        elif check and not numpy.array_equal(previous, sig):
            raise StateFileInconsistency(
                "Two processes describe different refinement trees for root element " + str(int(r)) +
                ". A process must see the whole tree of every root it touches (its own elements plus the halo "
                "copies of the others) for the refinement to be storable independently of the partition")
    sorted_roots = numpy.array(sorted(sig_by_root.keys()), dtype=numpy.int64)
    out_lens = numpy.array([len(sig_by_root[int(r)]) for r in sorted_roots], dtype=numpy.int32)
    out_sigs = numpy.concatenate([sig_by_root[int(r)] for r in sorted_roots]) if len(sorted_roots) else numpy.zeros((0,), dtype=numpy.int32)

    node_keys, node_lens, node_data = _dedup(merged["node_keys"], merged["node_lens"], merged["node_data"], "node", check)
    elem_keys, elem_lens, elem_data = _dedup(merged["elem_keys"], merged["elem_lens"], merged["elem_data"], "element", check)
    return {"roots": sorted_roots, "sig_lens": out_lens, "sig_data": out_sigs,
            "node_keys": node_keys, "node_lens": node_lens, "node_data": node_data,
            "elem_keys": elem_keys, "elem_lens": elem_lens, "elem_data": elem_data}


def save_interface_state(mesh: "InterfaceMesh", state: "DumpFile", check_consistency: bool = True) -> None:
    """Write one interface mesh's element-internal data.

    Interface elements carry all of a facet field's degrees of freedom - DL/D0 and the nodal DG spaces
    alike - in their own internal Data, and nothing else in the file holds them: the bulk record covers
    bulk elements and nodes only. Without this block a state file cannot reproduce the state it was
    written at, it can only reproduce the bulk part and refit whatever the loading process happened to
    be holding on the facets.

    Addressed by (root element index, packed refinement path, face index) rather than by element
    number, so the record survives a different partition - the same key the interior-facet halo scheme
    pairs facets across ranks with. Halo copies are skipped: their owner writes them."""
    nelem = mesh.nelement()
    keys = numpy.asarray(mesh.get_interface_element_structural_keys(), dtype=numpy.int64).reshape(nelem, 3)
    if nelem and numpy.any(keys[:, 0] < 0):
        raise StateFileInconsistency(
            "Interface mesh elements without a global base index; assign_global_base_element_indices() "
            "must run before the problem is distributed")
    owned = _owned_elements(mesh)
    data, lens = mesh.save_elemental_state()
    my_data, my_lens = _block_gather(data, lens, numpy.flatnonzero(owned))
    my_keys: NPAnyIntArray = keys[owned]
    distributed = _mesh_is_distributed(mesh)
    if distributed:
        gathered = _gather_blocks(my_keys, my_lens, my_data)
        if gathered is None:
            return
        my_keys, my_lens, my_data = gathered[0], gathered[1], gathered[2]
    my_keys, my_lens, my_data = _dedup(my_keys, my_lens, my_data, "interface element", check_consistency)
    for value in (my_keys, my_lens, my_data):
        state.numpy_data(lambda value=value: value, lambda v: v)  # type:ignore


def read_interface_state(state: "DumpFile") -> "tuple[NPAnyIntArray, NPAnyIntArray, NPFloatArray]":
    """Read back what save_interface_state wrote, without applying it.

    Applying is a separate step because the interface elements do not exist in their final form at the
    point the file is read: Problem._load_state rebuilds them from the loaded bulk mesh afterwards, in
    actions_after_adapt(). See Problem._apply_interface_states."""
    keys = state.numpy_data(lambda: numpy.zeros((0, 3), dtype=numpy.int64), lambda v: v)  # type:ignore
    lens = state.numpy_data(lambda: numpy.zeros((0,), dtype=numpy.int32), lambda v: v)  # type:ignore
    data = state.numpy_data(lambda: numpy.zeros((0,), dtype=numpy.float64), lambda v: v)  # type:ignore
    return keys, lens, data


def apply_interface_state(mesh: "InterfaceMesh", keys: "NPAnyIntArray", lens: "NPAnyIntArray",
                          data: "NPFloatArray") -> int:
    """Push a read interface record onto a rebuilt interface mesh. Returns how many elements it filled.

    Every element this process holds - halo copies included, so that a halo and its owner agree without
    a synchronisation - is looked up by its own key. An element whose key is not in the file keeps
    whatever the rebuild left it with (the refit, or the recovery expression), which is the right answer
    for a mesh that has been refined since the file was written."""
    nelem = mesh.nelement()
    if not nelem or not len(keys):
        return 0
    mine = numpy.asarray(mesh.get_interface_element_structural_keys(), dtype=numpy.int64).reshape(nelem, 3)
    index = {tuple(int(c) for c in k): i for i, k in enumerate(numpy.asarray(keys, dtype=numpy.int64))}
    offsets = numpy.concatenate(([0], numpy.cumsum(numpy.asarray(lens, dtype=numpy.int64))))
    # Start from what the mesh currently holds and substitute the blocks the file speaks for.
    # load_elemental_state() consumes one block per element in element order and uses `lengths` only
    # as a count check, so a "leave this one alone" entry has to be the element's OWN current block,
    # not an empty one.
    cur_data, cur_lens = mesh.save_elemental_state()
    cur_data = numpy.asarray(cur_data, dtype=numpy.float64)
    cur_lens = numpy.asarray(cur_lens, dtype=numpy.int64)
    cur_off = numpy.concatenate(([0], numpy.cumsum(cur_lens)))
    out_data: list[float] = []
    nfound = 0
    for ie in range(nelem):
        rec = index.get(tuple(int(c) for c in mine[ie]))
        block = (data[offsets[rec]:offsets[rec + 1]] if rec is not None
                 else cur_data[cur_off[ie]:cur_off[ie + 1]])
        if rec is not None:
            if len(block) != int(cur_lens[ie]):
                raise StateFileInconsistency(
                    "Interface element " + str(tuple(int(c) for c in mine[ie])) + " has " + str(int(cur_lens[ie])) +
                    " values but the state file holds " + str(len(block)) + " for it. The facet fields of this "
                    "interface differ from the ones the file was written with")
            nfound += 1
        out_data.extend(float(v) for v in block)
    mesh.load_elemental_state(out_data, [int(v) for v in cur_lens])
    return nfound


def save_mesh_state(mesh: "AnySpatialMesh", state: "DumpFile", check_consistency: bool = True) -> None:
    """Write the mesh part of a state file.

    Collective on a distributed mesh, where the contributions are gathered onto rank 0 and only rank
    0 has anything to write. On a mesh that is NOT distributed every rank holds the whole thing and
    _sorted_records hands each of them the complete set, so each writes its own copy - which matters
    for an in-memory snapshot (Problem._snapshot_state), where the streams are per-rank private and a
    rank that wrote nothing would be left unable to restore. It does not matter when writing a file,
    because save_state drops the redundant writers before reaching here."""
    distributed = _mesh_is_distributed(mesh)
    local = _local_contribution(mesh)
    records = _sorted_records(local, distributed, check_consistency)
    if distributed and get_mpi_rank() != 0:
        return
    assert records is not None
    for name in ("roots", "sig_lens", "sig_data", "node_keys", "node_lens", "node_data",
                 "elem_keys", "elem_lens", "elem_data"):
        state.numpy_data(lambda name=name: records[name], lambda v: v)  # type:ignore


def _decode_signature(root: int, signature: NPAnyIntArray) -> dict[tuple[int, int], int]:
    """Preorder son counts -> {(root, path): number of sons}."""
    out: dict[tuple[int, int], int] = {}
    position = 0

    def walk(path: int) -> None:
        nonlocal position
        nsons = int(signature[position])
        position += 1
        out[(root, path)] = nsons
        for s in range(nsons):
            walk(path * 8 + s + 1)

    walk(1)
    return out


def _replay_refinement(mesh: "AnySpatialMesh", roots: NPAnyIntArray, sig_lens: NPAnyIntArray, sig_data: NPAnyIntArray) -> None:
    """Refine the local elements until their trees have the shape the file describes."""
    offs = numpy.concatenate(([0], numpy.cumsum(numpy.asarray(sig_lens, dtype=numpy.int64))))
    nsons: dict[tuple[int, int], int] = {}
    for i, r in enumerate(roots):
        nsons.update(_decode_signature(int(r), sig_data[offs[i]:offs[i + 1]]))
    while True:
        keys = numpy.asarray(mesh.get_element_structural_keys(), dtype=numpy.int64).reshape(mesh.nelement(), 2)
        # Every element of the mesh is a leaf, so anything the file describes as having sons must be split.
        # Halo elements are refined along with the owned ones, which is what keeps the local trees whole.
        to_refine = [i for i, (r, p) in enumerate(keys) if nsons.get((int(r), int(p)), 0) > 0]
        if not to_refine:
            break
        mesh.refine_selected_elements_by_index(to_refine)


def _lookup(file_keys: NPAnyIntArray, my_keys: NPAnyIntArray, what: str) -> NPAnyIntArray:
    if len(my_keys) == 0:
        return numpy.zeros((0,), dtype=numpy.int64)
    if len(file_keys) == 0:
        raise StateFileInconsistency("The state file contains no " + what + " records, but the mesh has " + str(len(my_keys)))
    have, want = _row_view(file_keys), _row_view(my_keys)
    idx = numpy.searchsorted(have, want)
    idx = numpy.clip(idx, 0, len(have) - 1)
    missing = have[idx] != want
    if numpy.any(missing):
        first = int(numpy.flatnonzero(missing)[0])
        raise StateFileInconsistency(
            "The state file has no entry for " + what + " " + str(my_keys[first]) + " (" + str(int(numpy.sum(missing))) +
            " of " + str(len(my_keys)) + " missing). The mesh this state is being loaded into is not the one it was "
            "written from")
    return idx


def load_mesh_state(mesh: "BulkTemplateMesh", state: "DumpFile") -> None:
    """Read the mesh part of a state file. Every process reads all of it and picks what it holds.

    A bulk mesh rather than any spatial one: the replay below rewinds the refinement, which an
    interface mesh cannot do - it is rebuilt from its bulk instead."""
    read = {}
    for name in ("roots", "sig_lens", "sig_data", "node_keys", "node_lens", "node_data",
                 "elem_keys", "elem_lens", "elem_data"):
        read[name] = state.numpy_data(lambda: 0, lambda v: v)  # type:ignore

    while not mesh.unrefine_uniformly():  # down to the base mesh, then replay
        pass
    _replay_refinement(mesh, read["roots"], read["sig_lens"], read["sig_data"])

    # Interface meshes must exist before the nodal values are written, so that the additional
    # interface dofs are allocated on the nodes (same reason as in the pre-existing loader)
    for _, im in mesh._interfacemeshes.items():  # type:ignore
        im.rebuild_after_adapt()

    nelem = mesh.nelement()
    elem_keys = numpy.asarray(mesh.get_element_structural_keys(), dtype=numpy.int64).reshape(nelem, 2)
    flat, stride = mesh.get_element_node_indices()
    elem_nodes = numpy.asarray(flat, dtype=numpy.int64).reshape(nelem, int(stride)) if nelem else numpy.zeros((0, 1), dtype=numpy.int64)
    my_node_keys = _reconcile_node_keys(mesh, _node_keys(mesh, elem_keys, elem_nodes))

    idx = _lookup(read["node_keys"], my_node_keys, "node")
    data, lens = _block_gather(read["node_data"], read["node_lens"], idx)
    # cast: the generated stubs say Sequence[float]/Sequence[int], but nanobind's std::vector caster
    # takes these arrays as they are, and converting them to lists would copy the whole state.
    mesh.load_nodal_state(cast(Sequence[float], data), cast(Sequence[int], lens))

    idx = _lookup(read["elem_keys"], elem_keys, "element")
    data, lens = _block_gather(read["elem_data"], read["elem_lens"], idx)
    mesh.load_elemental_state(cast(Sequence[float], data), cast(Sequence[int], lens))


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
