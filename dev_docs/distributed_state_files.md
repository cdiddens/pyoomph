# State files (save_state/load_state) for distributed problems

Status: **implemented and tested** (the *global* single-file variant: one state file, written by rank
0, interchangeable between serial and distributed runs). The sharded variant (one file per rank) is
deliberately left out and is additive afterwards — §7 says exactly what it would add. Not supported:
eigen/continuation data and tracers in a distributed state (§8).

One assumption of the original design turned out to be wrong and is corrected in §3.1 — the halo layer
does not contain a node's whole element star, so the node keys have to be reconciled between the
processes.

Before this, `Problem.save_state` refused outright:

```python
if self.is_distributed():
    raise RuntimeError("Distributed save state does not work. Consider to set write_states=False ...")
```

(`pyoomph/generic/problem.py:7120`), and `load_state` likewise. That also takes out `--runmode
continue` and the dedicated plotting subprocess under `mpirun --distribute`, because the latter
*requires* `write_states=True` and reads the dumps in a **serial** process.

---

## 1. What is in a state file, and how each part behaves

`Problem.define_state_file` (`problem.py:6929`) writes one sequential stream. It splits into three
kinds of data, and only one of them is actually hard:

| kind | what | under `--distribute` |
| --- | --- | --- |
| scalars | time, dts, output step, global parameters, remesher counters, compression level | identical on every rank — nothing to do |
| mesh-structural | refinement pattern, `nelem`/`nnode` assertions, the flat nodal buffer, elemental internal data, tracers | **rank-local in every respect** |
| dof-indexed | eigenvalues/eigenvectors, the arclength dof-derivative vector | in the distributed dof numbering, which `distribute()` permutes *and* splits |

The mesh-structural part is local in three independent ways:

1. **The refinement pattern.** `TreeBasedRefineableMeshBase::get_refinement_pattern`
   (`src/thirdparty/oomph-lib/include/refineable_mesh.cc:51`) walks this rank's forest and numbers
   elements level by level *within this rank's* mesh.
2. **The node ordering.** `Mesh::_save_state` (`src/mesh.cpp:230`) writes nodes in
   `get_node_reordering(old_ordering=true)` order, i.e. order of first appearance when walking the
   local element vector.
3. **The partition is not in the file at all**, and METIS output is not reproducible across runs,
   library versions or core counts.

## 2. Two constraints that shape everything

**oomph only distributes uniformly refined meshes.** `Problem::distribute` refuses otherwise: "In
order to preserve the Tree and TreeForest structure, `Problem::distribute()` can only be called while
meshes are uniformly refined" (`problem.cc:700`). So at distribution time every leaf sits at the same
level; all non-uniform refinement happens afterwards, per rank.

**But it partitions at the leaf level.** `element_domain` has one entry per *leaf* element
(`problem.cc:738`), so with a pre-distribution uniform refinement level > 0 the leaves of one base
element can end up on different ranks. A tree is therefore **not** guaranteed to stay on one rank,
which rules out "index within the root's subtree" as an identifier and forces explicit tree paths.

`refine_selected_elements(Vector<unsigned>)` (`refineable_mesh.h:562`) refines by *local* element
number, which is what lets a rank replay a globally-described refinement on its own elements.

## 3. The idea: address everything structurally

Nothing rank-local may appear in the file. Every record is therefore addressed by a key that a rank
can compute from the mesh structure alone, and that a serial run computes identically:

* **global root index** — the index of the root (base-mesh) element in the undistributed mesh.
  Assigned before `distribute()` and carried on the element itself, so it survives distribution
  (oomph backs the element objects up and re-adds them; it does not recreate them). In a serial run
  it is just the base element's position, so serial and distributed agree by construction.
* **tree path** — the chain of `Tree::son_type()` from the root down to the element (`tree.h:214`),
  packed into one integer: `path = 1; for each level: path = path*8 + (son_type+1)`. The leading 1
  distinguishes "root" from "first son of ...", and 3 bits per level gives ~20 levels in an int64,
  far beyond any real refinement depth.
* **element key** = `(root index, tree path)`. Unique across the whole problem, partition-independent,
  and stable under repartitioning.
* **node key** = the minimum of `(element key, local node index)` over the elements containing the
  node.

### 3.1 The node key needs reconciling — the halo layer is not enough

The design above assumed that *every element containing a node this rank holds is present on this
rank*, as an owned element or as a halo copy, so that the local minimum equals the one a serial run
computes. **That assumption is false**, and the first distributed load said so:

```
StateFileInconsistency: The state file has no entry for node [2 1 0] (13 of 105 missing)
```

The halo layer covers the **face** neighbours of the owned elements. An element that touches a node
only diagonally — the fourth quad around a corner node — can be absent, and then this rank's minimum
is larger than the serial one: the record is written under one key and looked up under another.

The fix, which the original design already carried as its fallback, is now part of the normal path:
the candidate keys are exchanged along oomph-lib's shared node scheme (the same index-matched
correspondence `Mesh.get_shared_node_numpy_indices` provides for the mesh data merge) and reduced to
the minimum, repeated until nothing changes anywhere so that nodes shared by more than two processes
converge too. Every element of the star is owned by *some* process and that process holds the node, so
the minimum over the processes **is** the minimum over the whole star — which is exactly what a serial
run computes. In serial the reconciliation is a no-op.

The consequence for §6 is that a distributed *load* is collective as well: the keys have to be
reconciled before they can be looked up. It is still true that every rank reads the whole file and
that no values are scattered.

## 4. File layout

### 4.1 Header

Every state file starts with three strings and ends with a footer, and all four are checked on load
(`Problem._define_state_header`, `DumpFile.check_footer`):

```
  "pyoomph_dump"   magic; anything else is not a state file
  version          e.g. "0.1.1"; a file newer than this build is refused
  sharding         "global" = this one file holds the whole problem
  ...
  "EOF_pyoomph"    footer; also catches a truncated (still-being-written) file
```

None of those checks is decoration: a file from another program desynchronizes on the first array and
dies somewhere inside numpy, a newer file reads plausible-looking garbage, and one part of a sharded
set would restore a fraction of the mesh without complaining. The footer is checked from the last
bytes without trusting the length prefix it finds there — reading whatever those eight bytes happened
to encode used to end in `MemoryError` rather than "this is not a state file".

The **sharding** field is what makes §7 additive: nothing writes anything but `"global"` today, and a
reader that meets `"sharded"` refuses with a clear message instead of silently loading a fraction of
the problem. It carries no process count — a global file must not depend on how many processes wrote
it, which is exactly what the byte-identity test of §10 pins down.

`_define_state_header` is shared with `_get_time_of_state_file`, which peeks at the first entries of a
file without reading the rest; the order of the header entries is therefore part of the format and the
two must not drift apart.

Versions are compared componentwise (`DumpFile.version_at_least`), not as strings: `"0.10.0" <
"0.2.0"` lexicographically, which would pick the wrong format branch the first time a component
reaches double digits.

### 4.2 Mesh section

`_dump_version` goes 0.0.1 → 0.1.0 (structural mesh section) → 0.1.1 (sharding field in the header)
→ 0.1.2 (the adaptive time stepper's suggested next dt) → 0.1.3 (the tracers' position history)
→ 0.1.4 (interface/skeleton element data) → 0.1.5 (the endtime of the run statement that wrote
the file, so `--runmode continue` can tell a state written mid-statement from one written by an
earlier statement that has since finished).
The reader branches on it and keeps reading old files; the writer only writes the new format, for
serial runs too. That is the point — a serial and a distributed run must produce interchangeable
files, which also means the format is exercised by every serial test.

Per bulk mesh (interface meshes are not saved; `define_state_file` asserts that), instead of
"refinement pattern + one flat nodal blob":

```
  n_roots                      int
  root_indices                 int64[n_roots]        global root index, ascending
  root_signatures              uint8[...]            preorder refinement tree, 1 = refined, 0 = leaf
  n_nodes                      int
  node_keys                    int64[n_nodes, 2]     (element key packed, local node index), sorted
  node_value_offsets           int64[n_nodes+1]      prefix sums; nvalue differs between nodes
  node_values                  float64[...]          positions (all history), xi, values (all history)
  n_elements                   int
  element_keys                 int64[n_elements, 2]  sorted
  element_value_offsets        int64[n_elements+1]
  element_values               float64[...]          internal data + initial size/quality
```

Every array is a numpy block, so `DumpFile`'s existing compression applies unchanged.

Sorting by key is what makes the file independent of `nproc` *and* of METIS' mood: two runs of the
same problem on different core counts produce byte-identical files.

## 5. Writing: gather to rank 0

`DumpFile` (`pyoomph/output/states.py:38`) is a strictly sequential, self-describing stream — no
index, no seeking, and a compressed block's length is not known until it is compressed. So no rank
can know its byte offset in advance and concurrent writing into one file is not possible without
changing the format. Rank 0 writes; the others send it flat buffers:

* two `int64` key arrays, one `float64` value array and the offset array per mesh, via `Gatherv`
  (rank order, deterministic);
* rank 0 `lexsort`s by key and drops duplicates — every shared and halo node collapses here;
* scalars are identical everywhere, so rank 0 writes them without any gather;
* the `nelem`/`nnode` assertions become an `allreduce` over owned counts, i.e. a real global check.

Sizing: with the ≤200k-dof rule from `CLAUDE.md` a node carries ~15-30 doubles, so a state is
~10-50 MB — a fraction of a second to gather and write, and less than rank 0 already holds for the
merged plotting data. Parallel I/O would buy nothing here (§7).

**Two consistency checks, both cheap, both worth keeping on:**

1. *Duplicate keys must carry identical values.* Two ranks reporting the same node with different
   values means halo values are out of sync — exactly the class of bug the hanging-value and halo work
   keeps turning up. Assert instead of letting first-one-wins hide it.
2. *Refinement trees.* Two processes describing the same root differently means one of them cannot see
   the whole tree; the writer raises rather than storing whichever description arrived first. This has
   not fired — unlike the node-key assumption (§3.1), a rank does see the complete tree of every root
   it touches, because the sons of a refined element are face neighbours of each other.

Rank-0-only writing goes through the `run_on_rank_zero` pattern (`pyoomph/generic/mpi.py:154`), or a
failed write leaves the other ranks waiting in the next collective.

## 6. Reading: every rank reads everything

Each rank opens the file and reads all of it:

1. replay the refinement: for each of its roots, look up the signature and refine level by level with
   `refine_selected_elements` on local element numbers. Halo elements are refined along with the owned
   ones — they are in the local element vector and the file describes them too — which is what keeps
   the local trees whole;
2. compute the key of every local node and element, **reconcile the node keys** (§3.1);
3. look the keys up in the file's sorted key array (`numpy.searchsorted` over a row view) and write the
   values into the nodes.

No values are scattered, and **halo nodes get their values for free**, because they are in the file
like everything else and each rank simply looks up the keys of the halo nodes it holds too. That
removes the post-load synchronisation a scatter-based design would need. The only communication is the
key reconciliation in step 2, which moves keys, not data.

Cost is `nproc` × the read of a small file, and after the first reader it is in page cache. If it ever
hurts, rank 0 reads and broadcasts the arrays.

This is also what makes serial↔parallel interchange free in both directions: a serial run reads a
file written on 4 ranks and vice versa, because neither side's keys mention the partition.

## 7. The sharded variant, later

With this format, "one file per rank" is not a second format — it is the same records without the
gather and the sort. Reading N parts is reading their concatenation, so **the reader is unchanged**,
and a serial process can read a sharded set just as well (which was the fatal drawback of per-rank
files in the classic oomph-lib scheme). Adding it later means:

* a `state_file_sharding = "global" | "sharded"` switch, defaulting to `"global"`. The **header field
  already exists** (§4.1) and today's reader refuses anything but `"global"`, so a sharded file cannot
  be mistaken for a whole one in the meantime;
* a manifest written last as the commit point, so a crash cannot leave `--runmode continue` reading a
  half-written checkpoint;
* part discovery in `continue`/replot (a part set is one state, not n states);
* an allreduce of per-rank success before the manifest is committed;
* the duplicate/consistency checks of §5 move into the reader, since parts keep their duplicates;
* an offline merge tool (read parts, sort, dedup, write one file) — trivial with this format.

Estimated at ~1 day on top of phase 1. Not built now, because at the dof cap the gather is a fraction
of a second and it would be effort spent on nothing measurable. What matters is that the format
decision is made now, and it is.

MPI-IO (`Write_at_all` with an `Exscan` over compressed block sizes) is the other alternative: real
parallel bandwidth, but the payload becomes per-rank blocks whose content depends on `nproc` unless
sorted globally first — which is the gather it was meant to avoid — and the serial reader would need a
block table. Only worth revisiting if states ever reach GB scale.

## 8. Deliberately not supported in phase 1

* **dof-indexed data under distribution** — eigenvalues/eigenvectors and the arclength dof-derivative
  vector. Both are already optional (`eigen_data_in_states`, `continuation_data_in_states`), so a
  distributed state is a state without them, which is honest rather than silently wrong. They raise if
  explicitly requested on a distributed problem. The same keying extends to them later: every dof
  belongs either to a nodal value or to an element's internal data, both of which now have keys.
* **tracers** — particles have no natural order; they need an id or a sort by position before they can
  be gathered. They raise on a distributed save for now.
* **loading a state whose mesh template differs** (the remeshing path in `define_state_file`) is
  untouched and remains serial-only.

## 9. C++ additions

```cpp
// pyoomph::Mesh
void assign_global_base_element_indices();           // before distribute(); no-op-safe to repeat
std::vector<long> get_element_structural_keys();     // (root index, packed path) per local element
std::vector<int>  get_element_node_indices();        // nelement x max_nnode, -1 padded, into node_pt order
std::vector<unsigned char> get_refinement_signature(unsigned root);  // preorder, 1 = refined
std::vector<long> get_root_element_indices();        // global index of each local root
// raw state, in node_pt / element order rather than the old reordering
void save_nodal_state(std::vector<double>&, std::vector<int>& lengths);
void load_nodal_state(const std::vector<double>&, const std::vector<int>& lengths);
```

`BulkElementBase` gains one `long global_base_index` (-1 until assigned), set on root elements.

## 10. Tests (as built)

`tests/test_mpi_state_files.py` + `tests/mpi_state_file_worker.py`, marked `slow`, 9 tests, ~15 s.
The comparison is a numbering-independent fingerprint: each nodal value weighted by the node's
position, summed over the non-halo elements. Values that landed on the wrong nodes change it, however
plausible the resulting plot would look. Every load first overwrites all nodal values with -12345, so
a load that quietly does nothing cannot pass.

1. **serial round trip** — catches format bugs without any MPI.
2. **serial file read by 2, 3 and 4 processes**, and **distributed files (2, 3) read serially**. This
   is the actual promise: the file does not mention the partition, so anything reads anything.
3. **byte-identical files** written from identical data by 1, 3 and 4 processes. The strongest form of
   the same statement, and it pins down the record *order*, not just the content.
4. **adaptive refinement replayed** — a state saved from an adaptively refined mesh (1377 nodes)
   loaded into runs that never adapted (324 nodes), serially and on 3 processes. A uniformly refined
   mesh would pass even with a broken signature; this does not.
5. **distributed adaptive round trip** — written on 3 ranks, read serially.

`tests/test_state_file_restart.py` (3 tests, serial, ~2 s) covers what a restart is actually for:
that the reloaded problem is in the writer's state *and* continues as if never interrupted. The
residual vector, every history level, the pinned values and the **Jacobian** must be bit-identical
right after loading — they are, for a plain transient, for temporal adaptivity (including the adapted
`dt`) and for a moving mesh. After continuing, each side has run its own Newton solves, so round-off
is allowed and nothing more.

That second half found a real defect, which had nothing to do with the file format. A freshly built
problem has `_taken_already_an_unsteady_step` unset, so the first `solve(timestep=...)` after a load
took the branch for the very first unsteady step: re-initialise `dt`, re-apply the initial condition,
reset the step counter — and take the step with the **degraded first-order start** instead of
continuing the scheme. The state was restored perfectly; the run then drifted from an uninterrupted
one by O(dt²). `load_state` now derives "we are mid-transient" from the restored step counter (so old
files behave too).

It is worth knowing why this survived: on a problem that has settled, `du/dt → 0` and BDF1 agrees with
BDF2, so a diffusion problem run to near-steady state reproduces to 1e-16 either way. The deviation
only appears when the solution is genuinely still moving — 4.9e-4 on the moving-mesh case with a
boundary that keeps oscillating. A restart test built from a settled problem would have passed
throughout, which is why the moving-mesh case is in that file on purpose.

**How exactly the continuation matches depends on the linear solver**, and the test covers both:

| solver | state after loading | continuation, moving mesh |
| --- | --- | --- |
| SuperLU (scipy) | bitwise | **bitwise**, over 1 and over 3 further steps |
| Pardiso (default here) | bitwise | 2.2e-16 |
| Pardiso, `reuse_symbolic_factorisation=False` | bitwise | **bitwise** |

SuperLU factorises from scratch every time, so its solve is a pure function of the matrix and a
correctly restarted run reproduces the uninterrupted one exactly — which makes it the sharpest
available check that nothing about the restart differs.

Pardiso's residue comes from its **symbolic** reuse: when the sparsity pattern is unchanged it runs
MKL phase 22 instead of phase 12 (`PardisoSolver.reuse_symbolic_factorisation`, on by default), and a
restarted run reaches that analysis with a different matrix than an uninterrupted one, so the reused
analysis is not the same one and the numeric factorisation differs in the last bits. Established by
elimination, not by inspection: switching that flag off makes Pardiso bitwise as well. It is **not**
thread nondeterminism — unchanged with `MKL_NUM_THREADS=1` — and `try_to_reuse_solver`, which would
reuse the *numeric* factors, is off by default and plays no part.

So if bitwise reproducibility across a restart is ever wanted with Pardiso, that one flag buys it, at
the cost of a full reordering on every factorisation.

`tests/test_state_file_header.py` (7 tests, serial, not marked slow, ~2 s) covers what the file says
about itself: the magic/version/sharding header and the footer are present; a foreign file, a
truncated file and a file from a newer version are all refused; a file declaring itself sharded is
refused with a message naming the reason; and the version comparison is componentwise.

Still worth adding: `--runmode continue` under `--distribute` end to end, and a 3d mesh.

## 11. Cost against the old format

Serial, in-process, 100 save/load pairs per arm, interleaved (A,B,A,B) because wall-clock A/B on this
machine is easily corrupted by load. The legacy path is still there and still writes, which is what
makes the comparison possible at all (`_define_state_file_legacy`).

| mesh | save legacy | save structural | load legacy | load structural | file |
| --- | --- | --- | --- | --- | --- |
| 3721 nodes / 900 elements | 11.2 ms | 19.1 ms (1.7x) | 64.6 ms | 71.7 ms (1.11x) | 82.6 kB -> 93.6 kB |
| 14641 nodes / 3600 elements | 53.4 ms | 87.9 ms (1.64x) | 315.7 ms | 341.2 ms (1.08x) | 349 kB -> 390 kB |

Medians. The ratios are the same at 4x the size, i.e. the extra work is linear, not quadratic.

Where the extra time goes on the larger mesh: ~21 ms computing the addresses in C++ (element node
indices 7.3, nodal state 3.9, signatures 3.9, keys 2.9, elemental state 3.0), ~10 ms sorting and
deduplicating, 0.7 ms writing the three extra key arrays. Loading barely moves because it is dominated
by rebuilding the mesh, which both formats pay.

The file grows by ~13%: that is the keys, and it is what buys addressing that does not depend on the
partition.

**One quadratic path was found and removed while measuring this.** `get_refinement_signature(root)`
re-scanned the whole element vector to find the root, so writing a state cost `O(nroots * nelement)`:
222 ms of a 228 ms save on the small mesh, and the 22x regression that provoked this comparison. It is
now one pass for all roots (`get_all_refinement_signatures`).

If more is ever needed: the element node indices build a `std::map<Node*,unsigned>` that the key
computation builds again, and the two C++ calls could be merged - worth perhaps another 10 ms on the
larger mesh, which is not obviously worth the coupling.

## 12. Possible later: move the state handling into C++

The whole mesh section currently lives in Python (`meshstate.py`), and every double of the state
crosses the boundary twice on the way out and twice on the way back: C++ fills a `std::vector`, that
becomes a numpy array, numpy reorders it into key order, and `DumpFile` writes it. The addressing
itself - element keys, node keys, the tree signatures - is computed in C++ already, but the sorting,
the deduplication and the variable-length block indexing are numpy.

Porting it would buy roughly what §11 measured as the non-C++ part: ~10 ms of sort/dedup plus the
array copies on a 14641-node mesh, so on the order of a third of the extra cost, and rather more on a
large 3d mesh where the nodal blob dominates. It would also let the raw values go straight from the
nodes into the file without ever being materialised as a numpy array.

What makes it more than a mechanical port:

* **The file format is Python.** `DumpFile` writes `numpy.lib.format` blocks and zlib-compressed
  buffers, and the surrounding sections (templates, timestepper, global parameters, eigendata) stay in
  Python either way. So a port either reimplements the numpy block format in C++ - it is simple and
  stable, but it is a second implementation of the container - or keeps the writing in Python and moves
  only the computation, handing over one finished array per section. **The second is the sensible
  first step**: it keeps `define_state_file` readable as the definition of the format, and the arrays
  handed over are exactly the ones §4 lists.
* **The MPI would change hands.** The gather and the key reconciliation currently go through mpi4py;
  in C++ they would go through oomph-lib's communicator (`MPI_Gatherv`, `MPI_Alltoall`). That is not
  harder, but it moves the collective structure - the part that hangs a run when it is wrong - out of
  the place where it is easy to read and instrument. Whatever is ported should keep §3.1's
  reconciliation as one clearly named function, not scattered through the writer.
* **Nothing about the format changes.** The keys, the sort order and the sections are what make the
  file partition-independent; a port must reproduce them byte for byte, and the byte-identity test of
  §10 is what would prove it. That test is worth keeping precisely for this reason.

Not planned for now: at the sizes this framework runs (the ≤200k-dof rule in `CLAUDE.md`) a state
costs tens of milliseconds and is written once per output step. It becomes worth doing if states grow
into the hundreds of MB, or if checkpointing every step ever becomes normal.
