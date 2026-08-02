# Inverted elements: detection, Gmsh element winding, and remeshing as the response

Three pieces of work, at three different stages:

1. **Detection** — `oomph::InvertedElementError` raised from the assembly when the signed
   determinant of `dx/ds` turns non-positive. Already in the tree; backported and described in
   `src/thirdparty/INFO_oomph-lib` (section "elements.h / problem.cc … InvertedElementError").
   This document does not repeat that; it starts from what the detector does *not* solve.
2. **`GmshTemplate.fix_2d_orientation`** — **implemented** (section 4). Without it, detection and
   remeshing cannot be used together at all, which is how it was found.
3. **Remeshing as the response to an inversion** — designed, **not implemented** (sections 5–8).
   Section 5 is the part worth reading before writing any of it: it establishes where a remesh may
   and may not be performed, which is what the design follows from.

All file/line references are to the state of the tree at the time of writing (`5fcc7ee`), except
the orientation fix itself, which landed in `d700b8e`.

---

## 1. What the detector does and does not give you

Detection is global (`BulkElementBase::detect_inverted_elements`, `src/elements.cpp:293`), off by
default, switched on from Python with `set_detect_inverted_elements(True)`. When on, every
integration point of every element whose mapping is square gets its signed `det(dx/ds)` tested, and
a non-positive one raises `oomph::InvertedElementError` (`src/elements.cpp:4676-4723`).

Two places catch it, both backported alongside the exception: the adaptive timestepping loop
(`src/thirdparty/oomph-lib/include/problem.cc:11464`), which rejects the step and halves `dt`, and
the arclength continuation loop (`:10988`), which rejects the step and scales `Ds` by 2/3. Both
mirror the `NewtonSolverError` handling immediately above them.

So the response to an inversion today is *always* "take a smaller step". Section 3.2 is a case where
that response cannot work, no matter how small the step gets.

One more thing the flag is not: it is global, while `RemeshWhen` is per-domain. Any per-domain
option that arms it therefore turns detection on everywhere and everyone pays the ~2% assembly cost.
Making it per-domain would mean threading a flag down to the element, which is not obviously worth
it — but it needs saying in the user-facing documentation rather than being discovered.

---

## 2. The test case

`dev_docs/examples/inverted_element_notch.py`. Unit square, 10×10 quads, `LaplaceSmoothedMesh`, plus
a diffusing scalar. The deformation is prescribed entirely through the boundary: a Gaussian notch of
linearly growing depth pushed into the top edge, with the other three edges free to slide along
themselves.

```python
notch = self.notch_rate * t * exp(-((xi[0] - 0.5) / self.notch_width) ** 2)
eqs += DirichletBC(mesh_y=1 - notch, mesh_x=True) @ "top"
```

Three properties make it the right test rather than merely a mesh that breaks:

- **The domain stays valid.** At the moment of folding the notch is only about 0.13 deep. What folds
  is the harmonic extension into a non-convex shape, not the shape — Radó–Kneser–Choquet guarantees
  injectivity of a harmonic extension onto a *convex* target and says nothing here. So the fold is a
  discretisation artefact that a remesh genuinely repairs, which a collapsing domain would not be.
- **The scalar field is load-bearing.** Without a non-geometric unknown there is no temporal error
  norm to drive adaptivity with, and nothing for a remesh to interpolate. It is not decoration.
- **The flags are independent** (`detect`, `adaptive`, `remesh`, `tight`), so the four behaviours in
  section 3 are the same problem seen four ways rather than four problems.

The script reports `min det(dx/ds)` over the 3×3 Gauss points, computed in Python from the nodal
positions. This deliberately duplicates the C++ test rather than reusing it, so that the two can
disagree — which is exactly what settled section 4. A first version of the diagnostic used the
signed area of the four corner nodes instead and put the fold at t = 0.344; that is wrong, because
the mid-side nodes of a biquadratic quad fold well before the corner quadrilateral does. The
integration-point figure agrees with the C++ detector to six digits.

---

## 3. What it shows

### 3.1 No detection: the run continues on a folded mesh

```
### step     t=0.140000  min detJ=2.6416e-04  inverted elems=0
### step     t=0.160000  min detJ=-5.5251e-05  inverted elems=2
### step     t=0.180000  min detJ=-3.7466e-04  inverted elems=2
```

The fold is at t ≈ 0.1565 and the run proceeds to t = 1 without complaint, `min detJ` marching
steadily further negative. This is the pre-detector behaviour and the reason the detector exists:
`J` as pyoomph computes it is `sqrt(det(g_ab))`, non-negative by construction, so an inside-out
element integrates perfectly happily.

### 3.2 Detection plus adaptive dt: rejection cannot recover

```
### step     t=0.156540  min detJ=-9.8388e-13  inverted elems=1
### step     t=0.156540  min detJ=-2.4712e-12  inverted elems=2
### step     t=0.156540  min detJ=-3.9586e-12  inverted elems=2
Tried to reduce dt to 7.27596e-13 which is less than the minimum dt (1e-12).
```

189 rejections, `dt` driven from 6e-3 to the floor, and `t` frozen at 0.156540 — the fold time to
six digits. The run dies.

**This is not a tuning failure and no threshold fixes it.** The deformation is prescribed as a
function of `t`, so the fold sits at a fixed *time*. Halving `dt` only makes `t` approach that time
more slowly; the mesh at t = 0.1565 + ε is folded for every ε > 0. The adaptive loop is being asked
to solve a problem it structurally cannot: it can only change how far it steps, and the obstacle is
not a step length.

That is the whole argument for the feature. Remeshing is not an optimisation here, it is the only
available response.

### 3.3 A caveat that shapes the design: transient inversions

With detection on but `dt` fixed, the run aborts at t = 0.12 — *before* the fold, at a time when the
converged mesh is perfectly valid. The inversion is raised on an intermediate Newton iterate: the
first residual assembly of the step has the top boundary already at its new prescribed position
while the interior is still at the old one, and that transient configuration has a squashed layer
under the notch that is briefly inside out.

Adaptive `dt` self-corrects this (a smaller step means the first iterate is closer to the answer),
which is why 412 of the rejections in the successful run of section 4.3 came from iterates that were
never solutions. A trigger that remeshes on the *first* inversion would therefore remesh hundreds of
times, each one an interpolation that costs accuracy, to cure something a smaller step already
cures. Section 6.2 is the consequence.

---

## 4. `GmshTemplate.fix_2d_orientation` — implemented

Running `remesh,detect,adaptive` before this fix died shortly after the first remesh, with all 128
elements of the *freshly remeshed* mesh reported as inverted and `dt` driven to the floor.

### 4.1 Root cause

Gmsh orients its elements after the surface normal, i.e. after the winding of the curve loop. A loop
assembled programmatically has no reason to come out one way rather than the other. Reading the
signed area of the loops that `Remesher2d` wrote over one run:

```
REMESH_000000  Curve Loop(1) = {1, 3, -4, -2}   signed area = -0.97469
REMESH_000001  Curve Loop(1) = {1, 3, -4,  2}   signed area = -0.95296
REMESH_000002  Curve Loop(1) = {1, -2, 4, -3}   signed area = +0.93485
REMESH_000003  Curve Loop(1) = {1, -3, 4, -2}   signed area = +0.92040
```

which matches the observed flip-flopping exactly: all elements inverted after remeshes 0 and 1, none
after 2 and 3.

The Gmsh→pyoomph permutations were never the problem. `perm = [0, 4, 1, 7, 8, 5, 3, 6, 2]` maps a
counter-clockwise Gmsh quad9 onto pyoomph's tensor ordering orientation-preservingly; it faithfully
reproduces a clockwise one as clockwise.

This had presumably been latent since the remesher was written, and was harmless for the same reason
section 3.1 is: `sqrt(det(g_ab))` does not care. It stopped being harmless the moment something
tested the *signed* determinant.

### 4.2 The fix

`GmshTemplate.fix_2d_orientation` (default `True`, `pyoomph/meshes/gmsh.py:383`) relabels clockwise
elements as they are constructed. The table of corner cycles and reversal permutations is at
`gmsh.py:1653`, applied through `_orient_2d` at each of the five `add_*_2d_*` call sites
(`gmsh.py:1702-1722`).

Reversing a quad is a transpose of its `(s0,s1)` index pair. Reversing a triangle swaps corners 1
and 2, which also swaps the mid-side nodes `mid(0,1)` and `mid(0,2)` while `mid(1,2)` stays put —
pyoomph's `MeshTemplateElementTriC2` uses the same convention as Gmsh's `triangle6`, which
`MeshTemplateElementTriC1::convert_for_C2_space` (`src/meshtemplate.cpp:337`) pins down.

Two cases need nothing of their own. Scott-Vogelius splits keep the orientation of their parent, so
flipping the parent covers all three sub-triangles. A `mirror_mesh` half is corrected as a side
effect, mirroring being orientation-reversing.

Meshes in a 3d nodal space are skipped: a surface has no orientation to fix, the same case the C++
check skips for not having a square mapping.

A counter, `num_flipped_2d_elements`, reports what the last construction touched. Deliberately no
print — this would otherwise fire on every remesh of every script.

### 4.3 Verification

`dev_docs/examples/gmsh_orientation_matrix.py` runs two geometries × four element types × three
arms, 24 cases, all passing. The geometries are a non-convex L, so the winding is not something Gmsh
can silently normalise away, and a disk, which covers the curved-boundary/macro-element path. The
clockwise arm is forced with `plane_surface(reversed_order=True)`.

```
L    tris2  reverse=False fix=True   flipped=  0  area=0.75000000 OK  int(u)=0.01325301 -   detector=ok
L    tris2  reverse=True  fix=False  flipped=  0  area=0.75000000 OK  int(u)=nan        -   detector=INVERTED
L    tris2  reverse=True  fix=True   flipped= 58  area=0.75000000 OK  int(u)=0.01325301 OK  detector=ok
```

Four things are being separated here, and the middle arm is what stops the test passing vacuously:

- correctly wound meshes are flipped zero times, so nothing existing changes;
- the clockwise control really is rejected by the detector, so the failure mode is real;
- the repaired arm's integrated area stays exact **and** its Poisson solution matches the
  counter-clockwise arm to 1e-8. This pair is what rules out a wrong mid-side permutation: a
  mid-node landing on the wrong edge moves both, and neither alone would catch it. Area alone would
  not distinguish a wrong reversal from a right one on a symmetric element.

On the original problem: the notch case with `remesh,detect,adaptive` now runs 207 steps to t = 1.0
with **no inverted element at any step**, remeshing 15 times and absorbing 412 inversion rejections
on the way — see section 3.3 for what those 412 are. (The commit message for
`d700b8e` says seven; that figure came from a truncated log and undercounts. Nothing else in it
depends on the number.) `docs/source/tutorial/ale/remeshing.py` is unchanged and still remeshes
three times.

### 4.4 What it does not change

Interface normals were the obvious way this could have silently altered results, face elements being
plausible candidates to inherit the bulk element's winding. The divergence theorem settles it:
`∫ n·x ds` over the L must equal `2·area = 1.5`.

```
NORMALS reverse=False fix=True   flipped=  0  int n.x ds=+1.500000 (expected +1.500000) OK
NORMALS reverse=True  fix=False  flipped=  0  int n.x ds=+1.500000 (expected +1.500000) OK
NORMALS reverse=True  fix=True   flipped= 58  int n.x ds=+1.500000 (expected +1.500000) OK
```

Correct in all three arms, the inside-out control included. So normals do not follow the bulk
winding, and the fix is a pure relabelling: no existing result moves.

---

## 5. Where a remesh may fire, and where it may not

This is the constraint everything below follows from. Three candidate places, only the last usable:

**Inside the assembly**, where the exception is raised (`src/elements.cpp:4721`). Mid element-loop,
with the assembler holding pointers into the mesh. Not a candidate.

**In `actions_after_newton_solve`** — where `force_remesh` is called *today*
(`pyoomph/generic/problem.py:2776-2787`). This looks like the natural place and is not, because
oomph-lib calls it from inside `newton_solve()`, hence from inside `adaptive_unsteady_newton_solve`,
which brackets the whole thing with a flat-index dof snapshot:

```cpp
for (unsigned i = 0; i < n_dof_local; i++) dofs_current[i] = dof(i);   // :11360, before the loop
...
for (unsigned i = 0; i < ni; i++) dof(i) = dofs_current[i];            // :11643, on rejection
```

A remesh in between changes the dof count and their ordering, so the restore writes stale values
into unrelated dofs, or runs off the end of the mesh. **This is a live hazard for the existing
quality-based `RemeshWhen`, not only for anything new.** It has not been observed biting: in the
test case the inversion exception aborts before that hook is reached, and the temporal error never
rose enough to provoke an error-based rejection (a run with the tolerance tightened to 1e-7 still
produced none, the deformation being linear in `t` and therefore integrated exactly by BDF2). It is
recorded here as read off the source, not as a reproduction.

**In Python, after the C++ call returns** (`problem.py:6090-6106`). Safe, and the only place where
the state is coherent.

There is a second, independent reason the remesh cannot happen at the moment of detection. When the
exception is raised the mesh has *already folded*, so its outline may be self-intersecting and Gmsh
has nothing sensible to mesh. What we want to remesh is the last accepted configuration — and the
C++ rejection path restores exactly that, dofs and time both, at `problem.cc:11633-11657`. That
restore is another thing only the C++ loop can do, and it must happen before any remesh.

So: **detect inside, reject inside, remesh outside.**

---

## 6. The plan — not implemented

### 6.1 Shape

`RemeshingOptions(on_inverted_element=True)` arms `set_detect_inverted_elements(True)` and registers
the domain, in the same way `on_invalid_triangulation` already registers one
(`pyoomph/equations/generic.py:392` and `:421`, with the marking at `:530`). The rejection stays in
C++ where the state can be restored. The remesh happens in `Problem.solve()`, after the C++ call has
returned, followed by a retry of the step.

### 6.2 When to escalate

Not on the first inversion — section 3.3. Halving `dt` is the *correct* first response to a
transient iterate fold and is already implemented; what distinguishes a real fold is that halving
stops helping.

So: keep the existing catch, count *consecutive* inversion-caused rejections, and escalate only past
a threshold. `k` consecutive rejections is exactly a `dt` reduction of `2^-k`, so the count is a
proxy for "shrinking the step is not helping". Suggested default `k = 3`, configurable on
`RemeshingOptions`.

The retry after a remesh must use the **originally requested** `dt`, not the reduced one — the
reduction is precisely what was just judged useless.

### 6.3 Mechanism

- C++: `adaptive_unsteady_newton_solve` counts consecutive inversion rejections. Past the threshold,
  and only when a remesh trigger is armed, it throws a distinct recoverable exception *after* the
  restore block has run, rather than halving again.
- Python: `Problem.solve()` catches it, calls `force_remesh` on the reporting domains, retries the
  step at the original `dt`. A retry cap makes a domain that stays folded fail loudly instead of
  looping.
- The fixed-`dt` path needs none of the C++ work: the exception already propagates to Python today,
  so it is the same handler with the escalation logic skipped.

### 6.4 The existing trigger should move too

Fold the hazard in section 5 into the same change: `actions_after_newton_solve` sets a pending flag
instead of calling `force_remesh`, and the Python wrapper performs it once the C++ call has
returned. That closes it for `RemeshWhen` generally. Check the interaction with
`doubly_adaptive_unsteady_newton_solve`, which does spatial adaptation after the temporal loop and
so has a second thing happening between "step accepted" and "back in Python".

### 6.5 MPI

The escalation decision has to be collective, because `force_remesh` is. `_agree_on_domains_to_remesh`
(`problem.py:2804`) already does this for the domain set, but "did any rank see an inversion" needs
the same treatment: an inversion is inherently local to whichever rank owns the folded element, and a
rank that saw none must not sail past the escalation point while the others remesh.

---

## 7. Open decisions

- **Threshold semantics.** Consecutive-rejection count (proposed, default 3), or an explicit "`dt`
  fell below this fraction of the requested `dt`"? The count is simpler to implement; the ratio is
  more legible in a log.
- **Scope of the first change.** Whether moving the existing quality-based trigger off the
  inside-the-C++-loop path (6.4) belongs in the same commit as the new option, or lands separately
  so the new option arrives in isolation.

---

## 8. Order of work

1. ~~`fix_2d_orientation`~~ — done, `d700b8e`. Nothing else can be tested without it.
2. Decide 7.1 and 7.2.
3. Defer the existing remesh out of the C++ loop (6.4), on its own if 7.2 says so. Testable with the
   quality-based trigger alone, no new option needed.
4. `on_inverted_element` on the fixed-`dt` path only (6.3, third bullet). No C++ change; the whole
   Python handler can be built and tested here.
5. The C++ escalation (6.3, first bullet) and the adaptive path.
6. MPI (6.5), with a worker test alongside the existing ones in `tests/`.

The orientation matrix (4.3) is worth promoting from `dev_docs/examples/` into `tests/` at some
point; it runs in well under a minute and is the kind of thing that silently regresses.
