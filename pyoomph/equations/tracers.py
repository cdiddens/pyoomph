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

"""Passive tracer particles.

Tracers are advected by a prescribed vector field - usually the flow velocity - without feeding
back into it. Add :py:class:`TracerParticles` to a bulk domain and its particles move with the
field; add it to an interface and they are confined to that interface, advected by the *tangential*
part of the field and co-moving with the interface in the normal direction.

See ``dev_docs/tracers.md`` for the formulation and for what limits the accuracy.
"""

from .. import _pyoomph_core as _pyoomph

from ..expressions import evaluate_in_past, scale_factor
from ..expressions.generic import Expression, ExpressionOrNum
from ..generic.codegen import Equations, var, InterfaceEquations

from ..meshes.mesh import assert_spatial_mesh
from ..generic.mpi import get_mpi_nproc, get_mpi_min, get_mpi_max

from ..typings import *

import numpy

if TYPE_CHECKING:
    from ..meshes.mesh import AnyMesh, AnySpatialMesh
    # AnySpatialMesh and AnyMesh are string TypeAliases, so resolving an annotation that uses one
    # means evaluating the union THEY name in THIS module's namespace. sphinx_autodoc_typehints
    # executes this block for exactly that purpose, and without the concrete classes here it warns
    # "Cannot resolve forward reference ... name 'InterfaceMesh' is not defined" on every documented
    # signature that takes a mesh.
    from ..meshes.mesh import (InterfaceMesh, MeshFromTemplate1d, MeshFromTemplate2d,
                               MeshFromTemplate3d, ODEStorageMesh)
    from ..generic.codegen import EquationTree


# Number of nodal time-history levels the generated code is asked for. Two would only ever give a
# linear-in-time mesh configuration; the third makes the quadratic (BDF2-matched) one available, and
# is dropped at run time when the stored history cannot support it (an impulsive start, a first step).
_NUM_HISTORY_LEVELS = 3


###################################################################################################
# Seeding
###################################################################################################

class TracerSeed:
    """Base class of the strategies that decide where a :py:class:`TracerParticles` starts its
    particles. Every strategy checks that a candidate actually lies inside the mesh and reports how
    many did not, rather than silently creating particles in a hole of the domain."""

    def __init__(self, tag: int = 0):
        self.tag = tag

    def generate(self, mesh: "AnySpatialMesh", dim: int) -> numpy.ndarray:
        """Return an (N, dim) array of candidate positions, in nondimensional coordinates."""
        raise NotImplementedError

    #: Whether every process proposes the same candidates. When it does, seeding goes through
    #: ``add_tracers_collective``, which gives one particle per candidate with a
    #: partition-independent identity. A strategy that walks the local elements cannot promise this
    #: and sets it False, seeding each process's own share instead.
    global_candidates: bool = True

    def bounding_box(self, mesh: "AnySpatialMesh", dim: int) -> tuple[list[float], list[float]]:
        """Nondimensional bounding box of the mesh's nodes.

        Reduced over the processes, so that a distributed mesh gives the same box - and hence the
        same candidate lattice - everywhere, rather than one box per partition."""
        mins = [1e60] * dim
        maxs = [-1e60] * dim
        for n in mesh.nodes():
            for i in range(dim):
                x = n.x(i)
                mins[i] = min(mins[i], x)
                maxs[i] = max(maxs[i], x)
        if get_mpi_nproc() > 1:
            mins = [float(get_mpi_min(v)) for v in mins]
            maxs = [float(get_mpi_max(v)) for v in maxs]
        return mins, maxs


class TracerSeedPoints(TracerSeed):
    """Explicit positions, as an (N, dim) array or a list of points."""

    def __init__(self, positions: Any, tag: int = 0):
        super().__init__(tag)
        self.positions = positions

    def generate(self, mesh: "AnySpatialMesh", dim: int) -> numpy.ndarray:
        arr = numpy.atleast_2d(numpy.asarray(self.positions, dtype=float))
        if arr.shape[1] != dim:
            raise ValueError("TracerSeedPoints got " + str(arr.shape[1]) + "-dimensional positions for a " +
                             str(dim) + "-dimensional mesh")
        return arr


class TracerSeedGrid(TracerSeed):
    """An axis-aligned lattice with the given spacing over the mesh's bounding box.

    Works in 1, 2 and 3 dimensions. Candidates outside the mesh - in a hole, or outside a non-convex
    outline - are dropped and counted, which is why this is safe on a domain the bounding box does
    not describe.

    Args:
        spacing: distance between neighbouring candidates, dimensional if the problem is.
        bbox: ``(mins, maxs)`` to override the mesh bounding box.
        inset: how far to stay away from the bounding box faces, as a multiple of ``spacing``.
    """

    def __init__(self, spacing: ExpressionOrNum, bbox: tuple[Sequence[float], Sequence[float]] | None = None,
                 inset: float = 0.5, tag: int = 0):
        super().__init__(tag)
        self.spacing = spacing
        self.bbox = bbox
        self.inset = inset

    def generate(self, mesh: "AnySpatialMesh", dim: int) -> numpy.ndarray:
        d = float(self.spacing / mesh.get_problem().get_scaling("spatial"))
        if d <= 0:
            raise ValueError("TracerSeedGrid needs a positive spacing")
        if self.bbox is not None:
            mins, maxs = list(self.bbox[0]), list(self.bbox[1])
        else:
            mins, maxs = self.bounding_box(mesh, dim)
        axes: list[numpy.ndarray] = []
        for i in range(dim):
            lo = mins[i] + self.inset * d
            hi = maxs[i] - self.inset * d
            if hi < lo:
                lo = hi = 0.5 * (mins[i] + maxs[i])
            n = max(1, int(round((hi - lo) / d)) + 1)
            axes.append(numpy.linspace(lo, hi, n, endpoint=True))
        grids = numpy.meshgrid(*axes, indexing="ij")
        return numpy.stack([g.ravel() for g in grids], axis=-1)


class TracerSeedRandom(TracerSeed):
    """``n`` candidates drawn uniformly from the mesh's bounding box.

    Deterministic given ``rng_seed``, and - importantly under MPI - independent of how the mesh is
    partitioned: the candidates are drawn from the *globally* reduced bounding box, so every process
    proposes the same points and simply keeps the ones it owns.
    """

    def __init__(self, n: int, rng_seed: int = 0, tag: int = 0):
        super().__init__(tag)
        self.n = n
        self.rng_seed = rng_seed

    def generate(self, mesh: "AnySpatialMesh", dim: int) -> numpy.ndarray:
        mins, maxs = self.bounding_box(mesh, dim)
        rng = numpy.random.default_rng(self.rng_seed)
        out = rng.random((self.n, dim))
        for i in range(dim):
            out[:, i] = mins[i] + out[:, i] * (maxs[i] - mins[i])
        return out


class TracerSeedElement(TracerSeed):
    """One candidate at the centroid of each element (or ``per_element`` random points in it).

    Containment is free here, and the density follows the mesh rather than a bounding box, which is
    what you usually want on a graded or non-convex mesh.
    """

    #: Each process walks its own elements, so the candidates differ per process by construction.
    global_candidates = False

    def __init__(self, per_element: int = 1, rng_seed: int = 0, tag: int = 0):
        super().__init__(tag)
        self.per_element = per_element
        self.rng_seed = rng_seed

    def generate(self, mesh: "AnySpatialMesh", dim: int) -> numpy.ndarray:
        rng = numpy.random.default_rng(self.rng_seed)
        pts: list[list[float]] = []
        for ie in range(mesh.nelement()):
            e = mesh.element_pt(ie)
            nn = e.nnode()
            if nn == 0:
                continue
            if e.is_halo():
                continue  # somebody else's element; seeding it here would duplicate the particle
            corners = numpy.array([[e.node_pt(i).x(j) for j in range(dim)] for i in range(nn)])
            centroid = corners.mean(axis=0)
            pts.append(list(centroid))
            for _ in range(self.per_element - 1):
                # Convex combination of the nodes: always inside for a convex element, and close
                # enough to inside for a curved one that the containment check settles it.
                wts = rng.random(nn)
                wts /= wts.sum()
                pts.append(list(wts @ corners))
        if not pts:
            return numpy.zeros((0, dim))
        return numpy.array(pts)


class TracerSeedCallable(TracerSeed):
    """Positions from a user function ``fn(mesh) -> (N, dim) array``.

    Args:
        global_candidates: whether ``fn`` returns the same points on every process. Leave it True
            for a function of nothing but the global geometry; set it False if it inspects the
            process's own elements.
    """

    def __init__(self, fn: Callable[["AnySpatialMesh"], Any], tag: int = 0,
                 global_candidates: bool = True):
        super().__init__(tag)
        self.fn = fn
        self.global_candidates = global_candidates

    def generate(self, mesh: "AnySpatialMesh", dim: int) -> numpy.ndarray:
        return numpy.atleast_2d(numpy.asarray(self.fn(mesh), dtype=float))


###################################################################################################
# The equations
###################################################################################################

class TracerParticles(Equations):
    """Passive tracer particles advected by ``advection``.

    Where the equation is added decides what the particles do - there is no separate class and no
    flag for it:

      * on a **bulk** domain they follow the advection field itself. On a moving mesh the ALE term
        cancels analytically, so a particle in a mesh that moves under a zero advection field does
        not move at all;
      * on an **interface** (codimension 1) they are confined to it, advected by the *tangential*
        part of the field and co-moving with the interface in the normal direction.

    .. code-block:: python

        eqs  = NavierStokesEquations(...) + TracerParticles(seed=TracerSeedGrid(0.05))
        eqs += TracerParticles(seed=TracerSeedElement(), tracer_name="surf") @ "top"

    Args:
        advection: the velocity field the particles follow. Defaults to ``var("velocity")``.
        tracer_name: name of this collection on the mesh, for :py:meth:`~pyoomph.meshes.mesh.BaseMesh.get_tracers`.
        seed: where the particles start; see :py:class:`TracerSeed` and its subclasses. ``None``
            creates the collection but no particles, to be filled from Python.
        rtol, atol: tolerances of the adaptive sub-step controller, in units of the particle position.
        time_interpolation_order: order of the in-step interpolation of the mesh configuration.
            ``"auto"`` uses the best the stored nodal position history allows. This, not ``rtol``,
            is what caps the accuracy on a moving mesh.
        history_time: length of the rolling position history kept for trail plots. ``None`` keeps none.
        history_capacity: maximum number of history samples per particle.
        payloads: scalars integrated along each particle's path, as ``{name: source expression}``.
            Each source must be **dimensionless**, and is integrated over nondimensional time, so
            ``{"residence": 1}`` accumulates the time a particle has spent in the domain in units of
            the temporal scale. Multiply a dimensional rate by the appropriate scale yourself.
        statistics: print a one-line summary of each advection.
        fixed_substeps: if positive, take this many uniform sub-steps per timestep instead of
            letting the error controller choose. Only useful for order-of-convergence tests.
        max_substeps: hard upper bound on the sub-steps one particle may take within one timestep.
            Exceeding it raises - it is a backstop against a runaway, not a budget to be spent.
        max_migration_rounds: hard upper bound on the rounds of MPI migration and periodic
            re-injection per timestep. Exceeding it raises, since a particle still unfinished by
            then is bouncing between processes.
        max_periodic_wraps: how many times one particle may be re-injected at a periodic image
            within a single timestep.

    Every one of these is also a plain attribute of the underlying collection, so it can be changed
    at any time through :py:meth:`get_collection` - the constructor arguments only set the value the
    collection is (re-)bound with.
    """

    def __init__(self, advection: Expression = var("velocity"), *,
                 tracer_name: str = "tracers",
                 seed: TracerSeed | None = None,
                 rtol: float = 1e-8, atol: float = 1e-10,
                 time_interpolation_order: int | Literal["auto"] = "auto",
                 history_time: ExpressionOrNum | None = None,
                 history_capacity: int = 64,
                 payloads: dict[str, ExpressionOrNum] | None = None,
                 statistics: bool = False,
                 fixed_substeps: int = 0,
                 max_substeps: int = 1000000,
                 max_migration_rounds: int = 64,
                 max_periodic_wraps: int = 8):
        super(TracerParticles, self).__init__()
        if "@" in tracer_name or "/" in tracer_name:
            # Both are used to derive the names of the per-history-level and per-payload entries in
            # the generated code's function table.
            raise ValueError("A tracer name must not contain '@' or '/', but got " + repr(tracer_name))
        self.advection_expression = advection
        self.tracer_name = tracer_name
        self.seed = seed
        self.rtol = rtol
        self.atol = atol
        self.time_interpolation_order = time_interpolation_order
        self.history_time = history_time
        self.history_capacity = history_capacity
        self.payloads = dict(payloads) if payloads else {}
        self.statistics = statistics
        self.fixed_substeps = fixed_substeps
        self.max_substeps = max_substeps
        self.max_migration_rounds = max_migration_rounds
        self.max_periodic_wraps = max_periodic_wraps
        self._mesh: "AnySpatialMesh | None" = None
        self._last_advected_time: float | None = None

    # ----------------------------------------------------------------------------- collection

    def get_collection(self) -> _pyoomph.TracerCollection | None:
        """The C++ collection holding this equation's particles, or ``None`` before setup."""
        if (self._mesh is None) or (self.tracer_name not in self._mesh._tracers.keys()):  # type:ignore
            return None
        return self._mesh._tracers[self.tracer_name]  # type:ignore

    def _bind_mesh(self, mesh: "AnySpatialMesh"):
        """Attach to `mesh`, creating the collection only if this mesh does not have one yet.

        Creating it unconditionally - which is what this used to do - threw away every particle and
        every registered transfer interface on each call, and this is called more than once."""
        existing = mesh._tracers.get(self.tracer_name)  # type:ignore
        # Keyed on the DOMAIN NAME rather than on object identity: remeshing hands us a different
        # mesh object for the same domain, carrying the same collections over, and that is exactly
        # the case this guard must not fire on. What it is for is two TracerParticles claiming one
        # name on two genuinely different domains.
        if (existing is not None and self._mesh is not None
                and mesh.get_full_name() != self._mesh.get_full_name()):
            raise RuntimeError("Tracers named " + repr(self.tracer_name) +
                               " already exist on domain " + mesh.get_full_name())
        self._mesh = mesh
        if existing is None:
            coll = _pyoomph.TracerCollection(self.tracer_name)
            mesh._tracers[self.tracer_name] = coll  # type:ignore
        else:
            coll = existing
        coll._set_mesh(mesh)  # type:ignore
        coll._set_num_payloads(len(self.payloads))
        coll.rtol = self.rtol
        coll.atol = self.atol
        coll.history_capacity = self.history_capacity
        coll.time_interpolation_order = (-1 if self.time_interpolation_order == "auto"
                                         else int(self.time_interpolation_order))
        coll.fixed_substeps = self.fixed_substeps
        coll.max_substeps = self.max_substeps
        coll.max_migration_rounds = self.max_migration_rounds
        coll.max_periodic_wraps = self.max_periodic_wraps
        if self.history_time is not None:
            coll.history_window = float(self.history_time / mesh.get_problem().get_scaling("temporal"))
        return coll

    def before_assigning_equations_preorder(self, mesh: "AnyMesh"):
        if self._mesh is None:
            self._bind_mesh(assert_spatial_mesh(mesh))

    # ----------------------------------------------------------------------------- lifecycle

    def _init_output(self, eqtree: "EquationTree", continue_info: dict[str, Any] | None, rank: int):
        mesh = assert_spatial_mesh(eqtree._mesh)
        coll = self._bind_mesh(mesh)
        if continue_info is not None:
            return  # particles come from the state file, not from the seed
        if self.seed is not None and coll.nlocal() == 0:
            self._seed_particles(mesh, coll)
        coll._relocate_all(0)

    def _seed_particles(self, mesh: "AnySpatialMesh", coll: _pyoomph.TracerCollection):
        assert self.seed is not None
        dim = coll.get_coordinate_dimension()
        candidates = numpy.ascontiguousarray(self.seed.generate(mesh, dim), dtype=float)
        npay = len(self.payloads)
        n = len(candidates)
        if self.seed.global_candidates:
            # Collective: every process proposes the same candidates and exactly one keeps each, so
            # the particle set and its identities do not depend on the partitioning.
            outside = coll.add_tracers_collective(candidates.ravel().tolist(),
                                                  [self.seed.tag] * n, [0.0] * (n * npay))
        else:
            outside = 0
            for row in candidates:
                if coll.add_tracer([float(v) for v in row], self.seed.tag, [0.0] * npay) == 0:
                    outside += 1
        problem = mesh.get_problem()
        if outside and not problem.is_quiet():
            print("Tracers '" + self.tracer_name + "' on '" + mesh.get_full_name() + "': seeded " +
                  str(coll.nglobal()) + " particles, " + str(outside) + " of " + str(n) +
                  " candidates were outside the domain")

    def after_transient_solve(self):
        coll = self.get_collection()
        if coll is None:
            return
        assert self._mesh is not None
        problem = self._mesh.get_problem()
        # Even though this hook only fires for accepted timesteps, guard against advecting twice at
        # the same time: an adaptation recovery may re-solve, and a caller may drive solve() itself.
        now = float(problem.get_current_time(as_float=True, dimensional=False))
        if self._last_advected_time is not None and now <= self._last_advected_time:
            return
        self._last_advected_time = now
        coll._advect_all()
        if self.statistics and not problem.is_quiet():
            print("Tracers '" + self.tracer_name + "' on '" + self._mesh.get_full_name() + "': " +
                  coll.step_statistics())

    def after_remeshing(self, eqtree: "EquationTree"):
        # Remeshing replaces the mesh object, so self._mesh is stale by now and the collection still
        # points at the mesh that was thrown away. Re-bind to whatever the equation tree is holding -
        # by this point it has its elements, which _set_mesh needs - and then re-locate every
        # particle from its stored position, since the new mesh discretises the domain differently
        # and shares no elements with the old one.
        mesh = eqtree._mesh
        if mesh is not None:
            self._bind_mesh(assert_spatial_mesh(mesh))
        coll = self.get_collection()
        if coll is not None:
            coll._relocate_all(0)
        # This hook also fires after a state file was read, which may put the clock back - a
        # rollback, a restart from an earlier dump. The guard in after_transient_solve would then
        # refuse to advect until the run had caught up with the time it last saw, leaving the
        # particles frozen in place while the flow moved on. The restored particles have completed
        # the step ending at the restored time, which is exactly what the guard has to be told.
        if self._mesh is not None:
            problem = self._mesh.get_problem()
            self._last_advected_time = float(problem.get_current_time(as_float=True, dimensional=False))

    # ----------------------------------------------------------------------------- code generation

    def define_additional_functions(self):
        master = self._get_combined_element()
        cg = master._assert_codegen()
        scale = scale_factor("temporal") / scale_factor("spatial")

        # One entry per nodal time-history level. The C++ side blends them with the same Lagrange
        # weights it uses for the mesh configuration, which is what makes the two consistent; it
        # cannot be done here because the weights follow from t(0), t(1) and t(2), which are not
        # known at compile time and change with the step size.
        #
        # apply_on_others=True so that gradients and normals inside a past-level expression are
        # taken on the configuration that level belongs to, rather than on the current one.
        for k in range(_NUM_HISTORY_LEVELS):
            name = self.tracer_name if k == 0 else self.tracer_name + "@" + str(k)
            adv = self.advection_expression if k == 0 else evaluate_in_past(self.advection_expression, k, apply_on_others=True)
            cg._register_tracer_advection(name, scale * adv)
            for pi, (_, src) in enumerate(self.payloads.items()):
                psrc = src if k == 0 else evaluate_in_past(src, k, apply_on_others=True)
                # A payload source is a scalar, but the tracer machinery carries vectors; wrap it in
                # a one-component vector so the same registration path serves both.
                cg._register_tracer_advection(name + "/payload" + str(pi), _as_vector(psrc))

    def get_payload_names(self) -> list[str]:
        return list(self.payloads.keys())


def _as_vector(scalar: ExpressionOrNum) -> Expression:
    from ..expressions.generic import vector
    return vector([scalar])


class TracerTransferToInterface(InterfaceEquations):
    """Puts particles leaving the bulk through this interface onto the interface's own tracers.

    Where a bulk particle reaching the edge of its domain is otherwise dropped, this hands it to the
    collection living on the interface, where it continues as a confined particle: advected by the
    tangential part of the field and co-moving with the interface in the normal direction.

    The case it exists for is a free surface losing mass. The liquid recedes past the particle
    rather than the particle swimming out, so a parcel that reaches the surface belongs on it - and
    dropping it there both loses the particle and hides the accumulation, which on an evaporating
    droplet with a pinned contact line is the transport behind the coffee ring.

    .. code-block:: python

        d_eqs += TracerParticles(var("velocity"), seed=TracerSeedGrid(0.04*milli*meter))
        d_eqs += (TracerParticles(var("velocity"), tracer_name="surface", seed=...)
                  + TracerTransferToInterface()) @ "droplet_gas"

    The bulk domain and this interface must each carry a :py:class:`TracerParticles`. Nothing goes
    the other way: an interface particle cannot leave its interface, it can only reach the end of
    it, and it is pinned there.

    Args:
        interface_tracer_name: which collection on this interface receives them. Only needed when
            the interface carries more than one, where there is nothing to guess from.
    """
    required_parent_type = TracerParticles

    def __init__(self, interface_tracer_name: str | None = None):
        super(TracerTransferToInterface, self).__init__()
        self.interface_tracer_name = interface_tracer_name

    def _interface_equations(self) -> TracerParticles:
        """The TracerParticles living on this interface - the receiving side."""
        eqs = self.get_current_code_generator().get_equations()
        found = eqs.get_equation_of_type(TracerParticles, always_as_list=True)
        found = [e for e in found if isinstance(e, TracerParticles)]
        if self.interface_tracer_name is not None:
            found = [e for e in found if e.tracer_name == self.interface_tracer_name]
            if not found:
                raise RuntimeError("TracerTransferToInterface was asked for tracers named " +
                                   repr(self.interface_tracer_name) + " on " +
                                   self.get_mesh().get_full_name() + ", but there are none")
        if not found:
            raise RuntimeError("TracerTransferToInterface needs a TracerParticles on the interface "
                               + self.get_mesh().get_full_name() + " to hand the particles to")
        if len(found) > 1:
            raise RuntimeError("The interface " + self.get_mesh().get_full_name() + " carries " +
                               str(len(found)) + " tracer collections (" +
                               ", ".join(repr(e.tracer_name) for e in found) +
                               "), so TracerTransferToInterface needs interface_tracer_name to say "
                               "which one receives the particles")
        return found[0]

    def _wire(self):
        bulk = self.get_parent_equations()
        assert isinstance(bulk, TracerParticles)
        surf = self._interface_equations()
        bulkmesh = bulk._mesh
        bulkcol = bulk.get_collection()
        surfcol = surf.get_collection()
        if bulkmesh and bulkcol and surfcol:
            # Keyed on the BULK mesh's boundary index, exactly like a domain-to-domain transfer:
            # what the collection registers is "particles that leave me through this boundary go
            # there", and the boundary is the same object either way.
            bind = bulkmesh.get_boundary_index(self.get_mesh().get_name())
            bulkcol._set_transfer_interface(bind, surfcol)

    def before_assigning_equations_preorder(self, mesh: "AnyMesh"):
        self._wire()

    def _init_output(self, eqtree: "EquationTree", continue_info: dict[str, Any] | None, rank: int):
        # Re-register here too: both collections are bound in their own _init_output, which may run
        # after before_assigning_equations_preorder did.
        self._wire()


class TracerTransferAtInterface(InterfaceEquations):
    """Hands particles over between the tracer collections of two domains sharing this interface.

    Without this, a particle reaching the edge of its domain is dropped. Both sides must carry a
    :py:class:`TracerParticles` (they need not use the same advection field).

    Args:
        vice_versa: also register the transfer in the opposite direction.
    """
    required_parent_type = TracerParticles
    required_opposite_parent_type = TracerParticles

    def __init__(self, vice_versa: bool = True):
        super(TracerTransferAtInterface, self).__init__()
        self.vice_versa = vice_versa

    def _wire(self):
        mytr = self.get_parent_equations()
        othertr = self.get_opposite_parent_equations()
        assert isinstance(mytr, TracerParticles) and isinstance(othertr, TracerParticles)
        pmesh = mytr._mesh
        mycol = mytr.get_collection()
        othcol = othertr.get_collection()
        if pmesh and mycol and othcol:
            bind = pmesh.get_boundary_index(self.get_mesh().get_name())
            mycol._set_transfer_interface(bind, othcol)
            if self.vice_versa:
                opmesh = othertr._mesh
                if opmesh:
                    obind = opmesh.get_boundary_index(self.get_mesh().get_name())
                    othcol._set_transfer_interface(obind, mycol)

    def before_assigning_equations_preorder(self, mesh: "AnyMesh"):
        self._wire()

    def _init_output(self, eqtree: "EquationTree", continue_info: dict[str, Any] | None, rank: int):
        # Re-register here too: the parents bind their collections in their own _init_output, which
        # may run after before_assigning_equations_preorder did.
        self._wire()


class TracerPeriodicBoundaryCondition(InterfaceEquations):
    """Re-injects particles leaving through this interface at the periodic image of their position.

    The counterpart of :py:class:`TracerTransferAtInterface` for a domain that is periodic in
    itself: instead of handing the particle to a neighbouring domain, the collection takes it back
    at ``position + shift``, and it finishes the rest of its timestep from there - so a wrap costs
    nothing in accuracy and the particle keeps its identity and its payloads.

    .. code-block:: python

        # a channel of length L, periodic along x
        eqs += TracerPeriodicBoundaryCondition(vector(-L, 0)) @ "right"

    The shift is registered on the collection rather than on the boundary, and a particle that has
    left the mesh is offered every registered shift until one lands inside. So it does not matter
    which end of a periodic pair you attach this to, attaching it to both is harmless (the duplicate
    shift is dropped), and a particle leaving through a corner where two periodic directions meet is
    handled without any special case.

    Works under MPI: the periodic image of a point at one end of the domain is usually held by an
    entirely different process, which no halo exchange can reach, so a particle whose image is not
    local is offered to every process collectively and the one holding it takes it over.

    Args:
        shift: what to add to a particle's position to get its periodic image, as a vector
            expression or a sequence of numbers. Dimensional if the problem is.
        both_directions: also register ``-shift``, so that the pairing works whichever way a
            particle happens to leave. Turn it off only for a genuinely one-way injection.
    """
    required_parent_type = TracerParticles

    def __init__(self, shift: Expression | Sequence[ExpressionOrNum], both_directions: bool = True):
        super(TracerPeriodicBoundaryCondition, self).__init__()
        self.shift = shift
        self.both_directions = both_directions

    def _nondim_shift(self, mesh: "AnySpatialMesh", dim: int) -> list[float]:
        scal = mesh.get_problem().get_scaling("spatial")
        if isinstance(self.shift, Expression):
            comps = [self.shift[i] for i in range(dim)]
        else:
            comps = list(self.shift)
        if len(comps) != dim:
            raise ValueError("The periodic shift of the tracers must have " + str(dim) +
                             " components on this mesh, but got " + str(len(comps)))
        return [float(c / scal) for c in comps]

    def _wire(self):
        mytr = self.get_parent_equations()
        assert isinstance(mytr, TracerParticles)
        coll = mytr.get_collection()
        if coll is None or mytr._mesh is None:
            return
        shift = self._nondim_shift(mytr._mesh, coll.get_coordinate_dimension())
        coll._add_periodic_wrap(shift)  # type:ignore
        if self.both_directions:
            # 0.0 - v, not -v: the latter turns a zero component into -0.0, which then reads back
            # out of get_periodic_wraps() looking like something deliberate.
            coll._add_periodic_wrap([0.0 - v for v in shift])  # type:ignore

    def before_assigning_equations_preorder(self, mesh: "AnyMesh"):
        self._wire()

    def _init_output(self, eqtree: "EquationTree", continue_info: dict[str, Any] | None, rank: int):
        # As above: the parent binds its collection in its own _init_output, which may run after
        # before_assigning_equations_preorder did.
        self._wire()


from ..typings import _set_public_api
_set_public_api(globals())  # keep the typing helpers (Callable, List, ...) out of "from ... import *"
