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

"""Find over- and under-constrained degrees of freedom from the Jacobian.

A redundant boundary condition - two constraints acting on the same degree of freedom,
e.g. a kinematic boundary condition and an :py:class:`~pyoomph.meshes.bcs.EnforcedDirichlet`
meeting at a contact line - makes the Jacobian exactly rank deficient. Newton then stalls
or wanders off, with nothing in the residual output pointing at the culprit.

The singular vectors say precisely which dofs and which equations are involved, and they
do not require a nullspace computation: power iteration on ``(J^T J)^-1 = J^-1 J^-T``
converges to the singular vector of the smallest singular value at two back-substitutions
per step, reusing one sparse LU. That LU is what a direct solver builds for every Newton
step anyway, so the diagnosis costs about as much as a single linear solve rather than the
O(n^3) of a dense SVD.

Two approaches that look cheaper do not work here and were tried:
:py:func:`scipy.sparse.csgraph.structural_rank` reports *full* rank for the contact-line
case (the two constraint rows have different sparsity patterns, so the dependency is
numerical, not structural) and takes far longer than the LU; and the smallest diagonal
entries of ``U`` point at arbitrary bulk dofs, because the fill-reducing ordering and
partial pivoting scatter the deficiency over the factorisation.
"""

import numpy
import scipy.sparse
import scipy.sparse.linalg

from ..typings import TYPE_CHECKING, Any, List, Optional, Tuple, NPFloatArray

if TYPE_CHECKING:
    from ..generic.problem import Problem


class SingularMode:
    """One singular triplet of the Jacobian, as returned by :py:func:`analyse_jacobian_singularity`.

    Attributes:
        sigma: The singular value, measured as ``||J v||`` with ``v`` normalised.
        right: Right singular vector. Its large entries are the dofs that are left undetermined.
        left: Left singular vector. Its large entries are the equations that are redundant.
    """

    def __init__(self, sigma: float, right: NPFloatArray, left: NPFloatArray):
        self.sigma = sigma
        self.right = right
        self.left = left

    def participation_ratio(self) -> float:
        """Roughly the number of dofs carrying the mode. A redundant constraint gives a
        handful, a genuine physical zero mode (a constant pressure, a rigid translation)
        gives a number comparable to the system size."""
        return 1.0 / float(numpy.sum(self.right**4))

    def leading_dofs(self, ntop: int = 6, cutoff: float = 0.05, left: bool = False) -> List[int]:
        """The equation indices carrying the mode, largest first, stopping at ``cutoff``
        times the largest entry."""
        vec = self.left if left else self.right
        idx = numpy.argsort(-numpy.absolute(vec))[:ntop]
        thresh = cutoff * abs(vec[idx[0]])
        return [int(i) for i in idx if abs(vec[i]) >= thresh]


class JacobianSingularityInfo:
    """Result of :py:func:`analyse_jacobian_singularity`."""

    def __init__(self, modes: List[SingularMode], norm: float, shift: float, shift_reason: str = ""):
        self.modes = modes
        self.norm = norm  #: ``||J||_inf``, used to make the singular values dimensionless
        self.shift = shift  #: Diagonal perturbation the analysis had to fall back on, 0 if none
        self.shift_reason = shift_reason  #: Why that perturbation was needed

    @property
    def sigmas(self) -> List[float]:
        return [m.sigma for m in self.modes]

    @property
    def gap(self) -> float:
        """``sigma_1/sigma_2``. The decisive quantity: a rank deficiency shows up as one
        *isolated* tiny singular value, whereas an ill-conditioned but regular Jacobian has
        a whole cluster of small ones."""
        if len(self.modes) < 2 or self.modes[1].sigma == 0.0:
            return float("nan")  # a second exact zero mode: rank deficient by more than one
        return self.modes[0].sigma / self.modes[1].sigma

    @property
    def is_rank_deficient(self) -> bool:
        """Whether the Jacobian is singular rather than merely ill-conditioned."""
        if self.modes[0].sigma < 1e-13 * self.norm:
            return True  # singular to working precision, so the gap does not have to decide
        if self.shift > 0:
            return True  # a shift was needed at all, which happens for singular matrices only
        # Above machine zero, a small singular value is evidence only if it stands alone: a
        # whole cluster of them is what a badly scaled discretisation produces anyway.
        return self.modes[0].sigma / self.norm < 1e-9 and self.gap < 1e-2


def _shifted_lu(J: Any, shift: float) -> Any:
    eye = scipy.sparse.identity(J.shape[0], format="csc", dtype=J.dtype)
    return scipy.sparse.linalg.splu((J.tocsc() + shift * eye).tocsc())


def _factorise(J: Any, norm: float, perturbation: float) -> Tuple[Any, float]:
    """LU of ``J``, falling back to ``J + shift*I`` when it is singular to working precision.

    SuperLU refuses an exactly singular matrix, which is exactly the case worth diagnosing.
    A diagonal shift restores factorisability and moves the offending singular value from 0
    to about ``shift``; the singular *vector* only moves by O(shift/gap), so the mode is
    still located correctly. The reported singular values are measured against the
    unperturbed ``J``, so the shift does not leak into them.
    """
    try:
        return scipy.sparse.linalg.splu(J.tocsc()), 0.0
    except RuntimeError as re:
        if "singular" not in str(re.args[0]).lower():
            raise
        return _shifted_lu(J, perturbation * norm), perturbation * norm


def smallest_singular_modes(J: Any, k: int = 2, iterations: int = 20, seed: int = 0,
                            perturbation: float = 1e-10) -> JacobianSingularityInfo:
    """The ``k`` smallest singular triplets of a sparse matrix by deflated inverse power iteration.

    Costs one sparse LU plus ``2*k*iterations`` triangular solves - i.e. it scales like a
    single linear solve, not like a nullspace computation.

    Args:
        J: The (square, sparse) matrix, e.g. from :py:meth:`~pyoomph.generic.problem.Problem.assemble_jacobian`.
        k: How many modes to extract. At least 2 are needed for the gap criterion.
        iterations: Power iterations per mode.
        seed: Seed of the random starting vector, so that repeated runs agree.
        perturbation: Relative diagonal shift used if ``J`` cannot be factorised at all.
    """
    n = J.shape[0]
    norm = float(scipy.sparse.linalg.norm(J, numpy.inf))
    nmodes = max(k, 2)

    def compute(lu: Any) -> List[SingularMode]:
        rng = numpy.random.default_rng(seed)
        modes: List[SingularMode] = []
        converged: List[NPFloatArray] = []
        for _ in range(nmodes):
            v = rng.standard_normal(n)
            for w in converged:
                v -= (w @ v) * w
            v /= numpy.linalg.norm(v)
            for _ in range(iterations):
                # (J^T J)^-1 v = J^-1 (J^-T v), the two solves reusing the same factorisation
                v = lu.solve(lu.solve(v, "T"), "N")
                for w in converged:
                    # (J^T J)^-1 is symmetric, so deflation is a plain orthogonal projection
                    v -= (w @ v) * w
                nrm = numpy.linalg.norm(v)
                if not numpy.isfinite(nrm) or nrm == 0.0:
                    raise RuntimeError("Inverse power iteration broke down - the LU factorisation "
                                       "is likely useless. Retry with a larger 'perturbation'.")
                v /= nrm
            u = lu.solve(v, "T")
            u /= numpy.linalg.norm(u)
            # ||J v|| rather than the growth factor of the iteration: it is a direct measurement
            # on the true J, hence unaffected by any shift applied to make the LU succeed
            converged.append(v)
            modes.append(SingularMode(float(numpy.linalg.norm(J @ v)), v, u))
        modes.sort(key=lambda m: m.sigma)
        return modes

    lu, shift = _factorise(J, norm, perturbation)
    reason = "the Jacobian could not be factorised at all" if shift > 0 else ""
    modes = compute(lu)

    # SuperLU factorises an exactly singular matrix rather than refusing it about as often as
    # not. The first mode still comes out right, but the deflated ones need not: the iteration
    # amplifies mode 0 by 1/sigma_1^2, so deflation leaves a residue of relative size
    # eps*(sigma_2/sigma_1)^2 behind, and once that reaches 1 it is all that is left. Shifting
    # caps the amplification, at the price of a second LU in a case that was pathological anyway.
    eps = float(numpy.finfo(numpy.float64).eps)
    if shift == 0.0 and nmodes > 1:
        separation = modes[1].sigma / modes[0].sigma if modes[0].sigma > 0 else numpy.inf
        if separation > 1.0 / numpy.sqrt(eps):
            # sqrt(eps)*sigma_2 is the smallest shift that brings that residue back to O(1)
            # digits; below eps*||J|| it would drown in the rounding of J itself
            shift = min(max(numpy.sqrt(eps) * modes[1].sigma, eps * norm), numpy.sqrt(eps) * norm)
            reason = "the zero mode lies too far below the rest of the spectrum to deflate"
            modes = compute(_shifted_lu(J, shift))

    return JacobianSingularityInfo(modes, norm, shift, reason)


def _node_positions(problem: "Problem") -> dict[int, Tuple[str, Tuple[float, ...]]]:
    """equation number -> (mesh name, node position), from a single scan over all bulk nodes.

    Interface fields such as Lagrange multipliers are stored as additional values on the bulk
    node, so scanning the bulk meshes covers them too.
    """
    from ..meshes.mesh import ODEStorageMesh
    result: dict[int, Tuple[str, Tuple[float, ...]]] = {}
    for mesh_name, mesh in problem._meshdict.items():
        if isinstance(mesh, ODEStorageMesh):
            continue
        for node in mesh.nodes():
            pos = tuple(round(node.x(xi), 12) for xi in range(node.ndim()))
            for vi in range(node.nvalue()):
                eq = node.eqn_number(vi)
                if eq >= 0:
                    result[eq] = (mesh_name, pos)
            posdata = node.variable_position_pt()
            for vi in range(posdata.nvalue()):
                eq = posdata.eqn_number(vi)
                if eq >= 0:
                    result[eq] = (mesh_name, pos)
    return result


def analyse_jacobian_singularity(problem: "Problem", k: int = 2, ntop: int = 6, cutoff: float = 0.05,
                                 iterations: int = 20, perturbation: float = 1e-10,
                                 J: Optional[Any] = None, quiet: bool = False) -> JacobianSingularityInfo:
    """Report which dofs and equations make the Jacobian singular.

    Use this when Newton refuses to converge and a redundant or missing boundary condition is
    suspected. The problem must be initialised, but need not have been solved: an
    over-constraint is a property of the Jacobian and is present right from the initial
    condition.

    The printed report gives, per mode, the dofs left undetermined (right singular vector),
    the equations that conflict (left singular vector), and every dof sitting on the node
    that carries the mode - which is where a doubly-applied boundary condition becomes
    visible as two constraint fields on one node.

    Args:
        problem: The (initialised) problem.
        k: Number of modes to report. Two are always computed, for the gap criterion.
        ntop: Maximum number of dofs listed per singular vector.
        cutoff: Skip entries below this fraction of the largest one.
        iterations: Power iterations per mode.
        perturbation: Relative diagonal shift used if the Jacobian cannot be factorised at all.
        J: Jacobian to analyse. Assembled from the problem if not given.
        quiet: Only return the result, print nothing.

    Returns:
        The computed modes and the verdict, see :py:class:`JacobianSingularityInfo`.
    """
    if J is None:
        J = problem.assemble_jacobian(with_residual=False)
    info = smallest_singular_modes(J, k=k, iterations=iterations, perturbation=perturbation)
    if quiet:
        return info

    types, names = problem.get_dof_description()

    def name_of(i: int) -> str:
        return names[types[i]] if types[i] >= 0 else "<unnamed dof " + str(i) + ">"

    positions = _node_positions(problem)

    def at(i: int) -> str:
        entry = positions.get(i)
        return "" if entry is None else "  at (" + ", ".join(f"{c:g}" for c in entry[1]) + ")"

    print("=" * 80)
    print(f"Jacobian singularity analysis: n={J.shape[0]}, nnz={J.nnz}, ||J||_inf={info.norm:.4e}")
    if info.shift > 0:
        print(f"  Analysed with a diagonal shift of {info.shift:.3e}, because {info.shift_reason}.")
    print("  smallest singular values: " + ", ".join(f"{s:.4e}" for s in info.sigmas))
    print(f"  sigma_min/||J|| = {info.sigmas[0]/info.norm:.3e}     gap sigma_1/sigma_2 = {info.gap:.3e}")
    if info.is_rank_deficient:
        print("  VERDICT: rank deficient. An isolated zero mode means a redundant constraint or an")
        print("           unconstrained dof, not mere ill-conditioning.")
    elif info.gap < 1e-2:
        print("  VERDICT: one isolated, very small singular value. Likely a redundant constraint,")
        print("           although not down at machine precision.")
    else:
        print("  VERDICT: no isolated zero mode. The Jacobian is ill-conditioned, but its small")
        print("           singular values form a cluster, which is discretisation rather than a BC.")

    for m, mode in enumerate(info.modes[:k]):
        print(f"\n  --- mode {m}: sigma = {mode.sigma:.4e}, carried by about "
              f"{mode.participation_ratio():.1f} dofs")
        for label, is_left, meaning in (("right", False, "dofs that are left undetermined"),
                                        ("left ", True, "equations that are redundant or conflicting")):
            print(f"    {label} singular vector - {meaning}:")
            vec = mode.left if is_left else mode.right
            for i in mode.leading_dofs(ntop, cutoff, left=is_left):
                print(f"       {vec[i]:+.4f}  #{i:<7d} {name_of(i)}{at(i)}")

        # Everything on the node carrying the mode: a boundary condition applied twice shows
        # up here as two constraint fields sharing one node.
        lead = int(numpy.argmax(numpy.absolute(mode.right)))
        if lead in positions:
            mesh_name, pos = positions[lead]
            print("    all dofs at (" + ", ".join(f"{c:g}" for c in pos) + f") in mesh {mesh_name}:")
            for i in sorted(i for i, entry in positions.items() if entry[1] == pos):
                mark = "  <== carries the zero mode" if i == lead else ""
                print(f"       #{i:<7d} |v|={abs(mode.right[i]):.3f}  {name_of(i)}{mark}")
    print("=" * 80)
    return info
