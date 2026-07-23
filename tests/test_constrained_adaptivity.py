# Target test for combining local C1 dof-constraints (ConstrainFieldsToC1Space /
# ConstrainPositionsToC1Space) with genuine oomph-lib hanging nodes on
# adaptively refined (non-conforming) meshes.
#
# STATUS: currently pyoomph refuses this combination outright (a RuntimeError),
# so the test skips. See dev_docs/hanging_nodes_redesign.md section 5.5 for the
# architectural blocker uncovered while attempting the unification: correctly
# composing a C1 constraint with refinement hanging requires GLOBAL, per-node
# resolution of the constraint chain (a constrained "master" node's expansion
# lives in its home element and is not reachable from a neighbouring element,
# and pinning it severs the sensitivity chain that the hang needs). That is the
# named-HangInfo-with-flattening redesign, not a local per-element patch.
#
# When that lands, delete the skip: the assertion below is the correctness
# oracle. Poisson is LINEAR, so a single Newton step drives the residual to
# machine zero *iff* the assembled (analytic) Jacobian equals the true Jacobian;
# a wrong hang/constraint composition leaves a non-zero residual.

import numpy as np
import pytest
from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import CircularMesh
from pyoomph.equations.poisson import *


class ConstrainedPoissonProblem(Problem):
    def define_problem(self):
        self += CircularMesh(radius=1, segments=["NE"])
        eqs = PoissonEquation(source=1) + DirichletBC(u=0) @ "circumference"
        # Constraining to C1 needs a C1 space present in the bulk element.
        eqs += ScalarField("_dummyC1", space="C1") + DirichletBC(_dummyC1=0)
        # Constrain the C2 field u to C1 across the whole (adaptively refined)
        # domain, i.e. right on top of the genuine hanging nodes.
        eqs += ConstrainFieldsToC1Space("u")
        self += eqs @ "domain"


def test_constrained_adaptivity():
    with ConstrainedPoissonProblem() as problem:
        # Non-uniform refinement -> genuine hanging nodes overlapping the C1
        # constraint region.
        problem += RefineToLevel(2) @ "domain"
        problem += RefineToLevel(4) @ "domain/circumference"
        try:
            problem.solve()
        except RuntimeError as e:
            if "combined with genuine oomph-lib hanging nodes" in str(e):
                pytest.skip("constraint+refinement unification not yet implemented "
                            "(see dev_docs/hanging_nodes_redesign.md §5.5)")
            raise
        # Correctness oracle (linear problem): residual ~0 after the Newton step
        # certifies the analytic Jacobian, hence the composed hang/constraint
        # handling.
        res = float(np.max(np.abs(np.asarray(problem.get_residuals()))))
        assert res < 1e-9
