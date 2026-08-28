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

# How vector- and tensor-valued output reaches the vtu.
#
# The writer does not hand ParaView one scalar array per component. It groups the components of a
# vector into a single 3-component array and those of a tensor into a VTK tensor array, off the
# _vectorfields / _tensorfields component grids that define_vector_field, define_tensor_field and
# add_local_function fill in. Nothing about that grouping was covered before this module.
#
# Two conventions are load-bearing and asserted here rather than assumed:
#
#   * VTK accepts an array as a TENSORS attribute only with EXACTLY 9 components (full, row-major)
#     or 6 (symmetric). 1, 3 and 4 are rejected. The symmetric packing is (xx,yy,zz,xy,yz,xz) --
#     checked against vtkTensorGlyph, not read off a document.
#
#   * A component that is symbolically zero is never registered as a field at all (see
#     FiniteElementCode::register_local_expression), so it arrives at the writer MISSING rather than
#     zero, and every gap has to be filled in place. Padding only after the first present component
#     had been collected slid the remaining ones down a slot, so vector(0,u) came out as (u,0)
#     instead of (0,u,0). test_vector_with_a_zero_first_component_is_not_shifted pins that.

import os

import numpy
import pytest

from pyoomph import Problem, Equations, DirichletBC
from pyoomph.expressions import (var, var_and_test, testfunction, weak, grad, vector, matrix, dyadic,
                                 identity_matrix, partial_t)
from pyoomph.equations.generic import LocalExpressions, ProjectExpression
from pyoomph.output.meshio import MeshFileOutput
from pyoomph.meshes.simplemeshes import RectangularQuadMesh, CuboidBrickMesh, LineMesh, PointMesh

meshio = pytest.importorskip("meshio")

#: Slot order of a 6-component VTK symmetric tensor, as (row, column) into the full 3x3.
SYMMETRIC_SLOTS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


# ----------------------------------------------------------------------------------------------
# harness
# ----------------------------------------------------------------------------------------------

class _Poisson(Equations):
    """A scalar field with a non-trivial gradient, so that grad(u) makes a usable test vector."""

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(grad(u), grad(v)) - weak(1, v))


class _OutputProblem(Problem):
    def __init__(self, extra, dim=2, axisymmetric=False, operator=None):
        super().__init__()
        self.extra = extra
        self.dim = dim
        self.axisymmetric = axisymmetric
        self.operator = operator

    def define_problem(self):
        if self.axisymmetric:
            self.set_coordinate_system("axisymmetric")
            # Away from the axis: nothing here needs r=0 and the 1/r terms only add noise.
            self += RectangularQuadMesh(N=3, size=[1, 1], lower_left=[1, 0], name="domain")
        elif self.dim == 2:
            self += RectangularQuadMesh(N=3, name="domain")
        else:
            self += CuboidBrickMesh(N=2, domain_name="domain")
        eqs = _Poisson() + MeshFileOutput(operator=self.operator) + DirichletBC(u=0) @ "left"
        eqs += self.extra
        self += eqs @ "domain"


def _read_one_vtu(outdir):
    written = sorted(f for f in os.listdir(outdir) if f.endswith(".vtu"))
    assert written, sorted(os.listdir(outdir))
    m = meshio.read(os.path.join(outdir, written[0]))
    return {k: numpy.asarray(v) for k, v in m.point_data.items()}, m


def _write_and_read(tmp_path, extra, dim=2, axisymmetric=False, operator=None):
    """Solves, writes one vtu and returns (point_data, mesh) of it."""
    with _OutputProblem(extra, dim=dim, axisymmetric=axisymmetric, operator=operator) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.solve()
        p.output()
        outdir = os.path.join(p.get_output_directory(), "domain")
    return _read_one_vtu(outdir)


def _full_from_symmetric(S):
    """Unpacks a (N,6) symmetric tensor array into (N,3,3)."""
    out = numpy.zeros((S.shape[0], 3, 3))
    for slot, (i, j) in enumerate(SYMMETRIC_SLOTS):
        out[:, i, j] = S[:, slot]
        out[:, j, i] = S[:, slot]
    return out


# ----------------------------------------------------------------------------------------------
# vectors
# ----------------------------------------------------------------------------------------------

def test_vector_local_expression_is_one_array(tmp_path):
    """grad(u) arrives as a single 3-component array, not as g_x/g_y/g_z scalars."""
    data, _ = _write_and_read(tmp_path, LocalExpressions(g=grad(var("u"))))
    assert data["g"].shape[1] == 3
    assert not any(k.startswith("g_") for k in data), sorted(data)


def test_vector_with_a_zero_first_component_is_not_shifted(tmp_path):
    """vector(0,u) must be (0,u,0).

    Its x component is symbolically zero and therefore never registered, so it reaches the writer
    missing. Filling gaps only once something had already been collected shifted the rest down a
    slot and produced a two-component (u,0)."""
    data, _ = _write_and_read(tmp_path, LocalExpressions(v=vector(0, var("u"))))
    v = data["v"]
    assert v.shape[1] == 3
    assert numpy.allclose(v[:, 0], 0)
    assert numpy.allclose(v[:, 1], data["u"])
    assert numpy.allclose(v[:, 2], 0)
    # ... and the middle slot is not trivially zero, or the test above proves nothing
    assert numpy.max(numpy.abs(data["u"])) > 1e-3


# ----------------------------------------------------------------------------------------------
# tensors from local expressions
# ----------------------------------------------------------------------------------------------

def test_symmetric_tensor_local_expression_uses_the_six_component_packing(tmp_path):
    g = grad(var("u"))
    sym = matrix([[g[i] * g[j] for j in range(3)] for i in range(3)])
    data, _ = _write_and_read(tmp_path, LocalExpressions(g=g, sym=sym))
    S = data["sym"]
    assert S.shape[1] == 6
    gv = data["g"]
    want = numpy.stack([gv[:, i] * gv[:, j] for i, j in SYMMETRIC_SLOTS], axis=1)
    assert numpy.allclose(S, want)


def test_non_symmetric_tensor_local_expression_uses_the_nine_component_packing(tmp_path):
    g = grad(var("u"))
    x = var("coordinate")
    nonsym = matrix([[g[i] * x[j] for j in range(3)] for i in range(3)])
    data, m = _write_and_read(tmp_path, LocalExpressions(g=g, nonsym=nonsym))
    T = data["nonsym"]
    assert T.shape[1] == 9
    gv, X = data["g"], numpy.asarray(m.points)
    want = numpy.stack([gv[:, i] * X[:, j] for i in range(3) for j in range(3)], axis=1)
    assert numpy.allclose(T, want)
    # genuinely non-symmetric, else the 9 vs 6 choice is untested
    assert not numpy.allclose(T[:, 1], T[:, 3])


# ----------------------------------------------------------------------------------------------
# tensors from define_tensor_field
# ----------------------------------------------------------------------------------------------

class _TensorField(Equations):
    """A tensor unknown forced to a prescribed value, so the dofs are known in closed form."""

    def __init__(self, symmetric, dim):
        super().__init__()
        self.symmetric = symmetric
        self.dim = dim

    def _names(self):
        idx = "xyz"
        return [[("sig_" + (idx[min(i, j)] + idx[max(i, j)] if self.symmetric else idx[i] + idx[j]))
                 for j in range(self.dim)] for i in range(self.dim)]

    def define_fields(self):
        self.define_tensor_field("sig", "C2", symmetric=self.symmetric)

    def source(self, x, i, j):
        """sig_ij as a closed form. Both variants are linear or bilinear in the coordinates, so a
        C2 space represents them exactly and the projection below is not an approximation."""
        if self.symmetric:
            return x[i] * x[j]          # genuinely symmetric, so the shared test function is exact
        return x[i] + 10 * (i + 1)      # differs under i<->j, so the 9-component path is required

    def define_residuals(self):
        rows = self._names()
        sig = matrix([[var(n) for n in r] for r in rows])
        sigtest = matrix([[testfunction(n) for n in r] for r in rows])
        x = var("coordinate")
        src = matrix([[self.source(x, i, j) for j in range(self.dim)] for i in range(self.dim)])
        self.add_residual(weak(sig - src, sigtest))


@pytest.mark.parametrize("dim", [2, 3])
def test_symmetric_tensor_field_is_grouped(tmp_path, dim):
    """Six components, and in 2d the absent z row and column are zero-filled rather than dropped."""
    eqs = _TensorField(True, dim)
    data, m = _write_and_read(tmp_path, eqs, dim=dim)
    assert data["sig"].shape[1] == 6
    T = _full_from_symmetric(data["sig"])
    X = numpy.asarray(m.points)
    for i in range(3):
        for j in range(3):
            want = eqs.source(X.T, i, j) if (i < dim and j < dim) else 0 * X[:, 0]
            assert numpy.allclose(T[:, i, j], want), (dim, i, j)


@pytest.mark.parametrize("dim", [2, 3])
def test_full_tensor_field_is_grouped(tmp_path, dim):
    """Nine components, since sig_ij differs from sig_ji, and again a zero-filled z block in 2d."""
    eqs = _TensorField(False, dim)
    data, m = _write_and_read(tmp_path, eqs, dim=dim)
    assert data["sig"].shape[1] == 9
    T = data["sig"].reshape(-1, 3, 3)
    X = numpy.asarray(m.points)
    for i in range(3):
        for j in range(3):
            want = eqs.source(X.T, i, j) if (i < dim and j < dim) else 0 * X[:, 0]
            assert numpy.allclose(T[:, i, j], want), (dim, i, j)
    # the transpose really does differ, or the 9- versus 6-component choice is untested
    assert not numpy.allclose(T[:, 0, 1], T[:, 1, 0])


# ----------------------------------------------------------------------------------------------
# ProjectExpression
# ----------------------------------------------------------------------------------------------

def _projection_source():
    """A deliberately non-symmetric tensor, so "tensor" and "symmetric_tensor" cannot agree."""
    g = grad(var("u"))
    x = var("coordinate")
    return matrix([[g[i] * (j + 1) + x[j] for j in range(3)] for i in range(3)])


def test_project_expression_full_tensor(tmp_path):
    data, m = _write_and_read(tmp_path, ProjectExpression(field_type="tensor", dim=3,
                                                         T=_projection_source()) + LocalExpressions(g=grad(var("u"))),
                              dim=3)
    T = data["T"]
    assert T.shape[1] == 9
    gv, X = data["g"], numpy.asarray(m.points)
    want = numpy.stack([gv[:, i] * (j + 1) + X[:, j] for i in range(3) for j in range(3)], axis=1)
    assert numpy.allclose(T, want, atol=2e-3)


def test_project_expression_symmetric_tensor_takes_the_symmetric_part(tmp_path):
    """The off-diagonal pair shares one test function, so a non-symmetric source is symmetrised
    rather than rejected."""
    extra = ProjectExpression(field_type="symmetric_tensor", dim=3, T=_projection_source())
    data, m = _write_and_read(tmp_path, extra + LocalExpressions(g=grad(var("u"))), dim=3)
    S = data["T"]
    assert S.shape[1] == 6
    gv, X = data["g"], numpy.asarray(m.points)
    full = numpy.stack([numpy.stack([gv[:, i] * (j + 1) + X[:, j] for j in range(3)], axis=1)
                        for i in range(3)], axis=1)
    sym = (full + numpy.transpose(full, (0, 2, 1))) / 2
    want = numpy.stack([sym[:, i, j] for i, j in SYMMETRIC_SLOTS], axis=1)
    assert numpy.allclose(S, want, atol=2e-3)
    # and it really is not the unsymmetrised tensor
    assert not numpy.allclose(S[:, 3], full[:, 0, 1], atol=2e-3)


def test_project_expression_tensor_defaults_to_the_nodal_dimension(tmp_path):
    """Without dim=3 a planar tensor field has no out-of-plane entry at all, so the z row and
    column of the written tensor are zero."""
    data, _ = _write_and_read(tmp_path, ProjectExpression(field_type="tensor", T=_projection_source()))
    T = data["T"].reshape(-1, 3, 3)
    assert numpy.allclose(T[:, 2, :], 0)
    assert numpy.allclose(T[:, :, 2], 0)
    assert numpy.max(numpy.abs(T[:, :2, :2])) > 1e-3


# ----------------------------------------------------------------------------------------------
# equations that register a whole tensor
# ----------------------------------------------------------------------------------------------

def test_maxwell_stress_is_one_tensor_including_its_zz_entry(tmp_path):
    """sigma = eps*(E (x) E) - I*eps*|E|^2/2, all nine entries.

    sigma_zz = -eps|E|^2/2 is nonzero in a planar problem, so a tensor truncated to the nodal
    dimension would put a false zero in that slot."""
    from pyoomph.equations.electrostatics import ElectricPotentialEquations
    from pyoomph.equations.electrohydrodynamics import MaxwellStressEquations
    from pyoomph.equations.navier_stokes import NavierStokesEquations

    extra = (NavierStokesEquations(dynamic_viscosity=1, mass_density=1)
             + ElectricPotentialEquations(permittivity=1, permittivity_scale=1,
                                          add_maxwell_stress_to_momentum=False)
             + MaxwellStressEquations(output_stress=True)
             + DirichletBC(phi=0) @ "left" + DirichletBC(phi=1) @ "right"
             + DirichletBC(velocity_x=0, velocity_y=0) @ "top"
             + DirichletBC(velocity_x=0, velocity_y=0) @ "bottom"
             + DirichletBC(velocity_x=0, velocity_y=0) @ "left"
             + DirichletBC(velocity_x=0, velocity_y=0) @ "right"
             + DirichletBC(pressure=0) @ "bottom/left")
    data, _ = _write_and_read(tmp_path, extra)
    S = data["maxwell_stress"]
    assert S.shape[1] == 6
    E = data["electric_field"]
    assert E.shape[1] == 3
    E2 = numpy.sum(E * E, axis=1)
    want = numpy.stack([E[:, i] * E[:, j] - (E2 / 2 if i == j else 0) for i, j in SYMMETRIC_SLOTS], axis=1)
    assert numpy.allclose(S, want)
    # the entry the old component loop never emitted
    assert numpy.allclose(S[:, 2], -E2 / 2)
    assert numpy.max(numpy.abs(S[:, 2])) > 1e-3


def _viscoelastic_extra(formulation):
    from pyoomph.equations.viscoelastic import ViscoelasticEquations, OldroydB

    class _ImposedVelocity(Equations):
        def define_fields(self):
            self.define_field_by_substitution("velocity", vector(var("coordinate_y"), 0),
                                              also_on_interface=True)

    return _ImposedVelocity() + ViscoelasticEquations(
        model=OldroydB(), relaxation_time=1, polymer_viscosity=1, formulation=formulation,
        add_polymer_stress_to_momentum=False, space="C1")


@pytest.mark.parametrize("formulation", ["log-conf", "conformation"])
def test_viscoelastic_outputs_grouped_tensors(tmp_path, formulation):
    data, _ = _write_and_read(tmp_path, _viscoelastic_extra(formulation))
    for name in ("conformation", "polymer_stress"):
        assert data[name].shape[1] == 6, (formulation, name, sorted(data))
    assert not any(k.startswith("polymer_stress_") for k in data), sorted(data)


def test_conformation_zz_is_one_not_zero(tmp_path):
    """C_zz=1 identically when the out-of-plane component is not solved for.

    define_tensor_field's Cartesian grid stops at the in-plane block, so the slot used to be padded
    with a zero -- a zero on that diagonal reads as a collapsed configuration."""
    data, _ = _write_and_read(tmp_path, _viscoelastic_extra("conformation"))
    C = _full_from_symmetric(data["conformation"])
    assert numpy.allclose(C[:, 2, 2], 1)


def test_axisymmetric_conformation_carries_the_azimuthal_component(tmp_path):
    """In axisymmetry the azimuthal unknown belongs in the zz slot: at phi=0 the azimuthal
    direction is the Cartesian z of the r-z plane."""
    data, _ = _write_and_read(tmp_path, _viscoelastic_extra("conformation"), axisymmetric=True)
    C = _full_from_symmetric(data["conformation"])
    assert numpy.max(numpy.abs(C[:, 2, 2] - 1)) > 1e-3


# ----------------------------------------------------------------------------------------------
# what VTK itself accepts
# ----------------------------------------------------------------------------------------------

def test_vtk_accepts_the_written_tensors(tmp_path):
    """A tensor attribute takes 6 or 9 components and nothing else, so the packing above is not a
    private convention of this test module."""
    vtk = pytest.importorskip("vtk")
    g = grad(var("u"))
    x = var("coordinate")
    extra = LocalExpressions(sym=matrix([[g[i] * g[j] for j in range(3)] for i in range(3)]),
                             nonsym=matrix([[g[i] * x[j] for j in range(3)] for i in range(3)]))
    with _OutputProblem(extra) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.solve()
        p.output()
        outdir = os.path.join(p.get_output_directory(), "domain")
    written = sorted(f for f in os.listdir(outdir) if f.endswith(".vtu"))
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(os.path.join(outdir, written[0]))
    reader.Update()
    pd = reader.GetOutput().GetPointData()
    for name, ncomp in (("sym", 6), ("nonsym", 9)):
        arr = pd.GetArray(name)
        assert arr is not None, [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
        assert arr.GetNumberOfComponents() == ncomp
        pd.SetActiveTensors(name)
        active = pd.GetTensors()
        assert active is not None and active.GetName() == name


# ----------------------------------------------------------------------------------------------
# extrusions
#
# Both extrusion operators build an explicit rotation for vector fields and, until this was written,
# nothing at all for tensors: a component with no entry in field_operators falls through to a plain
# numpy.tile, so T_xx at phi=90 degrees still held T_rr and the extruded tensor was wrong everywhere
# off the starting angle. The rotation is Q T Q^T with Q the same basis matrix the vector operators
# apply one index at a time.
#
# The invariants below are the cheap way to pin that: a wrong Q breaks all of them at once, and none
# of them needs a hand-computed reference.
# ----------------------------------------------------------------------------------------------

from pyoomph.meshes.meshdatacache import (MeshDataRotationalExtrusion, MeshDataCartesianExtrusion,
                                          MeshDataCombineWithEigenfunction)

#: Q at each extrusion row: the images of the tensor's (r, z, phi) slots in Cartesian (x, y, z).
def _rotation_matrices(nrows):
    phis = numpy.linspace(0, 2 * numpy.pi, nrows, endpoint=False)
    c, s = numpy.cos(phis), numpy.sin(phis)
    zero, one = numpy.zeros(nrows), numpy.ones(nrows)
    Q = numpy.stack([numpy.stack([c, zero, -s], axis=1),
                     numpy.stack([s, zero, c], axis=1),
                     numpy.stack([zero, one, zero], axis=1)], axis=1)
    return phis, Q


def test_extrusion_keeps_local_expressions_at_the_right_length(tmp_path):
    """A local expression used to survive no extrusion at all.

    It lives in nodal_local_exprs and is evaluated against the ORIGINAL mesh, so it kept the original
    node count while nodal_values was expanded, and meshio refused the result outright with
    "len(points) = 784, but len(point_data[...]) = 49". That hit scalars and vectors as much as
    tensors, i.e. every extrusion of an output carrying a LocalExpressions."""
    data, m = _write_and_read(tmp_path, LocalExpressions(g=grad(var("u"))), axisymmetric=True,
                              operator=MeshDataRotationalExtrusion(n_segments=8))
    assert len(data["g"]) == len(m.points)
    assert data["g"].shape[1] == 3


def test_rotational_extrusion_rotates_a_tensor_into_the_cartesian_frame(tmp_path):
    """Constant Cartesian tensors stay constant, and e_r (x) e_r becomes the phi-dependent projector.

    Written in the tensor's own (r, z, phi) slots, so e_z (x) e_z is diag(0,1,0) going in and has to
    be diag(0,0,1) coming out - which is exactly what a missing rotation would get wrong."""
    extra = LocalExpressions(ez=matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                             ident=matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                             er=matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]]))
    data, m = _write_and_read(tmp_path, extra, axisymmetric=True,
                              operator=MeshDataRotationalExtrusion(n_segments=8))
    X = numpy.asarray(m.points)

    want_ez = numpy.zeros((3, 3))
    want_ez[2, 2] = 1
    assert numpy.allclose(_full_from_symmetric(data["ez"]), want_ez)
    assert numpy.allclose(_full_from_symmetric(data["ident"]), numpy.eye(3))

    # e_r (x) e_r is the projector onto (cos phi, sin phi, 0), the one case that varies with phi
    radius = numpy.hypot(X[:, 0], X[:, 1])
    off_axis = radius > 1e-9
    c, s = X[off_axis, 0] / radius[off_axis], X[off_axis, 1] / radius[off_axis]
    want_er = numpy.zeros((int(off_axis.sum()), 3, 3))
    want_er[:, 0, 0], want_er[:, 1, 1] = c * c, s * s
    want_er[:, 0, 1] = want_er[:, 1, 0] = c * s
    assert numpy.allclose(_full_from_symmetric(data["er"])[off_axis], want_er)


def test_rotational_extrusion_matches_q_t_qt_and_keeps_the_invariants(tmp_path):
    """The extruded tensor against Q T Q^T computed here, plus trace and determinant.

    A non-symmetric source, so the nine-component path is the one exercised and a transposed Q would
    show up rather than cancelling."""
    g, x = grad(var("u")), var("coordinate")
    T = matrix([[g[i] * (j + 1) + x[j] + 10 * (i + 1) for j in range(3)] for i in range(3)])
    flat, _ = _write_and_read(tmp_path / "flat", LocalExpressions(T=T), axisymmetric=True)
    ext, _ = _write_and_read(tmp_path / "ext", LocalExpressions(T=T), axisymmetric=True,
                             operator=MeshDataRotationalExtrusion(n_segments=8))
    Tf = flat["T"].reshape(-1, 3, 3)
    Te = ext["T"].reshape(-1, 3, 3)
    assert Tf.shape[1:] == (3, 3) and flat["T"].shape[1] == 9
    nnodes = Tf.shape[0]
    nrows = Te.shape[0] // nnodes
    assert nrows > 1 and Te.shape[0] == nrows * nnodes
    _phis, Q = _rotation_matrices(nrows)

    want = numpy.einsum("rai,nij,rbj->rnab", Q, Tf, Q).reshape(-1, 3, 3)
    assert numpy.allclose(Te, want)
    assert not numpy.allclose(Te[:, 0, 1], Te[:, 1, 0])      # still non-symmetric
    assert numpy.allclose(numpy.trace(Te, axis1=1, axis2=2).reshape(nrows, nnodes),
                          numpy.trace(Tf, axis1=1, axis2=2))
    assert numpy.allclose(numpy.linalg.det(Te).reshape(nrows, nnodes), numpy.linalg.det(Tf))


def test_rotational_extrusion_keeps_a_symmetric_tensor_at_six_components(tmp_path):
    """Q T Q^T preserves symmetry exactly, so the packing should not widen to nine."""
    g = grad(var("u"))
    extra = LocalExpressions(sym=matrix([[g[i] * g[j] for j in range(3)] for i in range(3)]))
    data, _ = _write_and_read(tmp_path, extra, axisymmetric=True,
                              operator=MeshDataRotationalExtrusion(n_segments=8))
    assert data["sym"].shape[1] == 6


# ----------------------------------------------------------------------------------------------
# eigenmodes under the extrusions. Some of these need a complex PETSc on PYTHONPATH, see CLAUDE.md.
# ----------------------------------------------------------------------------------------------

def _complex_petsc_skip_reason():
    """Importing slepc4py is not enough: a REAL build imports fine and then raises out of the solve.

    The tensor relaxation below stays real, so it only needs slepc4py at all; the mode-convention
    tests carry an I*m*u term and hand the solver a genuinely complex pair, which a real
    PETSc/SLEPc refuses with a RuntimeError rather than a skip."""
    try:
        import slepc4py  # type:ignore  # noqa: F401
        from petsc4py import PETSc  # type:ignore
    except Exception:
        return "petsc4py/slepc4py not available (PYTHONPATH must carry a complex PETSc build)"
    if PETSc.ScalarType is not numpy.complex128:
        return "the PETSc on PYTHONPATH is a real build; a complex eigenproblem needs a complex one"
    return None


class _TensorRelaxation(Equations):
    """dt(sig) = -sig + source: a tensor unknown with a non-empty mass matrix, so it has eigenmodes."""

    def define_fields(self):
        self.define_tensor_field("sig", "C2", symmetric=False)

    def define_residuals(self):
        sig, sigtest = var_and_test("sig")
        x = var("coordinate")
        source = matrix([[x[0] * x[1] + 0.5 * (i + 1) for _j in range(3)] for i in range(3)])
        self.add_residual(weak(partial_t(sig) + sig - source, sigtest))


class _EigenTensorProblem(Problem):
    def __init__(self, operator, axisymmetric):
        super().__init__()
        self.operator = operator
        self.axisymmetric = axisymmetric

    def define_problem(self):
        if self.axisymmetric:
            self.set_coordinate_system("axisymmetric")
            self += RectangularQuadMesh(N=2, size=[1, 1], lower_left=[1, 0], name="domain")
        else:
            self += RectangularQuadMesh(N=2, name="domain")
        self += (_TensorRelaxation()
                 + MeshFileOutput(operator=self.operator, eigenvector=0, eigenmode="real")) @ "domain"


def _eigen_tensor_output(tmp_path, operator, axisymmetric, **solve_kwargs):
    with _EigenTensorProblem(operator, axisymmetric) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.setup_for_stability_analysis(azimuthal_stability=axisymmetric,
                                       additional_cartesian_mode=not axisymmetric,
                                       analytic_hessian=False)
        p.solve()
        p.solve_eigenproblem(1, **solve_kwargs)
        p.output()
        outdir = os.path.join(p.get_output_directory(), "domain")
    return _read_one_vtu(outdir)[0]


@pytest.mark.parametrize("m", [0, 1, 2])
def test_rotational_extrusion_of_an_azimuthal_eigen_tensor(tmp_path, m):
    """Eigen tensor == Q (cos(m phi) Re + sin(m phi) Im) Q^T.

    The mode reconstruction and the basis rotation are fused into one outer product, so this checks
    them together against the two halves as they come out unextruded. A tensor unknown rather than a
    local expression, since only a field has real and imaginary halves to recombine."""
    pytest.importorskip("slepc4py", reason="the azimuthal eigensolve needs slepc4py")
    combine = MeshDataCombineWithEigenfunction(0)
    flat = _eigen_tensor_output(tmp_path / "flat", combine, True, azimuthal_m=m)
    ext = _eigen_tensor_output(tmp_path / "ext",
                               combine + MeshDataRotationalExtrusion(n_segments=8), True,
                               azimuthal_m=m)
    Re = flat["EigenRe_sig"].reshape(-1, 3, 3)
    Im = flat["EigenIm_sig"].reshape(-1, 3, 3)
    E = ext["Eigen_sig"].reshape(-1, 3, 3)
    nnodes = Re.shape[0]
    nrows = E.shape[0] // nnodes
    phis, Q = _rotation_matrices(nrows)
    # minus on the imaginary part: Re[T_hat*exp(I*m*phi)], see the sign-convention section below
    T = (numpy.cos(m * phis)[:, None, None, None] * Re[None]
         - numpy.sin(m * phis)[:, None, None, None] * Im[None])
    want = numpy.einsum("rai,rnij,rbj->rnab", Q, T, Q).reshape(-1, 3, 3)
    scale = max(numpy.max(numpy.abs(want)), 1.0)
    assert numpy.max(numpy.abs(want)) > 1e-3, "the eigenfunction is trivial, so nothing is tested"
    assert numpy.allclose(E, want, atol=1e-9 * scale)
    # no loose components left over: the caller's scalar loop reconstructs every one of them under
    # the result prefix, and the azimuthal names this rotation does not reuse have to be dropped
    assert not any(k.startswith("Eigen_sig_") for k in ext), sorted(ext)


@pytest.mark.parametrize("k", [1.5, 3.0])
def test_cartesian_extrusion_of_a_normal_mode_eigen_tensor(tmp_path, k):
    """The Cartesian extrusion translates rather than turns, so Q is the identity there.

    The base state therefore needs nothing a plain tile would not do -- asserted, so that a future
    rotation added by mistake shows up -- while the eigenmode still has to be recombined from its
    two halves."""
    pytest.importorskip("slepc4py", reason="the normal-mode eigensolve needs slepc4py")
    combine = MeshDataCombineWithEigenfunction(0)
    flat = _eigen_tensor_output(tmp_path / "flat", combine, False, normal_mode_k=k)
    ext = _eigen_tensor_output(tmp_path / "ext",
                               combine + MeshDataCartesianExtrusion(n_segments=6), False,
                               normal_mode_k=k)
    Re = flat["EigenRe_sig"].reshape(-1, 3, 3)
    Im = flat["EigenIm_sig"].reshape(-1, 3, 3)
    E = ext["Eigen_sig"].reshape(-1, 3, 3)
    S = ext["sig"].reshape(-1, 3, 3)
    nnodes = Re.shape[0]
    nrows = E.shape[0] // nnodes
    # numperiods=1 and phase=0, i.e. exactly one wavelength
    zs = numpy.linspace(0, 2 * numpy.pi / k, nrows, endpoint=True)
    want = (numpy.cos(k * zs)[:, None, None, None] * Re[None]
            - numpy.sin(k * zs)[:, None, None, None] * Im[None]).reshape(-1, 3, 3)
    assert numpy.max(numpy.abs(want)) > 1e-3
    assert numpy.allclose(E, want, atol=1e-9)
    assert numpy.allclose(S, numpy.tile(flat["sig"].reshape(-1, 3, 3), (nrows, 1, 1)))


# ----------------------------------------------------------------------------------------------
# lower-dimensional base meshes
#
# A tensor's slots are positional, so what each one MEANS depends on the mesh, and both extrusions
# have branches for a base that is one- or zero-dimensional. Two conventions decide the mapping:
#
#   * Axisymmetric: (r, z, phi) on a bulk mesh but (r, phi) on a radial one -- there is no axial
#     direction at all there. That is what define_tensor_field hands out ("_aa" at [1][1], not
#     [2][2]) and what directional_tensor_derivative assumes with azi = 2 if ndim == 2 else 1.
#   * Cartesian normal mode: the extra direction is appended, so it sits at slot ndim -- which is
#     exactly the axis the extrusion adds, at index ndim. Hence the identity there, at every base
#     dimension.
# ----------------------------------------------------------------------------------------------

class _ConstantScalar(Equations):
    """Enough of a field to make a solvable problem on a mesh with no interior."""

    def define_fields(self):
        self.define_scalar_field("u", "C2")

    def define_residuals(self):
        u, v = var_and_test("u")
        self.add_residual(weak(u - 1, v))


class _LowDimProblem(Problem):
    def __init__(self, mesh, coordsys, operator, extra):
        super().__init__()
        self.mesh = mesh
        self.coordsys = coordsys
        self.operator = operator
        self.extra = extra

    def define_problem(self):
        if self.coordsys is not None:
            self.set_coordinate_system(self.coordsys)
        self += self.mesh
        self += (_ConstantScalar() + self.extra
                 + MeshFileOutput(operator=self.operator)) @ "domain"


def _low_dim_output(tmp_path, mesh, coordsys, operator, extra):
    with _LowDimProblem(mesh, coordsys, operator, extra) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.solve()
        p.output()
        outdir = os.path.join(p.get_output_directory(), "domain")
    return _read_one_vtu(outdir)


def test_radial_mesh_rotational_extrusion_uses_the_r_phi_slot_layout(tmp_path):
    """On a radial mesh slot 1 is the AZIMUTHAL direction, not the axial one.

    Rotating it with the bulk (r,z,phi) matrix sends it to z instead, so e_phi (x) e_phi came out as
    the constant e_z (x) e_z rather than turning with phi. Both projectors are checked, plus the
    off-diagonal, which no diagonal-only test would catch."""
    extra = LocalExpressions(er=matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
                             ep=matrix([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
                             mix=matrix([[0, 1, 0], [1, 0, 0], [0, 0, 0]]))
    data, m = _low_dim_output(tmp_path, LineMesh(N=3, minimum=1, size=1, name="domain"),
                              "axisymmetric", MeshDataRotationalExtrusion(n_segments=8), extra)
    X = numpy.asarray(m.points)
    radius = numpy.hypot(X[:, 0], X[:, 1])
    c, s = X[:, 0] / radius, X[:, 1] / radius
    assert len(numpy.unique(numpy.round(c, 6))) > 1, "phi never varies, so nothing is tested"
    zero = numpy.zeros_like(c)
    r_hat = numpy.stack([c, s, zero], axis=1)
    phi_hat = numpy.stack([-s, c, zero], axis=1)
    assert numpy.allclose(_full_from_symmetric(data["er"]), numpy.einsum("ni,nj->nij", r_hat, r_hat))
    assert numpy.allclose(_full_from_symmetric(data["ep"]),
                          numpy.einsum("ni,nj->nij", phi_hat, phi_hat))
    assert numpy.allclose(_full_from_symmetric(data["mix"]),
                          numpy.einsum("ni,nj->nij", r_hat, phi_hat)
                          + numpy.einsum("ni,nj->nij", phi_hat, r_hat))


@pytest.mark.parametrize("mesh_kind", ["line", "point"])
def test_cartesian_extrusion_of_a_low_dimensional_mesh(tmp_path, mesh_kind):
    """The extra direction lands on the axis the extrusion adds: y for a line, x for a point.

    That is the identity mapping in both cases, because the coordinate system appends the extra
    direction at slot ndim and the extrusion adds its axis at index ndim as well."""
    mesh = (LineMesh(N=3, minimum=1, size=1, name="domain") if mesh_kind == "line"
            else PointMesh(domain_name="domain"))
    slot = 1 if mesh_kind == "line" else 0
    extra = LocalExpressions(ident=matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                             extra_dir=matrix([[1 if (i == slot and j == slot) else 0
                                                for j in range(3)] for i in range(3)]))
    data, _ = _low_dim_output(tmp_path, mesh, None,
                              MeshDataCartesianExtrusion(n_segments=4, use_k_for_length=False), extra)
    assert numpy.allclose(_full_from_symmetric(data["ident"]), numpy.eye(3))
    want = numpy.zeros((3, 3))
    want[slot, slot] = 1
    assert numpy.allclose(_full_from_symmetric(data["extra_dir"]), want)


# ----------------------------------------------------------------------------------------------
# the sign convention of the mode reconstruction
#
# pyoomph stores (Re,Im) of the complex amplitude of exp(I*m*phi), so the physical field is
# Re[u*exp(I*m*phi)] = cos(m phi)*Re - SIN(m phi)*Im, with phi the RIGHT-handed azimuth about the
# axial direction. Both extrusions used to write +sin for scalars, and the rotational one also
# negated phi_hat in its eigen vector, the two errors there composing into a clean mirror image --
# which still looks like a plausible mode, which is why it stood.
#
# The reference below is the coordinate system itself, not another reconstruction: d/dphi of a
# perturbation is a factor I*m, so for w = r*grad(u)_phi the stored amplitudes must satisfy
# Re_w = -m*Im_u and Im_w = +m*Re_u. Anchoring on that is what makes this test able to fail when
# every reconstruction in the file agrees with each other and all of them are wrong.
# ----------------------------------------------------------------------------------------------

class _ModeConvention(Equations):
    """u with a non-trivial eigenmode, plus w = r*grad(u)_phi, which must equal I*m*u."""

    def __init__(self, azimuthal):
        super().__init__()
        self.azimuthal = azimuthal

    def define_fields(self):
        self.define_scalar_field("u", "C2")
        self.define_scalar_field("w", "C2")

    def define_residuals(self):
        u, ut = var_and_test("u")
        w, wt = var_and_test("w")
        x = var("coordinate")
        self.add_residual(weak(partial_t(u), ut) + weak(grad(u), grad(ut)) - weak(x[0] * x[1], ut))
        # slot 2 is the azimuthal direction on an axisymmetric bulk mesh and the extra one under a
        # Cartesian normal mode; the 1/r of the azimuthal gradient is undone by the factor r
        azimuthal_gradient = x[0] * grad(u)[2] if self.azimuthal else grad(u)[2]
        self.add_residual(weak(w - azimuthal_gradient, wt))


class _ModeConventionProblem(Problem):
    def __init__(self, azimuthal, operator):
        super().__init__()
        self.azimuthal = azimuthal
        self.operator = operator

    def define_problem(self):
        if self.azimuthal:
            self.set_coordinate_system("axisymmetric")
            self += RectangularQuadMesh(N=3, size=[1, 1], lower_left=[1, 0], name="domain")
        else:
            self += RectangularQuadMesh(N=3, name="domain")
        self += (_ModeConvention(self.azimuthal) + DirichletBC(u=0) @ "left"
                 + DirichletBC(u=0) @ "right"
                 + MeshFileOutput(operator=self.operator, eigenvector=0,
                                  eigenmode="real")) @ "domain"


def _mode_convention_output(tmp_path, azimuthal, operator, **solve_kwargs):
    with _ModeConventionProblem(azimuthal, operator) as p:
        p.set_output_directory(str(tmp_path))
        p.quiet()
        p.setup_for_stability_analysis(azimuthal_stability=azimuthal,
                                       additional_cartesian_mode=not azimuthal,
                                       analytic_hessian=False)
        p.solve()
        p.solve_eigenproblem(1, **solve_kwargs)
        p.output()
        outdir = os.path.join(p.get_output_directory(), "domain")
    return _read_one_vtu(outdir)


@pytest.mark.parametrize("azimuthal", [True, False])
def test_stored_eigen_pair_is_the_amplitude_of_a_positive_exponential(tmp_path, azimuthal):
    """Re_w = -m*Im_u and Im_w = +m*Re_u, i.e. the stored pair is (Re,Im) of u_hat.

    This is what fixes the sign of every reconstruction below, so it is checked first and against
    the coordinate system rather than against any of them."""
    reason = _complex_petsc_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    wavenumber = 2.0
    kwargs = {"azimuthal_m": int(wavenumber)} if azimuthal else {"normal_mode_k": wavenumber}
    data, _ = _mode_convention_output(tmp_path, azimuthal, MeshDataCombineWithEigenfunction(0), **kwargs)
    Re_u, Im_u = data["EigenRe_u"], data["EigenIm_u"]
    Re_w, Im_w = data["EigenRe_w"], data["EigenIm_w"]
    scale = max(numpy.max(numpy.abs(Re_u)), numpy.max(numpy.abs(Im_u)))
    assert scale > 1e-6 and numpy.max(numpy.abs(Im_u)) > 1e-6, "eigenmode is trivial or real"
    assert numpy.allclose(Re_w, -wavenumber * Im_u, atol=1e-6 * wavenumber * scale)
    assert numpy.allclose(Im_w, wavenumber * Re_u, atol=1e-6 * wavenumber * scale)


@pytest.mark.parametrize("azimuthal", [True, False])
def test_extruded_scalar_eigenmode_uses_minus_sin_times_the_imaginary_part(tmp_path, azimuthal):
    """The extruded field is Re[u_hat*exp(I*m*phi)], with phi right-handed about the axis."""
    reason = _complex_petsc_skip_reason()
    if reason is not None:
        pytest.skip(reason)
    wavenumber = 2.0
    kwargs = {"azimuthal_m": int(wavenumber)} if azimuthal else {"normal_mode_k": wavenumber}
    combine = MeshDataCombineWithEigenfunction(0)
    flat, _ = _mode_convention_output(tmp_path / "flat", azimuthal, combine, **kwargs)
    extrusion = (MeshDataRotationalExtrusion(n_segments=8) if azimuthal
                 else MeshDataCartesianExtrusion(n_segments=6, use_k_for_length=True))
    ext, m = _mode_convention_output(tmp_path / "ext", azimuthal, combine + extrusion, **kwargs)
    X = numpy.asarray(m.points)
    if azimuthal:
        # the extrusion writes (r cos phi, r sin phi, axial), so phi is right-handed about +z
        in_plane, angle = numpy.hypot(X[:, 0], X[:, 1]), numpy.arctan2(X[:, 1], X[:, 0])
        axial = X[:, 2]
    else:
        in_plane, axial, angle = X[:, 0], X[:, 1], wavenumber * X[:, 2]
    P = _flat_points(tmp_path / "flat")
    src = numpy.array([int(numpy.argmin((P[:, 0] - in_plane[i]) ** 2 + (P[:, 1] - axial[i]) ** 2))
                       for i in range(len(X))])
    Re, Im = flat["EigenRe_u"][src], flat["EigenIm_u"][src]
    phase = wavenumber * angle if azimuthal else angle
    assert numpy.max(numpy.abs(Im)) > 1e-6
    assert numpy.allclose(ext["Eigen_u"], numpy.cos(phase) * Re - numpy.sin(phase) * Im, atol=1e-9)
    assert not numpy.allclose(ext["Eigen_u"], numpy.cos(phase) * Re + numpy.sin(phase) * Im, atol=1e-9)


def _flat_points(tmp_path):
    outdir = os.path.join(str(tmp_path), "domain")
    written = sorted(f for f in os.listdir(outdir) if f.endswith(".vtu"))
    return numpy.asarray(meshio.read(os.path.join(outdir, written[0])).points)
