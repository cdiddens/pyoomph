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
 
# Load this file as a filter in ParaView to use it.
#
# Input: written by MeshFileOutput(operator=MeshDataCombineWithEigenfunction(<index>)), i.e. the
# UNextruded solution carrying EigenRe_<name> / EigenIm_<name> alongside each base field. Both
# stability analyses are supported, selected by "Extrusion Mode":
#
#   Azimuthal - an axisymmetric r-z mesh (x=r, y=z) swept about the y axis. Vector and tensor
#               components turn with the frame.
#   Cartesian - a planar x-y mesh translated along z by one or more wavelengths 2*pi/k. The frame
#               does not turn, so only the mode factor applies.
#
# SIGN CONVENTION. pyoomph stores (Re,Im) of the complex amplitude of exp(I*m*phi), so the physical
# perturbation is Re[u_hat*exp(I*m*phi)] = cos(m*phi)*Re - SIN(m*phi)*Im, with phi the RIGHT-handed
# azimuth about the extrusion axis. That is verifiable without reference to any reconstruction:
# d/dphi of a perturbation is a factor I*m, so projecting w = r*grad(u)_phi must give
# Re_w = -m*Im_u and Im_w = +m*Re_u, which it does. Note that VTK's rotational extrusion about +y
# turns the OTHER way, so arctan2(z,x) is the left-handed azimuth and the right-handed one is
# arctan2(-z,x); the version of this filter that used arctan2(z,x) together with a +sin was correct
# only because the two sign errors cancelled.

import vtk
import numpy
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk #type:ignore

from vtkmodules.vtkCommonDataModel import vtkDataSet, vtkUnstructuredGrid
from vtkmodules.util.vtkConstants import (VTK_LINE, VTK_TRIANGLE, VTK_QUAD, VTK_WEDGE,
                                          VTK_HEXAHEDRON)
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase

# new module for ParaView-specific decorators.
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain #type:ignore


AZIMUTHAL, CARTESIAN = 0, 1

#: input cell type -> (swept cell type, nodes per input cell, whether the swept layer is reversed).
#:
#: A line sweeps into a quad, a triangle into a wedge, a quad into a hexahedron - so an interface
#: mesh (a curve in the r-z plane) sweeps into the surface of revolution, and a bulk mesh into a
#: solid. The flag is the node ordering: a wedge and a hexahedron list one face and then the other,
#: but a quad's four nodes run around its perimeter, so the second layer has to be reversed or the
#: cell comes out bow-tied.
#:
#: Only linear cells; a swept quadratic cell would need mid-layer nodes that the sweep does not
#: create, so those are split into linear ones first.
_SWEEP = {VTK_LINE: (VTK_QUAD, 2, True),
          VTK_TRIANGLE: (VTK_WEDGE, 3, False),
          VTK_QUAD: (VTK_HEXAHEDRON, 4, False)}

#: (row, column) of the six components of a VTK symmetric tensor, in VTK's order.
_SYMMETRIC_SLOTS = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))


def _as_matrix(values):
    """(N,6) or (N,9) tensor components -> (N,3,3)."""
    if values.shape[1] == 9:
        return values.reshape(-1, 3, 3)
    out = numpy.zeros((values.shape[0], 3, 3))
    for slot, (i, j) in enumerate(_SYMMETRIC_SLOTS):
        out[:, i, j] = values[:, slot]
        out[:, j, i] = values[:, slot]
    return out


def _from_matrix(matrices, ncomp):
    """(N,3,3) -> (N,6) or (N,9), matching the component count it came in as."""
    if ncomp == 9:
        return matrices.reshape(-1, 9)
    return numpy.stack([matrices[:, i, j] for i, j in _SYMMETRIC_SLOTS], axis=1)


@smproxy.filter(label="Pyoomph Eigen Extrusion")
@smproperty.input(name="Input")
class PyoomphEigenExtrusion(VTKPythonAlgorithmBase):
    def __init__(self):
        # Always an unstructured grid, even for the surface sweep: the output type has to be fixed
        # at construction, and "Bulk" switches between a polydata shell and a solid at execution
        # time. A shell is representable as an unstructured grid, the reverse is not.
        VTKPythonAlgorithmBase.__init__(self, nInputPorts=1, inputType="vtkDataSet",
                                        nOutputPorts=1, outputType="vtkUnstructuredGrid")
        self.mode = AZIMUTHAL
        self.m = 1
        self.k = 1.0
        self.nextru = 128
        self.angle_deg = 360
        self.numperiods = 1.0
        self.capping = 1
        self.bulk = 0
        self.eigenfactor = 1.0
        self.add_to_base = 0
        self.phi_shift = 0.0
        self.lambda_re = 0.0
        self.lambda_im = 0.0
        self.time = 0.0

    # ------------------------------------------------------------------ properties

    @smproperty.intvector(name="ExtrusionMode", default_values=0)
    @smdomain.xml("""<EnumerationDomain name="enum">
                       <Entry text="Azimuthal (m)" value="0"/>
                       <Entry text="Cartesian (k)" value="1"/>
                     </EnumerationDomain>""")
    def SetExtrusionMode(self, mode):
        self.mode = int(mode)
        self.Modified()

    @smproperty.intvector(name="Azimuthal Mode", default_values=1)
    @smdomain.intrange(min=0, max=20)
    def SetM(self, m):
        self.m = m
        self.Modified()

    @smproperty.doublevector(name="Wavenumber k", default_values=1.0)
    @smdomain.doublerange(min=1e-8, max=1000)
    def SetK(self, k):
        self.k = k
        self.Modified()

    @smproperty.intvector(name="Resolution", default_values=128)
    @smdomain.intrange(min=2, max=500)
    def SetNExtru(self, nextru):
        self.nextru = nextru
        self.Modified()

    @smproperty.doublevector(name="Angle", default_values=360)
    @smdomain.doublerange(min=1, max=360)
    def SetAngleDeg(self, angle_deg):
        self.angle_deg = angle_deg
        self.Modified()

    @smproperty.doublevector(name="Periods", default_values=1.0)
    @smdomain.doublerange(min=0.01, max=100)
    def SetNumPeriods(self, numperiods):
        self.numperiods = numperiods
        self.Modified()

    @smproperty.doublevector(name="Angle Shift", default_values=0)
    @smdomain.doublerange(min=0, max=360)
    def SetAngleShift(self, angle_shift):
        self.phi_shift = angle_shift / 180 * numpy.pi
        self.Modified()

    @smproperty.doublevector(name="Eigen_perturbation", default_values=1.0)
    def SetEigenFactor(self, eigenfactor):
        """Amplitude of the perturbation. Every Eigen_* array is this times the mode, geometry
        included, so it also sets how far the shape moves when AddToBaseState is on."""
        self.eigenfactor = eigenfactor
        self.Modified()

    @smproperty.intvector(name="AddToBaseState", default_values=0)
    @smdomain.xml("""<BooleanDomain name="bool"/>""")
    def SetAddToBaseState(self, value):
        """Fold the perturbation into the base state instead of keeping the two apart.

        Off: each field keeps its base value and the perturbation sits alongside it in Eigen_<field>.
        On: every base field becomes base + perturbation IN PLACE, and the points are displaced by
        Eigen_coordinate, so the deformed shape appears without a downstream Warp By Vector. The
        Eigen_* arrays stay, so the perturbation can still be coloured by on its own."""
        self.add_to_base = int(value)
        self.Modified()

    @smproperty.doublevector(name="Eigenvalue Re", default_values=0.0)
    def SetEigenvalueRe(self, value):
        self.lambda_re = value
        self.Modified()

    @smproperty.doublevector(name="Eigenvalue Im", default_values=0.0)
    def SetEigenvalueIm(self, value):
        self.lambda_im = value
        self.Modified()

    @smproperty.doublevector(name="Time", default_values=0.0)
    def SetTime(self, value):
        self.time = value
        self.Modified()

    @smproperty.intvector(name="Bulk", default_values=0)
    @smdomain.xml("""<EnumerationDomain name="enum">
                       <Entry text="Surface (boundary only)" value="0"/>
                       <Entry text="Bulk (solid)" value="1"/>
                     </EnumerationDomain>""")
    def SetBulk(self, bulk):
        """What gets swept: the boundary of the input, or every cell of it."""
        self.bulk = int(bulk)
        self.Modified()

    @smproperty.intvector(name="Capping", default_values=1)
    @smdomain.intrange(min=0, max=1)
    def SetCapping(self, capping):
        self.capping = capping
        self.Modified()

    # ------------------------------------------------------------------ the sweep

    def _layer_parameters(self):
        """The sweep parameter of each layer, and whether the sweep closes onto itself.

        Azimuthal: an angle in radians. Cartesian: an axial offset covering ``Periods``
        wavelengths, matching MeshDataCartesianExtrusion.
        """
        if self.mode == AZIMUTHAL:
            angle = numpy.deg2rad(self.angle_deg)
            closed = self.angle_deg >= 360 - 1e-8
            n = self.nextru
            return numpy.linspace(0, angle, n, endpoint=not closed), closed
        length = 2 * numpy.pi / self.k * self.numperiods
        return numpy.linspace(0, length, self.nextru + 1, endpoint=True), False

    def _place(self, points, parameter):
        """The input points moved to one layer of the sweep."""
        if self.mode == AZIMUTHAL:
            # About +y, the way vtkRotationalExtrusionFilter does it, so the surface and bulk paths
            # produce the same geometry: a point on +x moves towards -z.
            r, axial = points[:, 0], points[:, 1]
            return numpy.stack([r * numpy.cos(parameter), axial, -r * numpy.sin(parameter)], axis=1)
        moved = points.copy()
        moved[:, 2] = moved[:, 2] + parameter
        return moved

    def _linear_cells(self, inp):
        """(dataset, [(cell type, point ids), ...]) with every cell linear.

        A quadratic cell cannot be swept as it stands - the sweep creates no mid-layer nodes - so it
        is split into linear ones first. That keeps every node and every array, which is what writing
        the output with tesselate_tri=True would have done in the first place.
        """
        if any(inp.GetCellType(c) not in _SWEEP for c in range(inp.GetNumberOfCells())):
            triangulate = vtk.vtkDataSetTriangleFilter() #type:ignore
            triangulate.SetInputDataObject(0, inp)
            triangulate.Update()
            inp = triangulate.GetOutput()
        cells = []
        for c in range(inp.GetNumberOfCells()):
            ctype = inp.GetCellType(c)
            if ctype not in _SWEEP:
                raise RuntimeError("Cannot sweep VTK cell type " + str(ctype)
                                   + "; expected lines, triangles or quads")
            cell = inp.GetCell(c)
            cells.append((ctype, [cell.GetPointId(i) for i in range(_SWEEP[ctype][1])]))
        return inp, cells

    def _sweep(self, inp, cells, caps=()):
        """Sweeps ``cells`` one dimension up: lines into quads, triangles into wedges, quads into
        hexahedra. ``caps`` are cells added unswept at the first and last layer, to close the ends.

        VTK's own extrusion filters are not used for either Bulk or surface output. There is no
        rotational SOLID sweep at all, and vtkLinearExtrusionFilter has no resolution whatsoever - it
        produces a single slab, so the Cartesian surface came out with its two ends exactly one
        wavelength apart. Identical phase, no visible wave, and a shape that deformed uniformly
        instead of sinusoidally once it was added to the base state. Doing both sweeps here makes
        Resolution mean the same thing everywhere.
        """
        points = vtk_to_numpy(inp.GetPoints().GetData()).astype(float)
        npts = len(points)
        params, closed = self._layer_parameters()
        nlayers = len(params)
        out_points = numpy.concatenate([self._place(points, p) for p in params], axis=0)

        conn, types = [], []
        for layer in range(nlayers if closed else nlayers - 1):
            lo = layer * npts
            hi = ((layer + 1) % nlayers) * npts
            for ctype, ids in cells:
                out_type, _n, reverse = _SWEEP[ctype]
                far = [hi + i for i in (reversed(ids) if reverse else ids)]
                conn.append([lo + i for i in ids] + far)
                types.append(out_type)
        if caps and not closed:
            for layer in (0, nlayers - 1):
                for ctype, ids in caps:
                    conn.append([layer * npts + i for i in ids])
                    types.append(ctype)

        grid = vtkUnstructuredGrid()
        pts = vtk.vtkPoints() #type:ignore
        pts.SetData(numpy_to_vtk(numpy.ascontiguousarray(out_points), deep=1))
        grid.SetPoints(pts)
        grid.Allocate(len(conn))
        for out_type, ids in zip(types, conn):
            idlist = vtk.vtkIdList() #type:ignore
            for i in ids:
                idlist.InsertNextId(int(i))
            grid.InsertNextCell(out_type, idlist)

        # every layer carries a copy of the input point data; the transforms later rewrite it
        ipd, opd = inp.GetPointData(), grid.GetPointData()
        for i in range(ipd.GetNumberOfArrays()):
            src = ipd.GetArray(i)
            if src is None:
                continue
            tiled = numpy.tile(vtk_to_numpy(src), (nlayers,) + (1,) * (src.GetNumberOfComponents() > 1))
            arr = numpy_to_vtk(numpy.ascontiguousarray(tiled), deep=1)
            arr.SetName(src.GetName())
            opd.AddArray(arr)
        return grid

    def _bulk_extrude(self, inp):
        """The solid: sweep every cell of the input."""
        linear, cells = self._linear_cells(inp)
        return self._sweep(linear, cells)

    def _surface_extrude(self, inp):
        """The outer skin: sweep the boundary of the input instead of its cells.

        A two-dimensional input is reduced to the edges used by exactly one cell; a one-dimensional
        one - an interface mesh - already IS that boundary and is swept as it stands. The boundary is
        found here rather than with vtkFeatureEdges because that renumbers the points, and the caps
        have to index the same block the sweep does.
        """
        surface = vtk.vtkDataSetSurfaceFilter() #type:ignore
        surface.SetInputDataObject(0, inp)
        surface.Update()
        linear, cells = self._linear_cells(surface.GetOutput())
        planar = [(t, ids) for t, ids in cells if t != VTK_LINE]
        if not planar:
            return self._sweep(linear, cells)
        seen = {}
        for _ctype, ids in planar:
            for a, b in zip(ids, ids[1:] + ids[:1]):
                seen[(min(a, b), max(a, b))] = seen.get((min(a, b), max(a, b)), 0) + 1
        boundary = [(VTK_LINE, [a, b]) for (a, b), n in seen.items() if n == 1]
        return self._sweep(linear, boundary, caps=planar if self.capping else ())

    # ------------------------------------------------------------------ field transforms

    def _sweep_coordinate(self, points):
        """The parameter each output point sits at: the right-handed azimuth, or the axial offset.

        Read back from the geometry rather than from a layer index, so the surface and bulk paths
        share this. arctan2(-z, x), not arctan2(z, x): the extrusion turns +x towards -z, so the
        latter is the LEFT-handed azimuth and would conjugate the mode.
        """
        if self.mode == AZIMUTHAL:
            return numpy.arctan2(-points[:, 2], points[:, 0])
        return points[:, 2]

    def _mode_factors(self, sweep):
        """(weight on Re, weight on Im) of  u_hat*exp(I*(m*phi + lambda_im*t) + lambda_re*t) + c.c.

        The complex eigenvalue enters as a growth factor exp(lambda_re*t) and a phase shift
        lambda_im*t, so a travelling mode simply advances in phi (or z) with time. The factor two is
        the "+ c.c.": the field is u_hat*exp(...) + conjugate, i.e. twice the real part.

        The amplitude is folded in here rather than at each use, so every Eigen_* array is scaled by
        it -- scalars, vectors, tensors and the mesh displacement alike. It used to be applied only
        where a base array existed to add it to, which left Eigen_coordinate ignoring it entirely.
        """
        wavenumber = self.m if self.mode == AZIMUTHAL else self.k
        phase = wavenumber * (sweep - self.phi_shift) + self.lambda_im * self.time
        growth = 2.0 * self.eigenfactor * numpy.exp(self.lambda_re * self.time)
        return growth * numpy.cos(phase), -growth * numpy.sin(phase)

    def _frame(self, sweep):
        """The basis matrix Q whose columns are the images of the field's own component slots.

        Azimuthal: the slots are (r, axial, phi) and the frame turns with phi. Cartesian: the sweep
        translates, so the slots are already (x, y, z) and Q is the identity.
        """
        n = len(sweep)
        Q = numpy.zeros((n, 3, 3))
        if self.mode != AZIMUTHAL:
            Q[:] = numpy.eye(3)
            return Q
        c, s = numpy.cos(sweep), numpy.sin(sweep)
        Q[:, 0, 0], Q[:, 2, 0] = c, -s          # r_hat
        Q[:, 1, 1] = 1.0                        # axial
        Q[:, 0, 2], Q[:, 2, 2] = -s, -c         # phi_hat
        return Q

    def RequestData(self, request, inInfo, outInfo):
        inp = vtkDataSet.GetData(inInfo[0])
        # An empty input is not an error worth a traceback: a reader whose file is missing, or an
        # upstream filter that selected nothing, hands one down and the sweep below would fail on
        # GetPoints() returning None.
        if inp is None or inp.GetNumberOfPoints() == 0:
            return 1
        extr = self._bulk_extrude(inp) if self.bulk else self._surface_extrude(inp)
        pd = extr.GetPointData()
        names = {pd.GetArrayName(i): i for i in range(pd.GetNumberOfArrays())}
        ncomp = {n: pd.GetArray(i).GetNumberOfComponents() for n, i in names.items()}
        get = lambda n: vtk_to_numpy(pd.GetArray(n)).astype(float)

        points = vtk_to_numpy(extr.GetPoints().GetData()).astype(float)
        sweep = self._sweep_coordinate(points)
        Q = self._frame(sweep)
        wRe, wIm = self._mode_factors(sweep)

        # A vector arrives as a 3-component array holding (in-plane, in-plane, out-of-plane). The
        # out-of-plane slot is filled in the array itself for a vector the mesh writes ("normal",
        # the eigenperturbation of the position), but a FIELD's swirl component is a scalar array of
        # its own - "_phi" in the azimuthal coordinate system, "_normal" in the Cartesian one, since
        # the writer groups only _x/_y/_z. Detected per array rather than assumed from the mode, so
        # pointing the wrong mode at a dataset still transforms its vectors instead of dropping them.
        def companion(name):
            for suffix in ("_phi", "_normal"):
                if name + suffix in names:
                    return suffix
            return None

        def out_of_plane(name, values):
            suffix = companion(name)
            return get(name + suffix) if suffix is not None else values[:, 2]

        def eigen_pair(name):
            return ("EigenRe_" + name in names) and ("EigenIm_" + name in names)

        # Everything is collected here and written back at the end, rather than added to and removed
        # from the point data as we go: the bookkeeping of which source array had already been
        # dropped is what silently lost tensors and, later, vectors under a mismatched mode.
        result = {}      # final array name -> values
        consumed = set() # input arrays that must not survive into the output
        handled = set()  # base names whose eigen pair has been dealt with

        for name in list(names):
            if name.startswith("Eigen") or name in consumed:
                # already absorbed as the out-of-plane component of a vector visited earlier; the
                # companion is withdrawn from result below for the opposite array order
                continue
            has_eigen = eigen_pair(name)

            # ---- tensors: no companion array, the slots are positional inside the one array
            if ncomp[name] in (6, 9):
                consumed.add(name)
                result[name] = _from_matrix(
                    numpy.einsum("nai,nij,nbj->nab", Q, _as_matrix(get(name)), Q), ncomp[name])
                if has_eigen:
                    handled.add(name)
                    mode = (wRe[:, None, None] * _as_matrix(get("EigenRe_" + name))
                            + wIm[:, None, None] * _as_matrix(get("EigenIm_" + name)))
                    result["Eigen_" + name] = _from_matrix(
                        numpy.einsum("nai,nij,nbj->nab", Q, mode, Q), ncomp[name])
                    consumed.update(("EigenRe_" + name, "EigenIm_" + name))
                continue

            # ---- vectors
            if ncomp[name] == 3:
                arr = get(name)
                third = companion(name)
                consumed.add(name)
                if third is not None:
                    consumed.add(name + third)
                result[name] = numpy.einsum("nai,ni->na", Q,
                                            numpy.stack([arr[:, 0], arr[:, 1], out_of_plane(name, arr)], axis=1))
            if not has_eigen:
                continue
            re, im = get("EigenRe_" + name), get("EigenIm_" + name)
            if ncomp[name] == 1:
                values = wRe * re + wIm * im
            else:
                eigen_third = companion("EigenRe_" + name)
                if eigen_third is not None and ("EigenIm_" + name + eigen_third) not in names:
                    continue   # nothing consumed yet, so the two halves survive rather than vanish
                cyl = numpy.stack([wRe * re[:, 0] + wIm * im[:, 0],
                                   wRe * re[:, 1] + wIm * im[:, 1],
                                   wRe * out_of_plane("EigenRe_" + name, re)
                                   + wIm * out_of_plane("EigenIm_" + name, im)], axis=1)
                values = numpy.einsum("nai,ni->na", Q, cyl)
                if eigen_third is not None:
                    consumed.update(("EigenRe_" + name + eigen_third, "EigenIm_" + name + eigen_third))
                    # The companion is part of this vector, not a scalar in its own right. It has to
                    # be withdrawn from both directions: the leftover pass below would otherwise
                    # resurrect it, and if the array order happened to reach it BEFORE the vector it
                    # belongs to, this loop has already emitted it as a scalar of its own.
                    handled.add(name + eigen_third)
                    result.pop("Eigen_" + name + eigen_third, None)
            handled.add(name)
            consumed.update(("EigenRe_" + name, "EigenIm_" + name))
            result["Eigen_" + name] = values

        # ---- eigen pairs with no base array of their own. The mesh perturbation is the important
        # one: the coordinates ARE the points, so there is no base "coordinate" array for the loop
        # above to have started from - and that pair is what deforms the shape, i.e. the whole point
        # of plotting an interface eigenmode. Lagrange multipliers such as _kin_bc land here too.
        leftover = sorted({n[len("EigenRe_"):] for n in names if n.startswith("EigenRe_")}
                          & {n[len("EigenIm_"):] for n in names if n.startswith("EigenIm_")}
                          - handled)
        for name in leftover:
            re, im = get("EigenRe_" + name), get("EigenIm_" + name)
            if re.ndim > 1 and re.shape[1] == 3:
                # a vector, so it turns with the frame like any other
                cyl = (wRe[:, None] * numpy.stack([re[:, 0], re[:, 1],
                                                   out_of_plane("EigenRe_" + name, re)], axis=1)
                       + wIm[:, None] * numpy.stack([im[:, 0], im[:, 1],
                                                     out_of_plane("EigenIm_" + name, im)], axis=1))
                result["Eigen_" + name] = numpy.einsum("nai,ni->na", Q, cyl)
            elif re.ndim > 1:
                continue    # neither a scalar nor a vector: no frame to rotate it in
            else:
                result["Eigen_" + name] = wRe * re + wIm * im
            consumed.update(("EigenRe_" + name, "EigenIm_" + name))

        # ---- optionally fold the perturbation into the base state, geometry included
        if self.add_to_base:
            for name in [n for n in result if not n.startswith("Eigen_")]:
                mode = result.get("Eigen_" + name)
                if mode is not None and mode.shape == result[name].shape:
                    result[name] = result[name] + mode
            for name in list(names):
                # base arrays that needed no transform, so they are not in result yet
                if name.startswith("Eigen") or name in result or name in consumed:
                    continue
                mode = result.get("Eigen_" + name)
                if mode is not None and mode.shape == get(name).shape:
                    result[name] = get(name) + mode
            displacement_field = result.get("Eigen_coordinate")
            if displacement_field is not None:
                moved = vtk.vtkPoints() #type:ignore
                moved.SetData(numpy_to_vtk(numpy.ascontiguousarray(points + displacement_field), deep=1))
                extr.SetPoints(moved)

        for name in set(consumed) | set(result):
            present = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
            if name in present:
                pd.RemoveArray(present.index(name))
        for name, values in result.items():
            arr = numpy_to_vtk(numpy.ascontiguousarray(values), deep=1)
            arr.SetName(name)
            pd.AddArray(arr)

        if not extr.IsA("vtkUnstructuredGrid"):
            convert = vtk.vtkAppendFilter() #type:ignore
            convert.SetInputDataObject(0, extr)
            convert.MergePointsOff()   # keep the point ordering the transforms above were built on
            convert.Update()
            extr = convert.GetOutput()
        self.GetOutputDataObject(0).ShallowCopy(extr)
        return 1
