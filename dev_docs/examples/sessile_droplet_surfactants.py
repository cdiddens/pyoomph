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

"""Axisymmetric sessile droplet with insoluble surfactants and forced internal convection.

The point of the example is the last two lines of ``define_problem``: a
:py:class:`~pyoomph.equations.generic.SpatialErrorEstimator` added **on the free surface**, driving
the mesh from the surfactant distribution and from the curvature of the interface itself.

A body force near the symmetry axis pushes liquid upwards; incompressibility turns that into a
toroidal roll that returns along the free surface, sweeping the insoluble surfactant from the apex
towards the contact line. The surfactant piles up into a front there, and lowers the surface tension
where it does, so the interface needs resolution in a place no bulk field asks for it.

On interface refinement thresholds, and on why this needs a recent pyoomph, see the notes on
``RefinementThresholds`` below and ``dev_docs/spatial_error_estimators.md``.

To be precise about the second point, because it is easy to overclaim: the Z2 flux recovery used to
fit its polynomial in the first ``dim`` *global* coordinates, which parametrises an interface only
when the interface is a graph over those axes. This particular droplet, at a 90 degree contact
angle, still is one - the radius increases monotonically from apex to contact line - so the old code
merely became ill-conditioned near the contact line rather than failing (measured: the two agree to
1e-11 here). What it could not do at all is the general case: an overhanging contact angle, a closed
interface, a vertical wall, or any 3D surface at ``x = const``, all of which made the recovery matrix
exactly singular and killed the run inside ``DenseLU``. Those cases are pinned in
``tests/test_boundary_error_estimator.py``. The recovery is now fitted in a patch-local tangent
frame, which is well conditioned for all of them.
"""

from pyoomph import *
from pyoomph.expressions.units import *
from pyoomph.expressions.phys_consts import gas_constant

from pyoomph.equations.navier_stokes import *
from pyoomph.equations.ALE import *
from pyoomph.meshes.simplemeshes import CircularMesh


class RefinementThresholds(Equations):
    """Set the refine/unrefine thresholds of the mesh these equations are added to.

    Needed here because Z2 errors are normalised by *each mesh's own* recovered-flux norm, so an
    element's error is essentially its share of that mesh's total. A free surface with a hundred
    elements therefore reports errors two orders of magnitude larger than a bulk mesh with tens of
    thousands, and the Problem-wide default (1e-3/1e-4) would put every single interface element
    above the refine threshold - which is a uniform refinement wearing an error estimator's clothes.

    Note that rescaling the estimator *expression* instead does nothing at all: the normalisation
    divides it straight back out (dev_docs/spatial_error_estimators.md section 7.1). The
    ``weight=`` argument of SpatialErrorEstimator does survive the normalisation and is the right
    tool for balancing two criteria - but only against each other *on one mesh*. The free surface
    and the bulk are different meshes with their own thresholds, so it cannot help here.
    """

    def __init__(self, max_error, min_error):
        super().__init__()
        self.max_error, self.min_error = max_error, min_error

    def after_compilation(self, codegen):
        mesh = codegen._mesh
        assert mesh is not None
        mesh.max_permitted_error = self.max_error
        mesh.min_permitted_error = self.min_error


class InsolubleSurfactant(InterfaceEquations):
    """Transport of an insoluble surfactant along a moving interface.

    ``div(u*Gamma)`` on an interface is the surface divergence, so this is the usual
    advection-diffusion equation on a deforming surface, including the dilution by stretching.
    """

    def __init__(self, diffusivity):
        super().__init__()
        self.diffusivity = diffusivity

    def define_fields(self):
        self.define_scalar_field("Gamma", "C2",
                                 testscale=scale_factor("temporal") / scale_factor("Gamma"))

    def define_residuals(self):
        u = var("velocity")  # the interface moves with the fluid
        G, Gtest = var_and_test("Gamma")
        self.add_residual(weak(partial_t(G) + div(u * G), Gtest))
        self.add_residual(weak(self.diffusivity * grad(G), grad(Gtest)))


class SurfactantDropletProblem(Problem):
    def __init__(self):
        super().__init__()
        self.volume = 0.25 * milli * liter
        self.rho = 1000 * kilogram / meter**3
        self.mu = 1 * milli * pascal * second
        self.sigma0 = 72 * milli * newton / meter
        self.slip_length = 1 * micro * meter
        self.temperature = 20 * celsius

        self.Gamma0 = 1 * micro * mol / meter**2      # initial, uniform surfactant coverage
        self.surf_diffusivity = 1e-9 * meter**2 / second

        # Strength and width of the stirring force near the axis. The force is what makes this
        # "forced" convection rather than a Marangoni-driven flow: the surfactant gradient is a
        # consequence of the flow here, not its cause. Much above this the roll wins outright, the
        # interface overturns and the Newton solver gives up.
        self.forcing = 2000 * newton / meter**3
        self.forcing_width = 0.3
        # 90 degrees matches the initial hemisphere, so the run starts from a consistent state. An
        # overhanging angle needs to be reached by continuation (see the droplet_spread tutorials);
        # imposing one here from a hemisphere just diverges.
        self.contact_angle = 90 * degree

    def define_problem(self):
        self.set_coordinate_system("axisymmetric")
        R0 = square_root(3 * self.volume / (4 * pi) * 2, 3)  # radius of the initial hemisphere

        self += CircularMesh(radius=R0, segments=["NE"], outer_interface="interface",
                             straight_interface_name={"center_to_north": "axis",
                                                      "center_to_east": "substrate"})

        self.set_scaling(spatial=R0, temporal=1 * second,
                         velocity=scale_factor("spatial") / scale_factor("temporal"))
        self.set_scaling(pressure=self.sigma0 / R0, Gamma=self.Gamma0)

        # An upward push near the axis. The droplet is incompressible and closed, so what comes up
        # along the axis has to come back down along the free surface: a toroidal roll.
        r = var("coordinate_x")
        bulkforce = self.forcing * exp(-(r / (self.forcing_width * R0))**2) * vector(0, 1)

        eqs = MeshFileOutput()
        eqs += NavierStokesEquations(mass_density=self.rho, dynamic_viscosity=self.mu,
                                     bulkforce=bulkforce)
        eqs += PseudoElasticMesh()
        eqs += RefineToLevel(2)  # the raw CircularMesh is far too coarse to start from

        eqs += DirichletBC(velocity_x=0, mesh_x=0) @ "axis"
        eqs += DirichletBC(velocity_y=0, mesh_y=0) @ "substrate"
        eqs += NavierStokesSlipLength(self.slip_length) @ "substrate"

        # Surface tension falls where the surfactant accumulates.
        sigma = self.sigma0 - gas_constant * self.temperature * var("Gamma")

        ieqs = NavierStokesFreeSurface(surface_tension=sigma)
        ieqs += InsolubleSurfactant(self.surf_diffusivity)
        ieqs += InitialCondition(Gamma=self.Gamma0)
        ieqs += MeshFileOutput()

        # The new part. On the interface, "Gamma" resolves the surfactant front the roll builds up
        # near the contact line, and "normal" resolves the shape itself: the jump in the interface
        # normal is its curvature, so this refines wherever the free surface bends.
        ieqs += SpatialErrorEstimator("normal", Gamma=1)
        ieqs += RefinementThresholds(0.1, 0.02)

        eqs += ieqs @ "interface"
        eqs += NavierStokesContactAngle(self.contact_angle, wall_normal=vector(0, 1),
                                        wall_tangent=vector(-1, 0)) @ "interface/substrate"

        # Bulk refinement, so that the interface-driven refinement is visibly *extra* rather than
        # the only thing happening.
        eqs += SpatialErrorEstimator(velocity=1)

        self += eqs @ "domain"


if __name__ == "__main__":
    with SurfactantDropletProblem() as problem:
        problem.initial_adaption_steps = 2
        problem.max_refinement_level = 6
        problem.run(2 * second, startstep=0.02 * second, outstep=0.1 * second,
                    temporal_error=1, spatial_adapt=1)
