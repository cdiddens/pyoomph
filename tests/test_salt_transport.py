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

"""
Salt transport: the electroneutral model, and what evaporation does to a dissolved salt.

Three things have to hold, and each has a section below.

  1. THE SALT STAYS. When the solvent evaporates, the interface recedes past the liquid and the salt
     does not leave with it. That is not automatic -- the natural boundary condition of the transport
     equations is zero *diffusive* flux, which lets a receding surface sweep the salt out -- so it is
     imposed at the interface, and it is imposed differently for each of the three ALE forms the
     composition equations offer. The conservative (GCL) one conserves the dissolved amount to
     machine precision; the other two to the order of the time stepping.
  2. THE FIELD NAMES ARE SHARED WITH NERNST-PLANCK. c_Na_p means the same thing whether the ions came
     from this model or from PoissonNernstPlanck, so a surface tension law or a boundary condition
     written against it does not know which is running. What must *not* happen is both at once.
  3. SALT RAISES THE SURFACE TENSION, hence Marangoni flow towards the enriched region -- the
     opposite direction to a surfactant, which is worth getting right rather than assuming.

Each Problem gets its own output directory: several Problems in one directory share the JIT cache,
and variants that differ only in constructor flags then silently reuse the first one's compiled code.
"""

import itertools

import numpy
import pytest

from pyoomph import *
from pyoomph.expressions import *
from pyoomph.expressions.units import *
from pyoomph.equations.multi_component import *
from pyoomph.equations.ALE import LaplaceSmoothedMesh
from pyoomph.equations.salt_transport import SaltTransportEquations
from pyoomph.materials import *
from pyoomph.materials.mass_transfer import PrescribedMassTransfer
from pyoomph.meshes.simplemeshes import LineMesh, RectangularQuadMesh
import pyoomph.materials.default_materials  # noqa: F401  (registers the material library)
import pyoomph.materials.ions  # noqa: F401  (registers the ions and salts)

_run_counter = itertools.count()


def _fresh(problem):
    """Own output directory per Problem, see the note at the top."""
    problem.set_output_directory("run%d" % next(_run_counter))
    problem.set_c_compiler("system")
    return problem


def _salted(salt="NaCl", concentration=100 * milli * molar, name="water"):
    """A fresh liquid with one salt dissolved in it."""
    w = get_pure_liquid(name)
    w.add_salt(salt, concentration)
    return w


# =================================================================================================
#  The material side
# =================================================================================================

# Measured diffusivities of the dissolved salt at 25 C, 1e-9 m^2/s (CRC / Robinson & Stokes). The
# library stores neither of these: it has the two ion diffusivities and combines them.
_SALT_DIFFUSIVITIES = {"NaCl": 1.610, "KCl": 1.990, "CaCl2": 1.335, "LiCl": 1.370,
                       "Na2SO4": 1.230, "HCl": 3.340}


@pytest.mark.parametrize("name,D0", sorted(_SALT_DIFFUSIVITIES.items()))
def test_ambipolar_diffusivity_reproduces_the_measured_salt_diffusivity(name, D0):
    # A salt does not move at either ion's speed: whichever ion runs ahead builds a charge separation
    # whose field pulls it back, so the pair travels together at one rate. HCl is the case that shows
    # this is a real prediction and not a fitted average -- its ions differ by a factor of 4.6.
    salt = get_salt(name)
    D = salt.get_ambipolar_diffusivity(298.15 * kelvin)
    assert float(D / (1e-9 * meter ** 2 / second)) == pytest.approx(D0, rel=1e-2)


def test_add_salt_takes_a_salt_directly():
    w = get_pure_liquid("water")
    w.add_salt(get_salt("CaCl2"), 1 * milli * molar)
    w2 = get_pure_liquid("water")
    w2.add_salt("CaCl2", 1 * milli * molar)          # ... or its name
    for m in (w, w2):
        assert sorted(m.get_salts()) == ["CaCl2"]
        assert float(m.get_bulk_concentration("Cl-") / (mol / meter ** 3)) == pytest.approx(2.0)
        assert float(m.get_net_charge_number() / (mol / meter ** 3)) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(TypeError, match="is an ion, and a salt needs both"):
        get_pure_liquid("water").add_salt("Na+", 1 * milli * molar)


def test_two_salts_sharing_an_ion_add_up():
    # add_ion overwrites the ion's concentration, so this is the case that needs the salt table: the
    # ion carries the sum, and the salts stay distinguishable.
    w = get_pure_liquid("water")
    w.add_salt("NaCl", 1 * milli * molar)
    w.add_salt("Na2SO4", 1 * milli * molar)
    assert float(w.get_bulk_concentration("Na+") / (mol / meter ** 3)) == pytest.approx(3.0)
    assert float(w.get_net_charge_number() / (mol / meter ** 3)) == pytest.approx(0.0, abs=1e-12)
    assert sorted(w.get_salts()) == ["Na2SO4", "NaCl"]
    with pytest.raises(ValueError, match="already dissolved"):
        w.add_salt("NaCl", 1 * milli * molar)


def test_the_salt_raises_the_surface_tension_of_the_interface():
    plain = get_pure_liquid("water")
    salty = _salted(concentration=1 * molar)
    gas = get_pure_gas("air")
    T = 20 * celsius
    # "at the IC" resolves the salt field to the concentration it was dissolved at, which is what
    # keeps every scale that evaluates a surface tension working on a salted liquid.
    sigma_plain = plain.evaluate_at_condition((plain | gas).surface_tension, "IC", temperature=T)
    sigma_salty = salty.evaluate_at_condition((salty | gas).surface_tension, "IC", temperature=T)
    # 1 M NaCl: +1.64 mN/m, the tabulated increment. Salts are surface-depleted, so this is a rise.
    assert float((sigma_salty - sigma_plain) / (milli * newton / meter)) == pytest.approx(1.64, rel=1e-6)
    # ... and an unsalted liquid is untouched, bitwise: this touches every liquid-gas interface.
    assert str(sigma_plain) == str(plain.evaluate_at_condition(
        get_pure_liquid("water").default_surface_tension["gas"], "IC", temperature=T))


# =================================================================================================
#  What the equations do with it
# =================================================================================================

def test_composition_flow_picks_the_salt_up_by_itself():
    # The trap this closes: before, a salted material generated exactly the same system as an
    # unsalted one, and nothing said so.
    salted = CompositionFlowEquations(_salted())
    assert isinstance(salted.get_equation_of_type(SaltTransportEquations), SaltTransportEquations)
    plain = CompositionFlowEquations(get_pure_liquid("water"))
    assert plain.get_equation_of_type(SaltTransportEquations) is None
    off = CompositionFlowEquations(_salted(), salts=False)
    assert off.get_equation_of_type(SaltTransportEquations) is None
    with pytest.raises(RuntimeError, match="no salt is dissolved"):
        CompositionFlowEquations(get_pure_liquid("water"), salts=True)


def test_the_salt_fields_carry_the_nernst_planck_names():
    eqs = CompositionFlowEquations(_salted("CaCl2")).get_equation_of_type(SaltTransportEquations)
    assert isinstance(eqs, SaltTransportEquations)
    assert eqs.fieldname_of("CaCl2") == "c_CaCl2"
    # The ions are substitutions under the names PoissonNernstPlanck would solve for, so anything
    # written against them works under either model.
    assert eqs.ion_fieldname_of("Ca2+") == "c_Ca2_p"
    assert eqs.ion_fieldname_of("Cl-") == "c_Cl_m"
    assert eqs.get_ion_names() == ["Ca2+", "Cl-"]
    # ... and the charge density is not approximately zero, it is structurally zero.
    assert is_zero(eqs.get_charge_density())


def test_the_two_electrolyte_models_refuse_to_share_a_domain():
    # A substituted c_Na_p silently shadowed by a solved c_Na_p is the failure mode worth an error:
    # it would run, and it would be wrong.
    from pyoomph.equations.electrostatics import PoissonNernstPlanck

    class P(Problem):
        def define_problem(self):
            water = _salted()
            self.define_named_var(temperature=20 * celsius)
            self.set_scaling(spatial=1 * milli * meter, ion_concentration=100 * milli * molar,
                             potential=25 * milli * volt, temporal=1 * second)
            self.add_mesh(LineMesh(N=4, size=1 * milli * meter))
            eqs = SaltTransportEquations(fluid_props=water, wind=0)
            eqs += PoissonNernstPlanck(fluid_props=water, wind=0, temperature=20 * celsius)
            self.add_equations(eqs @ "domain")

    with pytest.raises(RuntimeError, match="Use one"):
        with _fresh(P()) as p:
            p.initialise()


# =================================================================================================
#  Evaporation: the salt has to stay behind
# =================================================================================================

class _EvaporatingFilm(Problem):
    """A 1d film of salt water evaporating at the right end, which recedes as it does.

    The left end is a wall, so the liquid does not move at all: the interface simply sweeps past it.
    That makes the answer exact -- the salt cannot go anywhere, so c(t)*L(t) must equal c0*L0 -- and
    it is the sharpest possible test of the interface condition, since every discretisation detail
    that is wrong shows up as salt appearing or disappearing.
    """

    def __init__(self, *, GCL=False, byparts=False, N=20, L=1 * milli * meter,
                 c0=100 * milli * molar, salt="NaCl", stabilization=None, salts="auto",
                 j=1e-4 * kilogram / (meter ** 2 * second), Deff=None, mode="dilute"):
        super().__init__()
        self.GCL, self.byparts, self.N, self.L = GCL, byparts, N, L
        self.c0, self.salt, self.j, self.Deff = c0, salt, j, Deff
        self.stabilization, self.salts = stabilization, salts
        #: "dilute" or "component", i.e. whether the salt is a concentration riding along or a
        #: composition field of its own.
        self.mode = mode
        self.T = 20 * celsius

    def define_problem(self):
        if self.mode == "component":
            water = Mixture(get_pure_liquid("water") + self.c0 * get_salt(self.salt),
                            temperature=self.T, salt_treatment="component")
        else:
            water = _salted(self.salt, self.c0)
        self.water = water
        self.interf = water | get_pure_gas("air")
        self.interf.set_mass_transfer_model(PrescribedMassTransfer(water=self.j)).projection_space = "C2"
        rho0 = water.evaluate_at_condition(water.mass_density, "IC", temperature=self.T)
        mu0 = water.evaluate_at_condition(water.dynamic_viscosity, "IC", temperature=self.T)
        U = self.j / rho0
        # spatial and velocity first: set_reference_scaling_to_problem derives the rest from them.
        self.set_scaling(spatial=self.L, velocity=U, temporal=self.L / U, pressure=mu0 * U / self.L)
        water.set_reference_scaling_to_problem(self, temperature=self.T)
        self.set_scaling(spatial=self.L, velocity=U, temporal=self.L / U, pressure=mu0 * U / self.L)
        self.set_scaling(ion_concentration=self.c0)
        self.define_named_var(temperature=self.T, absolute_pressure=1 * atm)

        eqs = CompositionFlowEquations(water, compo_space="C1", GCL=self.GCL, salts=self.salts,
                                       integrate_advection_by_parts=self.byparts or self.GCL,
                                       salt_stabilization=self.stabilization,
                                       salt_treatment=self.mode)
        if self.Deff is not None:   # a smaller diffusivity, i.e. a higher Peclet number
            if self.mode == "component":
                water.set_diffusion_coefficient(self.salt, self.Deff)
            else:
                salt_eqs = eqs.get_equation_of_type(SaltTransportEquations)
                assert isinstance(salt_eqs, SaltTransportEquations)
                salt_eqs.diffusivities[self.salt] = self.Deff
        eqs += LaplaceSmoothedMesh()
        eqs += (DirichletBC(mesh_x=0) + DirichletBC(velocity_x=0)) @ "left"
        eqs += MultiComponentNavierStokesInterface(self.interf) @ "right"
        eqs += IntegralObservables(L=1, rho=water.mass_density, w_water=var("massfrac_water"))
        eqs += IntegralObservables(sigma=self.interf.surface_tension, A=1) @ "right"
        if self.salts is not False:
            eqs += IntegralObservables(N_salt=var("c_" + self.salt))
            eqs += IntegralObservables(c_surf=var("c_" + self.salt)) @ "right"
        self += LineMesh(size=self.L, N=self.N)
        self += eqs @ "domain"


def _evaporate(prob, *, nsteps=20, shrink_to=0.5, end_time=None):
    """Run until the film has lost the given fraction of its height. Fixed steps: the point of these
    tests is how the error behaves in dt, which an adaptive stepper would hide.

    ``end_time`` overrides the duration. Needed when two problems are to be compared: the default is
    derived from each problem's own velocity scale, which is j/rho, and a denser liquid would then be
    run for longer rather than compared at the same instant.
    """
    with _fresh(prob) as p:
        p.initialise()
        o0 = p.get_mesh("domain").evaluate_all_observables()
        Tend = end_time if end_time is not None else (1 - shrink_to) * p.get_scaling("temporal")
        p.run(Tend, outstep=False, startstep=Tend / nsteps, maxstep=Tend / nsteps, temporal_error=None)
        o = p.get_mesh("domain").evaluate_all_observables()
        os_ = p.get_mesh("domain/right").evaluate_all_observables()
        res = {"L0": float(o0["L"] / meter), "L": float(o["L"] / meter),
               "T_end": float(Tend / second),
               "rho": float(o["rho"] / o["L"] / (kilogram / meter ** 3)),
               "w_water": float(o["w_water"] / o["L"]),
               "sigma": float(os_["sigma"] / os_["A"] / (milli * newton / meter))}
        if "N_salt" in o:  # absent when the salt transport was switched off
            res["N0"] = float(o0["N_salt"] / (mol / meter ** 2))
            res["N"] = float(o["N_salt"] / (mol / meter ** 2))
            res["c_surf"] = float(os_["c_surf"] / os_["A"] / (mol / meter ** 3))
        return res


def test_the_conservative_form_conserves_the_salt_exactly():
    # GCL: the transient term is the derivative of the whole integral and the advection uses the
    # velocity relative to the mesh, so "nothing leaves through a moving boundary" is the natural
    # boundary condition rather than something imposed to the accuracy of the time stepping.
    r = _evaporate(_EvaporatingFilm(GCL=True), nsteps=20)
    assert abs(r["N"] / r["N0"] - 1) < 1e-10
    # And the salt really did concentrate: the film halved, so with nothing lost the concentration
    # has to double. Measured against the *measured* length, because the GCL continuity equation
    # reaches the same interface position by a different route and is a percent off at 20 steps.
    assert r["c_surf"] / (r["N"] / r["L"]) == pytest.approx(1.0, rel=2e-2)   # nearly well mixed, Pe=0.03
    assert r["c_surf"] == pytest.approx(100.0 * r["L0"] / r["L"], rel=3e-2)


def test_the_surface_tension_follows_the_enriched_surface():
    # The whole chain in one number: the salt is held back by the interface condition, it piles up
    # at the receding surface, and the surface tension there follows the concentration it finds.
    r = _evaporate(_EvaporatingFilm(GCL=True), nsteps=20)
    water, gas = get_pure_liquid("water"), get_pure_gas("air")
    sigma0 = float(water.evaluate_at_condition((water | gas).surface_tension, "IC",
                                               temperature=20 * celsius) / (milli * newton / meter))
    assert r["sigma"] - sigma0 == pytest.approx(1.64 * r["c_surf"] / 1000.0, rel=1e-6)
    assert r["sigma"] > sigma0                       # a salt raises it
    assert r["c_surf"] > 190.0                       # ... and by then the film has nearly doubled it


@pytest.mark.parametrize("byparts", [False, True])
def test_the_non_conservative_forms_converge_at_second_order(byparts):
    # Without the conservative form the salt balance is only as good as the time stepping, which is
    # the honest statement about it: BDF2, so the error falls by four when dt is halved. Both ALE
    # forms need a different interface term to get here, and neither term is nothing.
    coarse = _evaporate(_EvaporatingFilm(byparts=byparts), nsteps=10)
    fine = _evaporate(_EvaporatingFilm(byparts=byparts), nsteps=20)
    e_coarse, e_fine = abs(coarse["N"] / coarse["N0"] - 1), abs(fine["N"] / fine["N0"] - 1)
    assert e_coarse < 2e-2 and e_fine < 5e-3
    assert e_coarse / e_fine == pytest.approx(4.0, rel=0.35)


def test_evaporation_builds_an_enrichment_layer_at_the_surface():
    # At a Peclet number of order one the salt cannot diffuse away from the receding surface as fast
    # as the surface catches up with it, so a boundary layer of thickness D/v builds up. This is the
    # gradient that drives the Marangoni flow further down, so it has to be right rather than merely
    # present. Mesh converged: N=60 and N=150 agree to 0.06%.
    r = _evaporate(_EvaporatingFilm(GCL=True, N=60, Deff=2e-11 * meter ** 2 / second),
                   nsteps=40, shrink_to=0.7)
    v = 1e-4 / 998.2                      # the recession speed, j/rho
    Pe = v * r["L"] / 2e-11
    ratio = r["c_surf"] / (r["N"] / r["L"])
    assert Pe == pytest.approx(3.5, rel=0.1)
    assert ratio == pytest.approx(2.31, rel=3e-2)
    # It is bounded by the quasi-steady profile c = c_s*exp(-v*xi/D), whose surface excess is
    # Pe/(1-exp(-Pe)). The layer is still building at the end of this run, so the measured excess
    # must sit below that and above the well-mixed value of 1 -- which is what pins the sign of the
    # whole effect.
    assert 1.0 < ratio < Pe / (1 - numpy.exp(-Pe))
    assert abs(r["N"] / r["N0"] - 1) < 1e-10   # still exactly conserved at this Peclet number


# =================================================================================================
#  Marangoni, and the agreement with Nernst-Planck
# =================================================================================================

class _SaltMarangoniPool(Problem):
    """A closed 2d pool whose free surface carries a salt gradient.

    No evaporation: the gradient is imposed as the initial state, which isolates what the surface
    tension law does with it from how the gradient got there.
    """

    def __init__(self, *, increment=None, W=1 * milli * meter, H=0.25 * milli * meter, N=(24, 6),
                 c0=100 * milli * molar, dc=50 * milli * molar):
        super().__init__()
        self.W, self.H, self.N, self.c0, self.dc, self.increment = W, H, N, c0, dc, increment
        self.T = 20 * celsius

    def define_problem(self):
        water = get_pure_liquid("water")
        salt = get_salt("NaCl")
        if self.increment is not None:
            # get_salt hands out a fresh instance, so this does not leak into the next problem.
            salt.surface_tension_increment = self.increment
        water.add_salt(salt, self.c0)
        interf = water | get_pure_gas("air")
        interf.set_mass_transfer_model(None)
        mu0 = water.evaluate_at_condition(water.dynamic_viscosity, "IC", temperature=self.T)
        U = 1.64 * milli * newton / meter / molar * self.dc / mu0     # the Marangoni velocity scale
        self.set_scaling(spatial=self.W, velocity=U, temporal=self.W / U, pressure=mu0 * U / self.W)
        water.set_reference_scaling_to_problem(self, temperature=self.T)
        self.set_scaling(spatial=self.W, velocity=U, temporal=self.W / U, pressure=mu0 * U / self.W)
        self.set_scaling(ion_concentration=self.c0)
        self.define_named_var(temperature=self.T, absolute_pressure=1 * atm)
        eqs = CompositionFlowEquations(water, compo_space="C1")
        eqs += InitialCondition(c_NaCl=self.c0 + self.dc * var("coordinate_x") / self.W)
        eqs += DirichletBC(velocity_x=0, velocity_y=0) @ "bottom"
        eqs += DirichletBC(velocity_x=0) @ "left"
        eqs += DirichletBC(velocity_x=0) @ "right"
        eqs += MultiComponentNavierStokesInterface(interf, static=True) @ "top"
        eqs += IntegralObservables(us=var("velocity_x"), A=1) @ "top"
        self += RectangularQuadMesh(N=self.N, size=[self.W, self.H])
        self += eqs @ "domain"


def _surface_velocity(increment):
    with _fresh(_SaltMarangoniPool(increment=increment)) as p:
        p.initialise()
        Tend = 0.02 * p.get_scaling("temporal")
        p.run(Tend, outstep=False, startstep=Tend / 10, maxstep=Tend / 10, temporal_error=None)
        o = p.get_mesh("domain/top").evaluate_all_observables()
        return float(o["us"] / o["A"] / (meter / second))


def test_salt_marangoni_pulls_towards_the_salt():
    # The direction is the point. A salt *raises* the surface tension, so the surface is pulled
    # towards the salt-rich end -- the opposite of a surfactant, which lowers it and is pulled away.
    # An evaporating drop therefore drives its surface towards wherever it is drying fastest.
    u_salt = _surface_velocity(None)                                   # the library value, +1.64
    u_none = _surface_velocity(0 * milli * newton / meter / molar)
    u_acid = _surface_velocity(-1.64 * milli * newton / meter / molar)  # what a strong acid does
    assert u_salt > 0                                     # towards +x, where the salt is
    assert abs(u_none) < 1e-8 * abs(u_salt)               # nothing at all without an increment
    assert u_acid == pytest.approx(-u_salt, rel=1e-3)     # and exactly reversed for the acid


class _RelaxingGradient(Problem):
    """An electroneutral salt gradient relaxing by diffusion, solved either way."""

    def __init__(self, mode, *, N=60, L=1 * micro * meter, c0=100 * milli * molar, amp=0.2):
        super().__init__()
        self.mode, self.N, self.L, self.c0, self.amp = mode, N, L, c0, amp
        self.T = 298.15 * kelvin

    def define_problem(self):
        from pyoomph.equations.electrostatics import (PoissonNernstPlanck, ElectrodeBC,
                                                      set_electrostatic_scaling, thermal_voltage)
        water = get_pure_liquid("water")
        water.add_salt("NaCl", self.c0)
        self.D = water.get_salts()["NaCl"].get_diffusivity(water, self.T)
        self.set_scaling(spatial=self.L, temporal=self.L ** 2 / self.D)
        set_electrostatic_scaling(self, potential=thermal_voltage(self.T), temperature=self.T,
                                  ion_concentration=self.c0)
        self.define_named_var(temperature=self.T)
        self.add_mesh(LineMesh(N=self.N, size=self.L))
        profile = self.c0 * (1 + self.amp * cos(pi * var("coordinate_x") / self.L))
        if self.mode == "salt":
            eqs = SaltTransportEquations(fluid_props=water, wind=0, temperature=self.T)
            eqs += InitialCondition(c_NaCl=profile)
        else:
            eqs = PoissonNernstPlanck(fluid_props=water, wind=0, temperature=self.T)
            eqs += InitialCondition(c_Na_p=profile, c_Cl_m=profile)
            eqs += ElectrodeBC(0) @ "left"
        # The observable is written against the ion field, which is a substitution in one model and
        # a solved dof in the other. That it can be written once is the point of the shared names.
        eqs += IntegralObservables(mode=var("c_Na_p") * cos(pi * var("coordinate_x") / self.L), V=1)
        self.add_equations(eqs @ "domain")


def _relax(mode, tau=0.05, nsteps=20):
    with _fresh(_RelaxingGradient(mode)) as p:
        p.initialise()
        o0 = p.get_mesh("domain").evaluate_all_observables()
        Tend = tau * p.get_scaling("temporal")
        p.run(Tend, outstep=False, startstep=Tend / nsteps, maxstep=Tend / nsteps, temporal_error=None)
        o = p.get_mesh("domain").evaluate_all_observables()
        return float(o["mode"] / o0["mode"])


def test_the_electroneutral_model_is_what_nernst_planck_reduces_to():
    # The claim the shared field names make: with a thin double layer and no applied field, solving
    # one salt with the ambipolar diffusivity and solving two ions plus a potential are the same
    # physics. The first Fourier mode must decay as exp(-pi^2 D_ambipolar t/L^2) in both, and the
    # two must agree far more closely than either agrees with the analytic value (which is exact
    # only in the continuum limit). lambda_D is 0.96 nm here against a 1 um box, so the double layer
    # has no room to matter.
    tau = 0.05
    a_salt, a_pnp = _relax("salt", tau), _relax("pnp", tau)
    analytic = numpy.exp(-numpy.pi ** 2 * tau)
    assert a_salt == pytest.approx(analytic, rel=2e-3)
    assert a_pnp == pytest.approx(analytic, rel=2e-3)
    assert a_salt == pytest.approx(a_pnp, rel=1e-4)


def test_switching_the_salt_off_leaves_a_uniform_surface_tension():
    # With salts=False and no Nernst-Planck there is no concentration field to read, so the shift
    # falls back to the uniform bulk value: the absolute surface tension is still right, there is
    # just no gradient. Without that fallback the interface would reference a field nobody defines.
    r = _evaporate(_EvaporatingFilm(GCL=True, salts=False), nsteps=10, shrink_to=0.8)
    water, gas = get_pure_liquid("water"), get_pure_gas("air")
    sigma0 = float(water.evaluate_at_condition((water | gas).surface_tension, "IC",
                                               temperature=20 * celsius) / (milli * newton / meter))
    assert r["sigma"] - sigma0 == pytest.approx(1.64 * 0.1, rel=1e-6)   # the 100 mM it was made with
    assert "N" not in r             # there is no salt field at all


def test_a_salted_mixture_transports_its_solvents_and_its_salt():
    # The case the whole thing is for: Mixture(water + 20% glycerol + 1 mM NaCl) handed to
    # CompositionFlowEquations, evaporating. The mass fractions and the salt are solved side by
    # side, they use different bookkeeping (fractions that sum to unity versus a concentration), and
    # the salt's diffusivity comes from the *mixture's* viscosity rather than water's.
    mix = Mixture(get_pure_liquid("water") + 20 * percent * get_pure_liquid("glycerol")
                  + 100 * milli * molar * get_salt("NaCl"))
    eqs = CompositionFlowEquations(mix)
    salt_eqs = eqs.get_equation_of_type(SaltTransportEquations)
    assert isinstance(salt_eqs, SaltTransportEquations)
    compo = eqs.get_equation_of_type(CompositionAdvectionDiffusionEquations)
    assert isinstance(compo, CompositionAdvectionDiffusionEquations)
    assert compo.fieldnames == ["massfrac_glycerol"]        # water is the passive one
    assert salt_eqs.fieldname_of("NaCl") == "c_NaCl"
    T = 20 * celsius
    # The glycerol thickens the solvent, so the salt has to move slower than it would in water.
    D_water = float(get_salt("NaCl").get_ambipolar_diffusivity(T) / (1e-9 * meter ** 2 / second))
    D_in_mix = float(mix.get_salts()["NaCl"].get_diffusivity(mix, T) / (1e-9 * meter ** 2 / second))
    assert D_in_mix < 0.7 * D_water


def test_the_surface_tension_law_works_under_nernst_planck_too():
    # The other half of the shared-names claim. The salt raises the surface tension through the *ion*
    # concentrations, which are a substitution in the electroneutral model and solved dofs here, so
    # the same interface object gives the same sigma under either -- and salt Marangoni is available
    # to a full electrohydrodynamic problem without writing a second surface tension law.
    from pyoomph.equations.electrostatics import (PoissonNernstPlanck, set_electrostatic_scaling,
                                                  thermal_voltage)
    T = 298.15 * kelvin
    c0 = 100 * milli * molar

    class P(Problem):
        def define_problem(self):
            water = _salted("NaCl", c0)
            self.interf = water | get_pure_gas("air")
            self.set_scaling(spatial=1 * micro * meter, temporal=1 * milli * second)
            set_electrostatic_scaling(self, potential=thermal_voltage(T), temperature=T,
                                      ion_concentration=c0)
            self.define_named_var(temperature=T)
            self.add_mesh(LineMesh(N=8, size=1 * micro * meter))
            eqs = PoissonNernstPlanck(fluid_props=water, wind=0, temperature=T)
            eqs += IntegralObservables(sigma=self.interf.surface_tension, A=1) @ "right"
            self.add_equations(eqs @ "domain")

    prob = P()
    with _fresh(prob) as p:
        p.initialise()
        o = p.get_mesh("domain/right").evaluate_all_observables()
    sigma = float(o["sigma"] / o["A"] / (milli * newton / meter))
    water, gas = get_pure_liquid("water"), get_pure_gas("air")
    sigma0 = float(water.evaluate_at_condition((water | gas).surface_tension, "IC", temperature=T)
                   / (milli * newton / meter))
    assert sigma - sigma0 == pytest.approx(1.64 * 0.1, rel=1e-6)   # 100 mM at +1.64 mN/m per M


# =================================================================================================
#  The salt as a real composition field
# =================================================================================================

def _component_mode_brine(concentration=1 * molar, salt="NaCl", T=20 * celsius):
    return Mixture(get_pure_liquid("water") + concentration * get_salt(salt),
                   temperature=T, salt_treatment="component")


def test_the_salt_becomes_a_component_with_a_mass_fraction():
    mix = Mixture(get_pure_liquid("water") + 20 * percent * get_pure_liquid("glycerol")
                  + 1 * molar * get_salt("NaCl"), temperature=20 * celsius)
    assert sorted(mix.components) == ["glycerol", "water"]          # dilute: the salt rides along
    mix.treat_salts_as_components()
    assert sorted(mix.components) == ["NaCl", "glycerol", "water"]
    assert sorted(mix.required_adv_diff_fields) == ["NaCl", "glycerol"]
    assert mix.passive_field == "water"                              # a salt is never the passive one
    ic = {k: float(v) for k, v in mix.initial_condition.items() if k.startswith("massfrac_")}
    assert sum(ic.values()) == pytest.approx(1.0, abs=1e-12)
    # 1 M NaCl is 58.44 g in a litre of a solution weighing about 1.09 kg.
    assert ic["massfrac_NaCl"] == pytest.approx(0.0535, rel=2e-2)
    assert mix.treat_salts_as_components() is None                   # idempotent


def test_the_partial_molar_volumes_add_up_to_the_measured_salt_values():
    # Stored per ion and combined by stoichiometry, the same way the ambipolar diffusivity is. The
    # measured salt values are what says the additivity holds.
    unit = (centi * meter) ** 3 / mol
    for name, ref in (("NaCl", 16.62), ("KCl", 26.85), ("CaCl2", 17.85), ("Na2SO4", 11.56),
                      ("MgSO4", -7.28), ("NaOH", -5.20), ("HCl", 17.83)):
        got = float(get_salt(name).get_apparent_molar_volume() / unit)
        assert got == pytest.approx(ref, abs=0.1)
    # Negative ones are electrostriction rather than an error: the ion pulls water in tighter than
    # it displaces, and the solution is denser than the solvent by more than the salt's own mass.
    assert float(get_salt("MgSO4").get_apparent_molar_volume() / unit) < 0


@pytest.mark.parametrize("wt,measured", [(0.05, 1034.1), (0.10, 1070.7), (0.20, 1148.0)])
def test_the_brine_density_follows_volume_additivity(wt, measured):
    # 1/rho = w_solvent/rho_solvent + w_salt*V_phi/M_salt, with V_phi at infinite dilution. That
    # volume grows with concentration, so the model is good near dilution and drifts: 0.1% at 5 wt%,
    # 1.5% at 20. Quoted rather than hidden, because it is the accuracy of this mode.
    T = 20 * celsius
    mix = _component_mode_brine()
    rho = float(mix.evaluate_at_condition(mix.mass_density,
                                          {"massfrac_water": 1 - wt, "massfrac_NaCl": wt},
                                          temperature=T) / (kilogram / meter ** 3))
    assert rho == pytest.approx(measured, rel=0.02)
    assert abs(rho / measured - 1) < (0.005 if wt <= 0.1 else 0.02)


def test_a_salt_with_no_volume_data_is_refused():
    # There is no harmless default for a volume, unlike the surface tension increment where zero
    # means "no effect".
    with pytest.raises(RuntimeError, match="No partial molar volume"):
        Mixture(get_pure_liquid("water") + 1 * milli * molar * get_salt("ZnSO4"),
                temperature=20 * celsius, salt_treatment="component")


def test_a_pure_liquid_is_told_what_to_do_instead():
    mix = Mixture(get_pure_liquid("water") + 1 * molar * get_salt("NaCl"))
    assert mix.is_pure                       # dilute mode leaves a single solvent pure
    with pytest.raises(RuntimeError, match="pure liquid cannot hold components"):
        CompositionFlowEquations(mix, salt_treatment="component")


def test_component_mode_carries_the_ion_fields_too():
    # c_<ion> means the same thing in every mode, which is what lets a surface tension law or an
    # activity coefficient be written once.
    from pyoomph.equations.salt_transport import SaltConcentrationsFromMassFractions
    mix = Mixture(get_pure_liquid("water") + 20 * percent * get_pure_liquid("glycerol")
                  + 1 * molar * get_salt("NaCl"), temperature=20 * celsius)
    eqs = CompositionFlowEquations(mix, salt_treatment="component")
    assert isinstance(eqs.get_equation_of_type(SaltConcentrationsFromMassFractions),
                      SaltConcentrationsFromMassFractions)
    # ... and the salt is transported by the composition equations, not by a second set of them.
    assert eqs.get_equation_of_type(SaltTransportEquations) is None
    compo = eqs.get_equation_of_type(CompositionAdvectionDiffusionEquations)
    assert isinstance(compo, CompositionAdvectionDiffusionEquations)
    assert "massfrac_NaCl" in compo.fieldnames


def test_both_modes_give_the_same_activity_by_different_routes():
    # The check that the mole-fraction basis factor is right rather than merely present. AIOMFAC's
    # coefficient goes with its own mole fraction, which counts the ions; pyoomph's Raoult law
    # multiplies by molefrac_*, which counts the salt-free solvents in dilute mode and the salt as
    # one particle per formula unit in component mode. The coefficient and the mole fraction must
    # therefore *both* differ between the modes, while their product -- the activity, which is the
    # physical quantity -- must not.
    T = 298.15 * kelvin
    m, r = 1.0, 0.2                       # molality of NaCl, and glycerol's share of the solvent
    M_salt = float(get_salt("NaCl").molar_mass / (kilogram / mol))
    M_w = float(get_pure_liquid("water").molar_mass / (kilogram / mol))
    M_g = float(get_pure_liquid("glycerol").molar_mass / (kilogram / mol))

    dilute = Mixture(get_pure_liquid("water") + r * get_pure_liquid("glycerol")
                     + 1 * molar * get_salt("NaCl"), temperature=T)
    dilute.set_activity_coefficients_by_unifac("AIOMFAC", use_multi_return=False)
    rho = dilute.get_reference_mass_density(T)
    ic_dilute = {"massfrac_water": 1 - r, "massfrac_glycerol": r, "temperature": T,
                 "c_Na_p": m * (mol / kilogram) * rho, "c_Cl_m": m * (mol / kilogram) * rho}
    n_w, n_g = (1 - r) / M_w, r / M_g
    x_saltfree = {"water": n_w / (n_w + n_g), "glycerol": n_g / (n_w + n_g)}

    component = Mixture(get_pure_liquid("water") + r * get_pure_liquid("glycerol")
                        + 1 * molar * get_salt("NaCl"), temperature=T, salt_treatment="component")
    component.set_activity_coefficients_by_unifac("AIOMFAC", use_multi_return=False)
    w_s = m * M_salt / (1 + m * M_salt)   # a molality is per kg of solvent, so this is exact
    ic_comp = {"massfrac_water": (1 - w_s) * (1 - r), "massfrac_glycerol": (1 - w_s) * r,
               "massfrac_NaCl": w_s, "temperature": T}
    n_wc, n_gc, n_s = (1 - w_s) * (1 - r) / M_w, (1 - w_s) * r / M_g, w_s / M_salt
    total = n_wc + n_gc + n_s
    x_component = {"water": n_wc / total, "glycerol": n_gc / total}

    for name in ("water", "glycerol"):
        g_dilute = float(dilute.evaluate_at_condition(dilute.activity_coefficients[name], ic_dilute))
        g_comp = float(component.evaluate_at_condition(component.activity_coefficients[name], ic_comp))
        assert g_dilute * x_saltfree[name] == pytest.approx(g_comp * x_component[name], rel=1e-9)
        assert abs(g_comp / g_dilute - 1) > 0.01      # the factor is not 1, so the test has teeth
    # And it is the factor the derivation says it is: 1 + n_salt/n_solvent.
    ratio = (float(component.evaluate_at_condition(component.activity_coefficients["water"], ic_comp))
             / float(dilute.evaluate_at_condition(dilute.activity_coefficients["water"], ic_dilute)))
    assert ratio == pytest.approx(1 + n_s / (n_wc + n_gc), rel=1e-9)


def _both_modes(c0, nsteps=20, shrink_to=0.6):
    dilute = _evaporate(_EvaporatingFilm(GCL=True, c0=c0, mode="dilute"),
                        nsteps=nsteps, shrink_to=shrink_to)
    # The same physical duration, not the same nondimensional one: each problem takes its velocity
    # scale from its own density, and a denser liquid would otherwise be run for longer.
    component = _evaporate(_EvaporatingFilm(GCL=True, c0=c0, mode="component"),
                           nsteps=nsteps, end_time=dilute["T_end"] * second)
    return dilute, component


def test_component_mode_conserves_the_salt_through_the_ordinary_composition_terms():
    # No salt-specific interface condition anywhere: a non-volatile component is the j_i = 0 case of
    # the term the composition equations already write for every component. The three ALE branches
    # the dilute treatment needed simply do not arise here.
    _, component = _both_modes(3 * molar)
    assert abs(component["N"] / component["N0"] - 1) < 1e-10
    assert component["c_surf"] > 4000.0        # and it really did concentrate


@pytest.mark.parametrize("c0,density_ratio,w_water", [(1 * milli * molar, 1.0001, 0.9999),
                                                      (3 * molar, 1.206, 0.761)])
def test_the_modes_agree_on_the_film_and_differ_on_what_it_is_made_of(c0, density_ratio, w_water):
    # A result worth knowing rather than assuming: with the evaporation rate prescribed, the two
    # modes thin the film *identically*, at any concentration. Under volume additivity the salt's
    # contribution to the volume is fixed -- it is conserved -- so a film losing water at rate j
    # loses volume at j/rho_solvent whatever else is dissolved in it. The dilute treatment is exact
    # for the geometry here.
    dilute, component = _both_modes(c0)
    assert component["L"] == pytest.approx(dilute["L"], rel=1e-6)
    assert component["c_surf"] == pytest.approx(dilute["c_surf"], rel=1e-6)
    # What it is not exact about is the liquid itself. At 3 molar the dilute treatment still thinks
    # this is water: same density, and a mass fraction of one. That is what feeds back into anything
    # that computes an evaporation rate rather than being handed one, into buoyancy, and into every
    # property that depends on composition.
    assert dilute["rho"] == pytest.approx(998.2, rel=2e-3)
    assert dilute["w_water"] == pytest.approx(1.0, abs=1e-12)
    assert component["rho"] / dilute["rho"] == pytest.approx(density_ratio, rel=1e-2)
    assert component["w_water"] == pytest.approx(w_water, rel=1e-2)
