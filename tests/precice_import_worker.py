#  @file
#  @author Christian Diddens
#  @author Duarte Rocha
#  @author Maxim de Wildt
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

# Can the preCICE adapter be imported, and its equation classes built, on a machine that has no
# preCICE at all?
#
# It has to be. Both PreCICE_Coupling tutorials import pyoomph.solvers.precice_adapter at the top and
# assemble their coupling equations unconditionally, and then - started without a participant, which
# is how the tutorial pass runs everything - throw those equations away and solve the monolithic
# domain. That branch needs nothing but pyoomph. While the adapter imported precice at module scope,
# it needed libprecice anyway, so both tutorials were permanent skips on every machine without it,
# the GitHub runners among them. The import now lives in initialize_problem(), the one method that
# creates a participant.
#
#   --phase import_only   import the adapter and build the three equation classes; the package must
#                         NOT have been imported on the way (checked via sys.modules, so this phase
#                         is meaningful on a machine that HAS preCICE too)
#   --phase coupled       actually ask for a participant; without the package this must fail, and
#                         fail with the ModuleNotFoundError the tutorial harness recognises - a lazy
#                         import must not turn a missing dependency into a silent no-op

import argparse
import sys


def phase_import_only():
    import pyoomph.solvers.precice_adapter as adapter
    if "precice" in sys.modules:
        raise AssertionError("importing the adapter imported precice itself - the import at the top "
                             "of pyoomph/solvers/precice_adapter.py is back")
    from pyoomph.expressions import var, partial_x
    eqs = adapter.PreciceProvideMesh("Dirichlet-Mesh")
    eqs += adapter.PreciceWriteData(**{"Heat-Flux": partial_x(var("u", domain=".."))})
    eqs += adapter.PreciceReadData(uD="Temperature")
    print("ADAPTER_IMPORTED equations=%s" % type(eqs).__name__)
    try:
        import precice  # noqa: F401
        print("PRECICE_PRESENT yes")
    except ModuleNotFoundError:
        print("PRECICE_PRESENT no")


def phase_coupled():
    from pyoomph import Problem, ODEEquations, InitialCondition
    from pyoomph.expressions import var_and_test, partial_t

    class Decay(ODEEquations):
        def define_fields(self):
            self.define_ode_variable("u")

        def define_residuals(self):
            u, ut = var_and_test("u")
            self.add_weak(partial_t(u) + u, ut)

    class DecayProblem(Problem):
        def define_problem(self):
            self += (Decay() + InitialCondition(u=1.0)) @ "ode"

    with DecayProblem() as problem:
        problem.precice_participant = "Dirichlet"
        problem.precice_config_file = "there-is-no-config.xml"
        problem.precice_initialise()
    print("COUPLED_RAN")  # only reached if a participant was really created


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["import_only", "coupled"])
    args = parser.parse_args()
    {"import_only": phase_import_only, "coupled": phase_coupled}[args.phase]()
    print("PYOOMPH_WORKER_DONE")
