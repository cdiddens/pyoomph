# A 1D Brusselator: the project's only PDE periodic-orbit case, used to measure what that assembly
# costs at PDE scale. Run as:  python bench_periodic_orbit_1d.py <outdir> [N] [NT]
#
# It validates itself against theory before measuring anything: the Hopf sits at b = 1 + a^2 = 2 with
# omega = a = 1, and the run prints both, so a wrong setup is obvious rather than silently benchmarked.
#
# Why it exists: the periodic-orbit assembly has its own routine
# (Problem::sparse_assemble_row_or_column_compressed_for_periodic_orbit), std::map-per-row, which the
# other tutorials never stress -- every one of them drives orbits from a 3-dof ODE where the whole
# solve is sub-millisecond. On this case the assembly is 64-74% of an orbit solve, because the orbit
# matrix is so sparse (0.17% at N=80/NT=60) that the factorisation is cheap by comparison. That is the
# measurement that justifies freezing the routine; see dev_docs/structural_assembly.md.
#
# Original description:
# 1D Brusselator: a genuine PDE with a Hopf bifurcation, used to measure what a PERIODIC ORBIT
# assembly costs at PDE scale. Homogeneous steady state (u,v) = (a, b/a); Hopf at b = 1 + a^2.
#
#   du/dt = Du u_xx + a - (b+1) u + u^2 v
#   dv/dt = Dv v_xx + b u - u^2 v
#
# Neumann (no-flux) on both ends, so the spatially uniform oscillatory mode is the critical one and
# the branch is a genuine limit cycle of the PDE rather than of a reduced ODE.
import sys, time, json
from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import LineMesh

class Brusselator(Equations):
    def __init__(self, a, b, Du, Dv):
        super().__init__()
        self.a, self.b, self.Du, self.Dv = a, b, Du, Dv
    def define_fields(self):
        self.define_scalar_field("u", "C2")
        self.define_scalar_field("v", "C2")
    def define_residuals(self):
        u, ut = var_and_test("u")
        v, vt = var_and_test("v")
        self.add_residual(weak(partial_t(u), ut) + weak(self.Du * grad(u), grad(ut))
                          - weak(self.a - (self.b + 1) * u + u**2 * v, ut))
        self.add_residual(weak(partial_t(v), vt) + weak(self.Dv * grad(v), grad(vt))
                          - weak(self.b * u - u**2 * v, vt))

class BrusselatorProblem(Problem):
    def __init__(self, N=40):
        super().__init__()
        self.N = N
        self.b = self.define_global_parameter(b=1.5)
        self.a, self.Du, self.Dv = 1.0, 0.02, 0.01
    def define_problem(self):
        self.add_mesh(LineMesh(N=self.N, size=1.0))
        eqs = Brusselator(self.a, self.b, self.Du, self.Dv)
        eqs += InitialCondition(u=self.a, v=self.b / self.a)
        self.add_equations(eqs @ "domain")

if __name__ == "__main__":
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    NT = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    p = BrusselatorProblem(N=N)
    with p:
        p.set_output_directory(sys.argv[1] + "/bru"); p.quiet()
        p.setup_for_stability_analysis(analytic_hessian=True)
        p.b.value = 1.5
        p.initialise(); p.solve()
        ev, _ = p.solve_eigenproblem(6)
        print("BRU base ndof=%d  leading eigenvalues=%s" % (p.ndof(), [complex(e) for e in ev[:3]]), flush=True)
        p.activate_bifurcation_tracking("b", "hopf")
        p.solve()
        print("BRU hopf at b=%.10g  omega=%.6g" % (float(p.b.value), p._get_bifurcation_omega()), flush=True)
        with p.switch_to_hopf_orbit(NT=NT) as orbit:
            _, n, nnz, _, v, c, s = p._assemble_residual_jacobian("")
            t = time.time()
            for _ in range(3): p._assemble_residual_jacobian("")
            t_asm = (time.time() - t) / 3
            t = time.time(); p.solve(); t_solve = time.time() - t
            print("BRUORBIT " + json.dumps({"N": N, "NT": NT, "orbit_ndof": int(n), "nnz": int(nnz),
                  "density_pct": 100.0*nnz/float(n)/float(n), "assembly_ms": t_asm*1e3, "orbit_solve_s": t_solve,
                  "T": float(orbit.get_T())}), flush=True)
