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

"""Which quantity validates which tutorial script, and why.

Keyed by the script's path inside the tutorial bundle, "Folder/script.py" - the same key the
harness's --only matches and its JSON report records. A script with no entry here is simply not
validated: the harness reports it as uncovered rather than failing it.

The numbers themselves are NOT here. They live in data/<Folder>/<stem>/reference.json and are
regenerated with

    python3 -u citools/test_all_tutorial_scripts.py --only <script> --update-validation

This file holds the judgement instead: which quantity is the meaningful one for this script, and -
wherever a tolerance is looser than the default - why it has to be. A relaxed tolerance with no
reason next to it is the one nobody can assess a year later, so Check takes a reason= and it is
expected to be used.

To add a script, run

    python3 -u citools/test_all_tutorial_scripts.py --only <script> --propose-validation

which prints an entry listing the data files the run left behind and what its fingerprint has to
offer. Keep the quantity that means something, drop the rest, then generate the numbers.
"""

from . import Evolution, FinalState, Fingerprint, Stdout

VALIDATION: "dict[str,list]" = {

  # ---------------------------------------------------------------------------------------------
  # Oscillators and pendulums: a fixed output grid and a smooth solution, so the whole evolution
  # is compared at the default tolerance. These are the scripts a regression in the time
  # integration, the ODE assembly or the code generation shows up in first.
  # ---------------------------------------------------------------------------------------------
  "Temporal_ODEs/predefined_harmonic_oscillator.py": [
      Evolution("harmonic_oscillator.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/custom_harmonic_oscillator.py": [
      Evolution("harmonic_oscillator.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/dimensional_oscillator_with_units.py": [
      # Dimensional: the columns carry [s] and [m], so this also pins the scaling down.
      Evolution("harmonic_oscillator.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/custom_math_driven_oscillator.py": [
      Evolution("harmonic_oscillator.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/custom_math_dimensional_tennis.py": [
      # The ball bounces between two moving walls, i.e. the solution is only piecewise smooth, and
      # the time stepper has to work its way through each bounce. That makes the whole trajectory
      # depend on the exact arithmetic: over four ranks it took different steps from the first
      # bounce on and ended 0.4% away. Serially it is reproducible to the last digit, so the check
      # is kept and excused in the MPI pass rather than loosened everywhere.
      Evolution("ball.txt", skip_under=("mpi",),
                reason="the steps through each bounce differ over several ranks"),
      Fingerprint(skip_under=("mpi",),
                  reason="the steps through each bounce differ over several ranks"),
  ],
  "Temporal_ODEs/van_der_pol_method_1.py": [
      Evolution("vdPol_oscillator.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/van_der_pol_method_2.py": [
      # The same limit cycle as method 1, spelled out differently - and the two files are compared
      # against their own references, not against each other, so a drift in either is caught.
      Evolution("vdPol_oscillator.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/coupled_oscillators_method_1.py": [
      Evolution("coupled_oscillator.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/coupled_oscillators_method_2.py": [
      Evolution("coupled_oscillator.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/coupled_oscillators_method_3.py": [
      # Two separate ODE domains coupled across, hence two files.
      Evolution("oscillator1.txt"),
      Evolution("oscillator2.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/pendulum_generalized_coordinate.py": [
      Evolution("pendulum.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/pendulum_lagrange_multiplier.py": [
      # The same pendulum as the generalized-coordinate one, but constrained: the Lagrange
      # multiplier column is the interesting one, since it is what the constraint produces.
      Evolution("pendulum.txt"),
      Fingerprint(),
  ],

  # ---------------------------------------------------------------------------------------------
  # Time stepping schemes. Comparing the schemes against each other IS the tutorial, so every
  # scheme's own output is compared rather than one representative: a scheme that silently falls
  # back to another would otherwise go unnoticed.
  # ---------------------------------------------------------------------------------------------
  "Temporal_ODEs/oscillator_TPZ_scheme.py": [
      # The energy column is the point of this one: TPZ conserves it, BDF2 does not.
      Evolution("harmonic_oscillator.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/oscillator_fully_implicit_schemes.py": [
      Evolution("osci_timestepping_BDF1/harmonic_oscillator.txt"),
      Evolution("osci_timestepping_BDF2/harmonic_oscillator.txt"),
      Evolution("osci_timestepping_Newmark2/harmonic_oscillator.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/time_stepping_schemes.py": [
      Evolution("osci_timestepping_scheme_BDF1/anharmonic_oscillator.txt"),
      Evolution("osci_timestepping_scheme_BDF2/anharmonic_oscillator.txt"),
      Evolution("osci_timestepping_scheme_Boole/anharmonic_oscillator.txt"),
      Evolution("osci_timestepping_scheme_MPT/anharmonic_oscillator.txt"),
      Evolution("osci_timestepping_scheme_Simpson/anharmonic_oscillator.txt"),
      Evolution("osci_timestepping_scheme_TPZ/anharmonic_oscillator.txt"),
      Evolution("osci_timestepping_scheme_Newmark2/anharmonic_oscillator.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/parallel_running.py": [
      # Six sequential runs at different spring constants, each into its own output directory.
      # (The harness does not start this script under --mpirun at all: it spawns its own.)
      Evolution("parallel_running/dim_osci_seq_run_k_0.1/harmonic_oscillator.txt"),
      Evolution("parallel_running/dim_osci_seq_run_k_0.2/harmonic_oscillator.txt"),
      Evolution("parallel_running/dim_osci_seq_run_k_0.5/harmonic_oscillator.txt"),
      Evolution("parallel_running/dim_osci_seq_run_k_1/harmonic_oscillator.txt"),
      Evolution("parallel_running/dim_osci_seq_run_k_2/harmonic_oscillator.txt"),
      Evolution("parallel_running/dim_osci_seq_run_k_5/harmonic_oscillator.txt"),
  ],

  # ---------------------------------------------------------------------------------------------
  # Bifurcations. These mostly write nothing at all, and their answer is a global parameter value -
  # the r of the fold, the rho of the Hopf point - which is exactly what the fingerprint records
  # alongside the eigenvalue that vanishes there. For these the fingerprint IS the validation.
  # ---------------------------------------------------------------------------------------------
  "Temporal_ODEs/bifurcation_transient_transcritical.py": [
      Evolution("transcritical.txt"),
      Fingerprint(),
  ],
  "Temporal_ODEs/bifurcation_stationary_transcritical.py": [
      Fingerprint(),
  ],
  "Temporal_ODEs/bifurcation_eigenvalues_transcritical.py": [
      # Two start points, each solved to a stationary state whose stability is then read off its
      # eigenvalue. Only the last of the two is in the fingerprint, so the printed value of the
      # first is picked up from the output - the one place it exists.
      Stdout(r"stationary solution x=([-\d.eE+]+) with r=([-\d.eE+]+)"),
      Fingerprint(),
  ],
  "Temporal_ODEs/bifurcation_fold_tracking.py": [
      # r at the fold, and the eigenvalue that is zero there.
      Fingerprint(),
  ],
  "Temporal_ODEs/bifurcation_pitchfork_tracking.py": [
      Fingerprint(),
  ],
  "Temporal_ODEs/bifurcation_hopf_tracking_lorenz.py": [
      # Three parameters (beta, rho, sigma) and the purely imaginary eigenvalue: the Hopf point of
      # the Lorenz system, which is the strongest single number in this chapter.
      #
      # The state is deliberately NOT compared. The Lorenz system is invariant under
      # (x,y,z) -> (-x,-y,z), so the Hopf point sits on a symmetric pair of branches and the Newton
      # solve may land on either: over four ranks it converged to the mirror solution, with x and y
      # exactly negated. The parameters were identical to the last digit, which is the point - the
      # bifurcation is the answer here, the representative of it is not.
      Fingerprint(only=["params.*", "eigenvalues.*", "ndof"],
                  reason="the state is fixed only up to the Lorenz symmetry (x,y,z)->(-x,-y,z)"),
  ],
  "Temporal_ODEs/bifurcation_fold_arclength.py": [
      # An arclength continuation turns around at the fold, so its first column (r) does not grow
      # and the rows have to be matched by index.
      Evolution("fold.txt", match="rows", reason="the continuation parameter turns around at the fold"),
      Fingerprint(),
  ],
  "Temporal_ODEs/bifurcation_fold_arclength_eigen.py": [
      Evolution("fold.txt", match="rows", reason="the continuation parameter turns around at the fold"),
      # r, x and the critical eigenvalue along the whole branch, written by hand and without a
      # header, so the columns are named by position. This is the strong check: it says that the
      # eigenvalue really does cross zero where the branch turns around.
      Evolution("fold_with_eigen.txt", match="rows",
                reason="a continuation in r, which turns around at the fold"),
      Fingerprint(),
  ],
  "Temporal_ODEs/bifurcation_pitchfork_arclength_eigen.py": [
      Evolution("pitchfork.txt", match="rows", reason="a continuation, so the time column stands still"),
      # Both sides of the super- and the subcritical pitchfork, with the eigenvalue along each.
      Evolution("super_with_eigen_1.txt", match="rows", reason="a continuation in r"),
      Evolution("super_with_eigen_2.txt", match="rows", reason="a continuation in r"),
      Evolution("sub_with_eigen_1.txt", match="rows", reason="a continuation in r"),
      Evolution("sub_with_eigen_2.txt", match="rows", reason="a continuation in r"),
      Fingerprint(),
  ],
  "Temporal_ODEs/bifurcation_transcritital_arclength_eigen.py": [
      Evolution("transcritical.txt", match="rows", reason="a continuation, so the time column stands still"),
      # Both branches through the transcritical point, with the eigenvalue along each.
      Evolution("trans_with_eigen_1.txt", match="rows", reason="a continuation in r"),
      Evolution("trans_with_eigen_2.txt", match="rows", reason="a continuation in r"),
      Fingerprint(),
  ],
  "Temporal_ODEs/bifurcation_branch_switching.py": [
      # All four branches, because switching onto the right one is what this script is about: a
      # branch that comes out as a copy of its neighbour is the failure worth catching.
      Evolution("branch_trivial.txt", match="rows", reason="a continuation in r, which turns around"),
      Evolution("branch_nontrivial.txt", match="rows", reason="a continuation in r, which turns around"),
      Evolution("branch_pitchfork_plus.txt", match="rows", reason="a continuation in r, which turns around"),
      Evolution("branch_pitchfork_minus.txt", match="rows", reason="a continuation in r, which turns around"),
      Evolution("ode.txt", match="rows",
                reason="the transient is restarted from several initial conditions, so the time "
                       "column jumps back"),
      Fingerprint(),
  ],
  "Temporal_ODEs/deflated_solve.py": [
      Fingerprint(),
  ],
  "Temporal_ODEs/deflated_continuation.py": [
      # One file per branch the deflation found, written without a header (hence the positional
      # column names): finding all four branches, and no spurious fifth, is the whole point.
      Evolution("branch_00.txt", match="rows", reason="a continuation in r, which turns around"),
      Evolution("branch_01.txt", match="rows", reason="a continuation in r, which turns around"),
      Evolution("branch_02.txt", match="rows", reason="a continuation in r, which turns around"),
      Fingerprint(),
  ],
  "Temporal_ODEs/pendulum_gencoord_eigenvalues.py": [
      # The two eigenvalues of the linearised pendulum, i.e. +-i*sqrt(g/L).
      Fingerprint(),
  ],
  "Temporal_ODEs/pendulum_lagrange_eigenvalues.py": [
      # The same spectrum, from the constrained formulation: five dofs, five eigenvalues, and the
      # physical pair (+-i, indices 1 and 2 of the sorted spectrum) has to come out the same as in
      # the generalized-coordinate script above.
      #
      # The other three are the "infinite" eigenvalues a differential-algebraic system has, which
      # the discretisation renders as about -4.1e5 and +2.1e5 +- 3.6e5i: they are set by the
      # conditioning of the constraint rather than by the pendulum, and they moved by 6.6% between
      # a serial run and the same run over four ranks. Pinning them down would be pinning down an
      # artefact, so they are left out here rather than excused with a loose tolerance.
      Fingerprint(skip=["eigenvalues.0.*", "eigenvalues.3.*", "eigenvalues.4.*"],
                  reason="the three remaining eigenvalues are the constraint's 'infinite' ones, "
                         "whose finite value is a property of the discretisation"),
  ],

  # ---------------------------------------------------------------------------------------------
  # Periodic orbits and Floquet multipliers.
  # ---------------------------------------------------------------------------------------------
  "Temporal_ODEs/langford_floquet.py": [
      # The best check in this chapter: floquet.txt holds the numerically computed Floquet
      # multiplier AND the analytical one side by side over 60 values of mu, so comparing it pins
      # down the orbit, its period and the multiplier extraction at once.
      #
      # Serially. Over four ranks the continuation lands on mu values that differ in the fifth
      # decimal, and the orbit it ends on is at a different phase (y comes out with the opposite
      # sign), so neither the table nor the final state lines up. Where it stopped still does, to
      # within that drift, and keeping that one loose check means the MPI pass is not blind here.
      Evolution("floquet.txt", skip_under=("mpi",),
                reason="the continuation lands on slightly different mu values over several ranks"),
      Fingerprint(skip_under=("mpi",),
                  reason="the run ends at a different phase of the same orbit over several ranks"),
      Fingerprint(only=["params.*"], rtol=1e-3,
                  reason="the mu the continuation stops at drifts by about 4e-5 between a serial "
                         "run and four ranks; the tight comparison is the Fingerprint above, which "
                         "runs serially"),
  ],
  "Temporal_ODEs/hopf_switch.py": [
      # The orbit files are named after the rho they were found at, so their names are not a stable
      # key - the fingerprint's rho and the orbit's dof extent are.
      Fingerprint(),
  ],
  "Temporal_ODEs/manual_orbit.py": [
      # 730 dofs: the orbit is discretized in time, so the fingerprint's per-dof-type extent is the
      # shape of the whole orbit rather than one state.
      #
      # The orbit is picked out of a chaotic transient, which is as path-dependent as it sounds:
      # over four ranks the script arrived at a 727-dof orbit instead of a 730-dof one, mirrored by
      # the Lorenz symmetry on top. rho - where the orbit was found - agrees, and that is what
      # stays checked everywhere.
      Evolution("lorenz_smoothed.txt", skip_under=("mpi",),
                reason="the orbit is found after a chaotic transient, so its discretisation "
                       "differs over several ranks"),
      Fingerprint(skip_under=("mpi",),
                  reason="727 dofs instead of 730, and mirrored by the Lorenz symmetry"),
      Fingerprint(only=["params.*"]),
  ],

  # ---------------------------------------------------------------------------------------------
  # Chaos. These cannot be validated the way the rest of the chapter can, and pretending otherwise
  # would produce a reference that fails on the next machine for reasons that are not bugs.
  # ---------------------------------------------------------------------------------------------
  "Temporal_ODEs/adaptive_lorenz_attractor.py": [
      # Chaotic AND adaptively stepped, which is the worst combination available: outstep=True
      # writes a row per step, so the instants are a property of the machine, and by t~40 a
      # round-off difference in the initial condition has grown to the size of the attractor.
      # What is left is the first stretch of the trajectory, interpolated onto the recorded
      # instants - and the interpolation over an adaptive BDF2 step is itself the error that sets
      # the tolerance here, not the time integration.
      Evolution("lorenz_attractor.txt", until_time=10.0, match="interp", rtol=2e-2,
                reason="adaptive output instants, so the comparison interpolates, and h^2*y'' of a "
                       "Lorenz trajectory is percent-level"),
      Fingerprint(only=["ndof"],
                  reason="the final state at t=100 is a chaotic trajectory's position, which is "
                         "not reproducible across platforms by any tolerance"),
  ],
  "Temporal_ODEs/lorenz_lyapunov.py": [
      # The three Lyapunov exponents of the Lorenz attractor, from the tail of the run where they
      # have settled. A time average over a chaotic trajectory: the trajectory itself decorrelates
      # between platforms, so only the average converges, and only to within its sampling error at
      # t=200. Loose as it is, this still pins the sign structure (one positive, one zero, one
      # strongly negative) and the Gram-Schmidt reorthonormalisation that produces it.
      # The file has no header, hence the positional column names.
      #
      # This script is what found the aliasing in PETSCSolver._aux_backsolve() - every right-hand
      # side of a multi-solve came back as an alias of one PETSc Vec, so all k perturbations were
      # identical and the file was pure NaN over two ranks or more. That is fixed, and
      # tests/test_mpi_lyapunov.py is the gate for it (it compares the exponents ACROSS regimes at
      # atol=1e-7, which it can do because its problem is linear).
      #
      # Excused in the MPI pass all the same, and for a reason that no tolerance can fix. Two runs
      # of a CHAOTIC trajectory that differ in the last bit have separated completely by t=40, so
      # from then on each is averaging over a different stretch of the attractor and the two
      # finite-time estimates differ by their sampling error - measured at up to 9 % over the
      # window below, against 4-digit agreement at t=30 while the runs were still correlated. The
      # serial comparison is the strong one: two independent serial generations of this chapter came
      # out bit-identical.
      Evolution("lyapunov.txt", from_time=195.0, rtol=5e-2, skip_under=("mpi",),
                reason="a time average over a chaotic trajectory, still converging as 1/t at t=200, "
                       "and sampled over a different stretch of the attractor once a run on another "
                       "rank count has decorrelated"),
  ],
  "Temporal_ODEs/langford_time_integration.py": [
      # Same situation: a long chaotic transient, then orbits whose output directories are named
      # after the mu they were found at. Only mu itself is a stable quantity here.
      Fingerprint(only=["ndof", "params.*"],
                  reason="the trajectory is chaotic, so its final state is not reproducible"),
  ],

  # ===============================================================================================
  # Spatial_PDEs: stationary problems, so there is no evolution to compare - what is checked is the
  # solved field. The 1d Poisson family writes its nodes to a text file; everything else writes vtu
  # only, and for those the fingerprint's per-dof-type extent IS the solution (a Stokes script's
  # velocity extremes and pressure norm pin the field down without a single node position).
  # ===============================================================================================

  # --- the 1d Poisson family, on a LineMesh(minimum=-1, size=2, N=100) ------------------------
  "Spatial_PDEs/poisson.py": [
      # -u'' = 1 on [-1,1] with u(+-1)=0, i.e. u = (1-x^2)/2 and max(u) = 1/2 exactly; the recorded
      # fingerprint has 0.4999999999999123, so the discretisation is exact for this quadratic on C2
      # and what the reference pins is round-off.
      #
      # reduce=False here and nowhere else in this chapter: node by node over 201 nodes, which is
      # the strongest form the check has and the one case where it is clearly safe - a fixed 1d
      # LineMesh, written through a single output with no adaptation. It is also the only production
      # use of that path, which otherwise only tests/test_tutorial_validation.py exercises.
      FinalState("domain_*.txt", reduce=False),
      Fingerprint(),
  ],
  "Spatial_PDEs/poisson_neumann.py": [
      # The same source with a Neumann flux on one side, so the solution is no longer symmetric.
      FinalState("domain_*.txt"),
      Fingerprint(),
  ],
  "Spatial_PDEs/poisson_coupled.py": [
      # Two Poisson equations sourcing each other (u by w, w by -10u), so a regression in the
      # coupling shows up in one field and not the other - which is why both columns are compared.
      FinalState("domain_*.txt"),
      Fingerprint(),
  ],
  # These two impose the SAME Robin condition by two different mechanisms - a Lagrange multiplier
  # pair, and an equivalent Neumann flux - and the tutorial's point is that they agree. Each is
  # compared against its own reference, so a drift in either is caught; that they agree with EACH
  # OTHER is then a consequence, and the dof counts differ (203 vs 201) because the multiplier is
  # itself an unknown.
  "Spatial_PDEs/poisson_robin_via_lagrange.py": [
      FinalState("domain_*.txt"),
      Fingerprint(),
  ],
  "Spatial_PDEs/poisson_robin_via_neumann.py": [
      FinalState("domain_*.txt"),
      Fingerprint(),
  ],
  "Spatial_PDEs/poisson_pure_neumann_nullspace.py": [
      # Pure Neumann: the problem is singular up to a constant, and a Lagrange multiplier pins the
      # average value. The multiplier is the scalar that says the nullspace was removed correctly,
      # and it is the whole content of lambda_space.txt - one row, hence match="rows" (a single
      # instant cannot be looked up by a growing abscissa).
      FinalState("domain_*.txt"),
      Evolution("lambda_space.txt", match="rows", reason="the file holds one row, the converged "
                                                         "multiplier"),
      Fingerprint(),
  ],

  # --- 2d/axisymmetric Poisson, vtu only ------------------------------------------------------
  "Spatial_PDEs/poisson_2d.py": [
      Fingerprint(),
  ],
  # The two adaptive ones. ndof is part of the fingerprint on purpose: it pins the refinement
  # pattern, and a changed error estimator or a changed refinement decision is exactly the kind of
  # regression worth catching. It is also the first thing to relax (skip=["ndof", "dofs.*.n"]) if
  # another platform's estimator legitimately stops at a different element count - which cannot be
  # known from one machine, so it is left strict until a nightly elsewhere says otherwise.
  "Spatial_PDEs/poisson_2d_adaptive.py": [
      Fingerprint(),
  ],
  "Spatial_PDEs/poisson_axisymm_adaptive.py": [
      Fingerprint(),
  ],
  "Spatial_PDEs/cr_static_condensation.py": [
      # Crouzeix-Raviart with the interior dofs condensed out. The dof count is the point of the
      # script, so the fingerprint's ndof is the quantity that would notice condensation breaking.
      Fingerprint(),
  ],
  "Spatial_PDEs/helmholtz_pml.py": [
      # A perfectly matched layer: the field has to decay inside the layer rather than reflect, and
      # a reflection would move the extremes of both dof groups.
      Fingerprint(),
  ],

  # --- meshing ---------------------------------------------------------------------------------
  # These solve a Poisson problem on a hand-built mesh; the mesh IS what the script demonstrates, so
  # the dof count and the field extent together say the template still produces the same mesh.
  "Spatial_PDEs/mesh_Lshape_by_hand.py": [
      Fingerprint(),
  ],
  "Spatial_PDEs/mesh_fish_dimensional_curved.py": [
      Fingerprint(),
  ],
  "Spatial_PDEs/mesh_helical_line.py": [
      Fingerprint(),
  ],
  # ...and these two get their mesh from gmsh, which is an unpinned dependency (pygmsh>=7.1.17 in
  # pyproject.toml; 4.15.1 on the machine the reference was taken on). A different gmsh meshes the
  # fish differently, which moves the dof count outright and the field extremes with it, so pinning
  # either would make the check a test of the gmsh version. What is left is loose but not empty: a
  # broken boundary condition or a sign error in the weak form moves a Poisson solution by O(1),
  # which 1 % catches comfortably.
  "Spatial_PDEs/mesh_gmsh_fish_mesh_modes.py": [
      Fingerprint(skip=["ndof", "dofs.*.n"], rtol=1e-2,
                  reason="the mesh comes from gmsh, whose version is not pinned, so the element "
                         "count and with it the discretisation error are not ours to fix"),
  ],
  "Spatial_PDEs/mesh_gmsh_fish_with_holes.py": [
      Fingerprint(skip=["ndof", "dofs.*.n"], rtol=1e-2,
                  reason="the mesh comes from gmsh, whose version is not pinned, so the element "
                         "count and with it the discretisation error are not ours to fix"),
  ],

  # --- Stokes ----------------------------------------------------------------------------------
  # Saddle-point systems, vtu only. The fingerprint separates velocity_x, velocity_y and pressure
  # into their own dof types, so it says what a single norm would not: a pressure mode that drifts
  # while the velocity stays put is visible here.
  "Spatial_PDEs/stokes.py": [
      Fingerprint(),
  ],
  "Spatial_PDEs/stokes_dimensional.py": [
      # The same flow with units, so this also pins the scaling: a wrong non-dimensionalisation
      # changes the dof extremes while the vtu still looks plausible.
      Fingerprint(),
  ],
  "Spatial_PDEs/stokes_pressure_fix.py": [
      # Pressure fixed at one point instead of by a constraint - the thing that would break here is
      # the pressure level, which is its own dof type in the fingerprint.
      Fingerprint(),
  ],
  "Spatial_PDEs/stokes_no_normal_flow.py": [
      Fingerprint(),
  ],
  "Spatial_PDEs/stokes_nonnewtonian.py": [
      # Same mesh and dof count as stokes.py (862) but a shear-rate-dependent viscosity, so the
      # velocity extremes are what distinguishes the two - and would notice the constitutive law
      # silently reverting to Newtonian.
      Fingerprint(),
  ],
  "Spatial_PDEs/stokes_flow_around_object.py": [
      # 25055 dofs, the largest in the chapter.
      Fingerprint(),
  ],

  # --- the cavity inverse-problem pair ---------------------------------------------------------
  "Spatial_PDEs/cavity_forward_problem.py": [
      # U_vs_T.txt is the forward sweep: the observable U at each prescribed T, which is the curve
      # the inverse problem below then inverts. T is set by the script, so the abscissa is exact.
      Evolution("U_vs_T.txt"),
      Fingerprint(),
  ],
  "Spatial_PDEs/cavity_inverse_problem.py": [
      # The answer is the parameter the inverse problem recovers, and the fingerprint records every
      # global parameter - here Udesired - alongside the state it was recovered from.
      Fingerprint(),
  ],
}
