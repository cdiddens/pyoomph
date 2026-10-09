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

  # ===============================================================================================
  # SpatioTemporal_PDEs: transient fields. The nodal output is numbered per output step, so
  # FinalState takes the last one - the instants are prescribed by run(), so which file that is does
  # not depend on the machine even where the time STEPPING is adaptive.
  #
  # Where a script writes a line-per-output observable file on an adaptive time grid, the series is
  # REDUCED rather than matched row by row: FinalState over such a file compares min/max/mean/l2 of
  # each column, which is invariant under the grid and still says what the run did (the deepest
  # pinch, the largest fragment count, the volume's bounds). That is the honest check for a quantity
  # whose instants are a property of the machine, and it is what match="interp" would only
  # approximate at a tolerance wide enough to hide a real drift.
  # ===============================================================================================

  "SpatioTemporal_PDEs/wave_eq.py": [
      # 1d wave equation on a fixed mesh; the last output is the field after the pulse has travelled.
      FinalState("domain_*.txt"),
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/wave_eq_doubleslit.py": [
      # The screen is the physics: I is the intensity, so its extremes and l2 are the interference
      # pattern. A regression in the slit geometry or the wave speed moves the fringes and with them
      # these numbers.
      FinalState("domain__screen_*.txt"),
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/wave_eq_drums.py": [
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/convdiffu_simple.py": [
      # Adaptive in both space and time. ndof is left strict for the same reason as the adaptive
      # Spatial_PDEs scripts: it pins the refinement, and it is the first thing to relax if another
      # platform's error estimator stops elsewhere.
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/convdiffu_SUPG.py": [
      # SUPG against the unstabilised scheme: what distinguishes them is the over- and undershoot at
      # the front, i.e. exactly min(c) and max(c) of the last profile. A stabilisation that silently
      # stopped being applied would show up there first.
      FinalState("domain_*.txt"),
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/lubrication.py": [
      FinalState("domain_*.txt"),
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/lubrication_spreading.py": [
      # 1001 nodes, adaptive in time; the last output is at a prescribed instant all the same.
      FinalState("domain_*.txt"),
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/lubrication_coalescence.py": [
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/marangoni_instability.py": [
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/navier_stokes.py": [
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/navier_stokes_around_object.py": [
      # globals.txt is 11 rows at the ten prescribed output times, so the abscissa is exact. UStokes
      # is the Stokes-drag velocity the object settles at, which is the quantity the script is about.
      Evolution("globals.txt"),
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/rayleigh_taylor_instability.py": [
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/heated_cylinder.py": [
      # Two problems, one per refinement criterion, each into its own output directory - comparing
      # the criteria IS the script, so both are pinned.
      #
      # Serially only, for the same reason as moffatt_eddies below: the refinement is rank-dependent.
      # Over four ranks problem 0 stopped at 74167 dofs against 74146 and problem 1 at 18257 nodes
      # against 18237, and the field statistics follow the mesh - velocity_y's l2 by 0.51 %, the
      # near-zero tracer l2 on the cylinder by 0.43 %. A refinement criterion that is evaluated per
      # partition is a different criterion, so there is nothing here for a tolerance to fix.
      Fingerprint(skip_under=("mpi",),
                  reason="the adaptation stops at a different mesh over four ranks: 74167 dofs "
                         "against 74146"),
      Fingerprint(index=1, skip_under=("mpi",),
                  reason="the adaptation stops at a different mesh over four ranks: 18257 nodes "
                         "against 18237"),
  ],
  "SpatioTemporal_PDEs/moffatt_eddies.py": [
      # Likewise two, one per corner angle k.
      #
      # Serially only: one of the two adaptive scripts in these two chapters whose refinement is
      # RANK-dependent (heated_cylinder above is the other). Over four ranks it stopped at 73740
      # dofs against 73705, 35 more, and every field statistic follows from the different mesh -
      # the pressure l2 by 1.1e-4, its mean by 1.3 %, and problem 1's near-zero pressure mean by a
      # factor of four. Not a tolerance problem: a different mesh is a different discretisation.
      #
      # Worth recording that the two are the exception. convdiffu_simple, marangoni_instability,
      # navier_stokes, lubrication_coalescence, laplace_smoothed_mesh and cantilever all adapt too
      # and all came through the four-rank pass untouched, so rank-dependent refinement is not a
      # general property of the adaptation - which is why it is excused on these two rather than
      # the counts being dropped from every adaptive entry.
      Fingerprint(skip_under=("mpi",),
                  reason="the adaptation stops at a different mesh over four ranks: 73740 dofs "
                         "against 73705"),
      Fingerprint(index=1, skip_under=("mpi",),
                  reason="the adaptation stops at a different mesh over four ranks"),
  ],
  "SpatioTemporal_PDEs/kuramoto_sivanshinsky.py": [
      # This script starts from a DIFFERENT INITIAL CONDITION on every run, and that - not round-off
      # amplified by chaos - is why nothing in its final state can be pinned. The initial condition
      # is a DeterministicRandomField with no seed= passed, and "deterministic" there means only
      # that evaluating the field twice at the same point agrees (which pyoomph requires of any
      # CustomMathExpression); the cloud itself is redrawn per run. The script writes no observable
      # series to fall back on either, vtu only.
      #
      # So each run samples the L=50 attractor from an independent start, and what the fingerprint
      # would compare is an ensemble, not a trajectory. Measured over eight runs: h's l2 spreads by
      # 23 % of itself, its max by 22 % and its mean by 8.5 % of the field's extent, and the
      # boundary lines are worse still (left/h's max by 55 %). Meanwhile every per-group node count
      # and ndof came out identical in all eight.
      #
      # Eight and not two deliberately. The first pair of runs happened to agree to 4.4e-05 on h's
      # l2, which would have justified pinning it at rtol=1e-3 with apparent margin; the spread is
      # four orders of magnitude wider than that pair showed. Two runs cannot bound the spread of a
      # quantity whose distribution is the thing being measured - the same trap as
      # Moving_Mesh/beads_on_string.py's bimodal max(z_min).
      #
      # ndof is therefore all that is compared, and it is worth keeping: the mesh is fixed, so it
      # says the run was set up and reached the end. That is a weak check and is meant to read as
      # one. Seeding the field would make this script fully checkable, but a tutorial is
      # documentation and is not rewritten to suit its test.
      Fingerprint(only=["ndof"],
                  reason="an unseeded DeterministicRandomField initial condition, so every run "
                         "starts from a different state; over eight runs h's l2 spreads by 23 %"),
  ],
  "SpatioTemporal_PDEs/kuramoto_sivanshinsky_bifurcation.py": [
      # A fold of the hexagonal state, continued in delta and gamma. The parameters at the fold are
      # the answer, and they are what the fingerprint records alongside the critical eigenvalue.
      Evolution("hexfold.txt", match="rows", reason="a continuation, so the first column does not grow"),
      # The script ends in the AUGMENTED fold-tracking state, so the dof vector carries the null
      # eigenvector as well: the "(not described)" group is its 7614 components plus the one
      # continuation unknown (which is why that group's max is exactly gamma). A null eigenvector is
      # defined only up to sign, and over four ranks it came out with the other one - measured, not
      # inferred: the sum of its components is +4.981098 serially and -4.981063 there, a ratio of
      # -0.999993, and the group's min moved from -0.10261 to -0.08665, which is precisely the
      # negated positive extreme. So min, max and mean of that group say which sign the eigensolver
      # happened to return, and nothing about the fold.
      #
      # l2 and n are kept, because neither depends on the orientation: l2 is exactly invariant under
      # a sign flip and n is structural. Skipped unconditionally rather than skip_under=("mpi",) -
      # the sign is not a property of the rank count, and another BLAS would be as free to flip it.
      # Everything that IS the answer stays compared: delta and gamma at the fold, the critical
      # eigenvalue, ndof, and both physical fields, all of which came through the four-rank pass.
      #
      # kuramoto_sivanshinsky_arclength_eigen.py needs none of this: it ends on a plain state and
      # has no unnamed group at all.
      Fingerprint(skip=["dofs.(not described).min",
                        "dofs.(not described).max",
                        "dofs.(not described).mean"],
                  reason="a null eigenvector is defined only up to sign, and over four ranks the "
                         "eigensolver returned the other one (component sum ratio -0.999993)"),
  ],
  "SpatioTemporal_PDEs/kuramoto_sivanshinsky_arclength_eigen.py": [
      # gamma, h_rms, and the real and imaginary parts of the critical eigenvalue along the branch.
      # Serially that reproduces exactly; across rank counts the eigenvalue does not. Measured
      # against four ranks: Re(eigenvalue) came out 0.00370 where the reference holds 0.00428, which
      # is 1.85 % of that column's whole range (-0.0311..0.0164) - the accuracy of a near-zero
      # eigenvalue of a 7614-dof system, not a drift. No tolerance should cover that and still be a
      # check, so the strong serial comparison is kept and the MPI pass leaves this one to the
      # fingerprint, which does hold there.
      Evolution("hexdots.txt", match="rows", skip_under=("mpi",),
                reason="a continuation, so the first column does not grow; and the critical "
                       "eigenvalue moves by ~2 % of its range across rank counts"),
      Fingerprint(),
  ],
  "SpatioTemporal_PDEs/viscoelastic_cylinder.py": [
      # The best-anchored check in the tutorial: cylinder_drag.txt holds the drag on the confined
      # cylinder at Wi = 0.1, 0.5 and 0.7 (the three rows of Claus & Phillips 2013 Fig. 12), and
      # their Table 3, P=18 column gives 130.364 at Wi=0.1. tests/test_viscoelastic_cylinder.py
      # measures this mesh at +0.013 % to +0.187 % of those values over Wi=0.1..0.5, so what the
      # reference pins here is a published benchmark and not just yesterday's output.
      #
      # match="rows": the three rows are stationary solves at successive Wi, so the time column
      # stands still and there is no abscissa to look an instant up by.
      Evolution("cylinder_drag.txt", match="rows",
                reason="three stationary solves at successive Wi, so the time column stands still"),
      Fingerprint(),
  ],

  # ===============================================================================================
  # Moving_Mesh: the mesh positions are unknowns too, so the fingerprint's coordinate dof types are
  # part of the solution rather than background. Four of these remesh mid-run, which changes ndof
  # while the script is running; their entries say what that costs.
  # ===============================================================================================

  "Moving_Mesh/ALE_correction.py": [
      Fingerprint(),
  ],
  "Moving_Mesh/free_surface.py": [
      Fingerprint(),
  ],
  "Moving_Mesh/laplace_smoothed_mesh.py": [
      # The smoother IS the script: what it produces is the node positions, which are dofs here, so
      # the coordinate groups' extents are the thing being checked.
      Fingerprint(),
  ],
  "Moving_Mesh/solid_oscillations.py": [
      Fingerprint(),
  ],
  "Moving_Mesh/cantilever.py": [
      # Loaded by the global parameter P, whose value the fingerprint records with the deflection.
      Fingerprint(),
  ],
  "Moving_Mesh/compressed_disc.py": [
      # disc_output.txt is a compression sweep: P against the numerically computed radius AND the
      # linear-theory prediction, side by side. Comparing it therefore checks the two against each
      # other as well as against the reference, which is what the script is for. P is set by the
      # script and grows, so the abscissa is exact.
      Evolution("disc_output.txt"),
      Fingerprint(),
  ],
  "Moving_Mesh/remeshing.py": [
      Fingerprint(),
  ],
  "Moving_Mesh/beads_on_string.py": [
      # The interface shape after the beads have formed, and the minimum radius over the run. The
      # series is reduced rather than matched: this script is adaptive in time AND remeshes, so its
      # instants are a property of the machine while min(r_min) - how far the neck thinned - is not.
      # Two runs of this script agree on the interface reductions only to about 1e-5 - measured, by
      # running it twice: it is adaptive in time AND remeshes, and a remesh decision near the bead
      # necks flips on round-off. 1e-3 is therefore the honest tolerance here, and it still catches
      # a bead that forms in the wrong place.
      FinalState("liquid__interface_*.txt", rtol=1e-3,
                 reason="two runs agree to ~1e-5 on these reductions; the script remeshes and the "
                        "decision near a neck flips on round-off"),
      # minimum.txt is deliberately NOT checked, and it took three attempts to establish why.
      # Reducing the whole series fails because an adaptive run writes a machine-dependent number of
      # rows (the l2 of z_min moved by 3.3 %), so that was narrowed to the extremes - and the
      # extremes are bimodal: over five repeats max(z_min) came out 21.0198 three times and
      # 18.8496 twice, nothing in between. z_min is the axial POSITION of the thinnest neck, and a
      # beads-on-string jet has several necks competing for that title, so which one wins is a
      # discrete choice that round-off decides. No tolerance covers a discrete branch, and widening
      # one to 11 % would not be a check any more.
      #
      # The two checks above cover the physics regardless: the interface shape is what the script
      # produces, and the fingerprint carries the whole state.
      # The fingerprint gets the same 1e-3 as the interface, and for the same measured reason. At
      # the default 1e-5 it sat exactly on this script's reproducibility and failed intermittently:
      # liquid/bottom/log_conformation_xy.l2 came out 1.47e-5 apart on one repeat, which is the
      # script's own run-to-run spread and not a drift. Note that the field-scale rule does not help
      # there and should not - an l2 IS the field's scale, so the comparison on it is already the
      # right question; what was wrong was the tolerance.
      Fingerprint(rtol=1e-3,
                  reason="two runs agree to ~1e-5 at best; the script remeshes and is adaptive in "
                         "time, so a decision near a neck flips on round-off"),
  ],
  "Moving_Mesh/rayleigh_plateau.py": [
      # Pinch-off is a finite-time singularity, and it amplifies ONE ULP. Measured over four runs on
      # one machine: they agree bit-for-bit for 148 rows of minimum.txt (to t=8.6259), differ by
      # 2.44e-16 there, exceed 1e-6 relative one row later and 1e-3 by t=8.76, and finish with a
      # deepest neck radius spread across 0.000379..0.000400 - 5 %. The final state is worse still:
      # two runs remeshed differently, 299 dofs of mesh_y against 433. So neither the reduced series
      # nor the final fingerprint is reproducible on a single machine, and no tolerance would make
      # them mean anything.
      #
      # What IS reproducible is everything before the singularity, and exactly - the instants
      # included, which is why match="exact" works there. That is the whole check. There is
      # deliberately no Fingerprint: the state it would record is past the singularity.
      # ...and serially only. Over four ranks the adaptive stepper lands on different instants even
      # before the singularity - 10 of the 64 stored ones were missing, the worst 5.8e-07 away - so
      # the lookup match="exact" rests on does not hold there. Excused rather than switched to
      # match="interp", which would weaken the serial comparison everywhere to buy an MPI one.
      Evolution("minimum.txt", until_time=8.5, skip_under=("mpi",),
                reason="past t=8.63 the pinch-off singularity amplifies one ULP to 5 % within 80 "
                       "rows (measured over four runs), and across rank counts the adaptive "
                       "instants themselves shift by ~6e-7"),
  ],
  "Moving_Mesh/rayleigh_plateau_pinchoff.py": [
      # The two quantities that make this script the topology test it is: max of the fragment count
      # (how many drops the jet broke into) and the bounds on volume (which the surgery must
      # conserve). Both fall out of reducing the columns, and both are invariant under the adaptive
      # time grid - which matching rows would not be.
      #
      # stats=("min","max") although this script came out fully reproducible here (112 of 112
      # fingerprint keys and all 71 rows bit-identical over two runs): the row count of an adaptive
      # run is not something one machine can promise, and the mean, the l2 and the count would carry
      # that straight into a cross-platform failure. The extremes are what the check is about.
      FinalState("pinchoff.txt", stats=("min", "max"),
                 reason="the row count of an adaptive run is machine-dependent even where this one "
                        "reproduced exactly here"),
      # The final state is past the pinch-off, so it is rank-dependent: over four ranks 76 of the
      # fingerprint's entries moved, up to 0.44 % on mesh_x's l2. The reduced extremes above came
      # through that pass untouched, which is the whole argument for reducing a series rather than
      # pinning the state it ends in - max(fragments) and the bounds on volume are the same on one
      # rank and on four.
      Fingerprint(skip_under=("mpi",),
                  reason="the state after a topological surgery is rank-dependent; 76 entries "
                         "moved over four ranks, up to 0.44 %"),
  ],
  # The droplet-spreading family. Each varies one ingredient - the slip length, free slip, a
  # hyperelastic tangential shift, Marangoni plus gravity - on the same spreading drop, so what
  # distinguishes them is the contact-line position and the interface shape, i.e. the coordinate dof
  # groups the fingerprint separates out.
  "Moving_Mesh/droplet_spread_sliplength.py": [
      Fingerprint(),
  ],
  "Moving_Mesh/droplet_spread_free_slip.py": [
      Fingerprint(),
  ],
  "Moving_Mesh/droplet_spread_hyperelastic_tangential_shift.py": [
      Fingerprint(),
  ],
  "Moving_Mesh/droplet_spread_marangoni_and_gravity.py": [
      # Continued in its parameters, so contact_angle, gravity_factor and sigma_gradient are the
      # answer as much as the shape is - and all three come out bit-identical between runs.
      #
      # The pressure does not, and the measurement says why rather than how much: over two runs
      # EVERY pressure group shifted by exactly the same constant, -0.414154 on its min, its max and
      # its mean alike, and volume_constraint/volume_lagrange shifted by -0.414155. The pressure
      # level and that multiplier are one degree of freedom, and the solve lands anywhere along it -
      # a near-nullspace, not a drift. Everything else agrees to 1e-10 or better (the mesh
      # coordinates to 1e-15, the velocities to 1e-10), so skipping the gauge costs nothing and
      # pinning it would pin an arbitrary choice.
      #
      # Worth a look independently of this check: a volume constraint is supposed to DETERMINE that
      # level, so the pair being free suggests the constraint is degenerate here.
      Fingerprint(skip=["dofs.*pressure*", "dofs.volume_constraint/*"],
                  reason="the pressure level and the volume Lagrange multiplier are one gauge "
                         "freedom - measured as an identical -0.414154 shift of every pressure "
                         "group's min, max and mean"),
  ],
  "Moving_Mesh/droplet_spread_3d.py": [
      # The only 3d script in these two chapters, 36 dof groups.
      Fingerprint(),
  ],
}
