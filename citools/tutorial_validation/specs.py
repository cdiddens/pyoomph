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
      #
      # rtol=1e-4, and the number is measured rather than chosen. This script is NOT reproducible
      # run to run at the default 1e-5, and not because of MPI: against the committed reference a
      # fresh SERIAL run deviates by 7.6e-06 on h's l2, 8.7e-06 on lapl_h's and 1.11e-05 on
      # lapl_h's max - already outside 1e-5 - while three four-rank runs deviate LESS (2.7e-06 to
      # 6.0e-06) and differ among themselves by up to 9.9e-06. So the spread is the continuation's
      # own, at any rank count, and the committed reference is one draw from it. The default
      # tolerance sat exactly on that spread, which is the worst place for it: the same check failed
      # on h's l2 in one pass and passed in another.
      #
      # The fold location is still pinned to something meaningful at 1e-4: delta and gamma deviate
      # by 2.2e-06 and 1.3e-06 over the same four runs, and a genuinely different fold would move
      # them by far more than 1e-4. Keeping them at 1e-5 in a second check was considered and
      # dropped - it doubles the entry to buy one order on a quantity that is already four times
      # inside the looser bound.
      Fingerprint(skip=["dofs.(not described).min",
                        "dofs.(not described).max",
                        "dofs.(not described).mean"],
                  rtol=1e-4,
                  reason="a null eigenvector is defined only up to sign, and over four ranks the "
                         "eigensolver returned the other one (component sum ratio -0.999993); and "
                         "this script's own run-to-run spread is ~1e-05 at any rank count, measured "
                         "over four runs, so the default tolerance sat exactly on it"),
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
      # gauge_free on the pressure rather than skip, which is what this entry did when it was the
      # first script of this shape: the level is free but the pressure's spread is not, and the
      # spread is where a regression in the traction balance or the Marangoni stress would show.
      # Multiple_Domains/falling_droplet.py has the same structure and the same measurement.
      #
      # The volume Lagrange multiplier stays in skip= and deliberately: it is a SCALAR group, n=1,
      # so its max - min is identically zero and comparing that would be a check in appearance only.
      # Where a gauge-free group has one entry there is nothing invariant left to compare, and
      # saying so beats dressing it up.
      Fingerprint(gauge_free=["*pressure*"], skip=["dofs.volume_constraint/*"],
                  reason="the pressure level and the volume Lagrange multiplier are one gauge "
                         "freedom - measured as an identical -0.414154 shift of every pressure "
                         "group's min, max and mean; the pressure is compared by its spread, and "
                         "the multiplier is a scalar with no invariant part left"),
  ],
  "Moving_Mesh/droplet_spread_3d.py": [
      # The only 3d script in these two chapters, 36 dof groups.
      Fingerprint(),
  ],
  # ===============================================================================================
  # Advanced_Linear_Dynamics: the chapter where a validated answer is worth most, because it is the
  # one whose quantities are hardest to eyeball. Three real pyoomph defects were found here while
  # this validation was being built - a PETSc multi-solve returning aliases of one Vec, a
  # non-Jacobian matrix inheriting the Jacobian's symmetry proof, and the Lyapunov adjoint route
  # leaking its tracked parameter - and all three were invisible to a pass/fail run.
  #
  # Most of these scripts write their answer as a CURVE against a scanned parameter rather than
  # against time: a dispersion relation lambda(k), a frequency response, a critical curve Bo_c(V).
  # The abscissa then comes from a numpy.linspace or an explicit list and is exact, which is why
  # match="exact" is right here even though nothing in the chapter is a time series.
  #
  # The harness picks the PETSc build per script (see needs_complex_petsc in
  # citools/test_all_tutorial_scripts.py): azimuthal_stability= and additional_cartesian_mode=
  # route to the complex build, the rest to the real one. The reference data below was generated
  # under exactly that routing, so it must be regenerated if the routing changes - a script that
  # silently fell back to the scipy eigensolver would compute something else entirely.
  # ===============================================================================================

  "Advanced_Linear_Dynamics/turing_dispersion.py": [
      # The dispersion relation of a Turing system: two eigenvalues over k in linspace(0,1,400),
      # which is what tells you the system is Turing-unstable and at which wavenumber. The dominant
      # k read off this curve (0.46) is what turing_transient.py then builds its domain around, so
      # these two scripts are a pair and this one carries the quantitative half.
      #
      # 77 of the 400 rows, k=0..0.19, hold a COMPLEX-CONJUGATE PAIR: the real parts tie to 1e-15
      # and the imaginary parts are +-1.3067, so which of the two the solver returns first is
      # arbitrary and the sign of ImLambda1 with it. Two serial runs came out bit-identical (all
      # 2000 numbers), but four ranks did NOT: 14 of 256 compared values flipped sign and nothing
      # else moved at all - 1.30664760204 against -1.30664760204, the same magnitude to every
      # digit.
      #
      # So the two imaginary columns are compared by magnitude. |Im lambda| is the oscillation
      # frequency and is the physics; its sign says which conjugate came back first, which is not a
      # result. Nothing is lost by it either: a real eigenvalue has Im = 0, and the two members of
      # a conjugate pair have equal |Im|. The real parts stay signed, because their sign is the
      # stability and is the whole point of a dispersion relation.
      Evolution("dispersion.txt", abs_columns=("ImLambda1", "ImLambda2"),
                reason="which member of a complex-conjugate pair the eigensolver returns first is "
                       "arbitrary - measured as 14 of 256 values flipping sign, magnitudes "
                       "identical, between one rank and four"),
      Fingerprint(),
  ],
  "Advanced_Linear_Dynamics/rivulet.py": [
      # The growth rate of a rivulet against the axial wavenumber, at three contact angles - the
      # whole point of the script is how the curve changes between 60, 90 and 120 degrees, so all
      # three are pinned rather than only the last. 50 k values each, from an explicit scan.
      Evolution("for_60_deg_SL_0.01.txt"),
      Evolution("for_90_deg_SL_0.01.txt"),
      Evolution("for_120_deg_SL_0.01.txt"),
      # The proposer also offered a FinalState over each of these as "for_..._SL_*.txt". That glob
      # matches the one file the Evolution already compares row by row, so it would only restate a
      # weaker form of the same check.
      #
      # The fingerprint is serial-only, and for a structural reason rather than a numerical one:
      # this script builds its problem at module level ("problem=RivuletProblem()", no with-block),
      # so Problem.release() is never called and the only route left is the atexit fallback - which
      # refuses to collect under MPI on purpose, because a collective entered at interpreter
      # shutdown is a hang rather than a diagnostic (the ranks arrive at their own pace, and one may
      # already have finalised MPI). rivulet.py is the first script in the tutorial set to take that
      # path. The three dispersion curves above come from files and are checked under MPI as usual,
      # so what is lost here is only the state dump.
      Fingerprint(skip_under=("mpi",),
                  reason="the problem is never released (no with-block), so the fingerprint can "
                         "only come from the atexit fallback, which does not collect under MPI"),
  ],
  "Advanced_Linear_Dynamics/linear_response_drum.py": [
      # The frequency response of a driven axisymmetric drum, projected onto the first ten
      # Fourier-Bessel modes, over 1000 frequencies from linspace(1,1000,1000).
      #
      # The header is part of the check and not just decoration: each column is named
      # "mode_i[...](f=<resonance>)" with the analytic undamped resonance of that mode, computed
      # from the Bessel roots and c/R. A column name that changed would be reported as "the columns
      # changed" rather than as a numeric diff, which is the clearest failure message in the module
      # - and it pins the drum's wave speed and radius without a separate check. The resonances are
      # analytic, so they cannot drift with the solver; that is exactly why they make a good header.
      #
      # The response spans orders of magnitude between a resonance peak and the troughs between
      # them, which is what the per-column range rule is for: an off-resonance amplitude is judged
      # against the peak of its own column, not against itself.
      Evolution("response.txt"),
      Fingerprint(),
  ],
  "Advanced_Linear_Dynamics/linear_response_oscillator.py": [
      # The cleanest check in the chapter, and it takes 0.4 s: the frequency response of a damped
      # harmonic oscillator over omega in linspace(0.01,3,300), computed by PeriodicDrivingResponse
      # AND in closed form, written side by side as (A/F)_num and (A/F)_ana.
      #
      # Both columns are pinned, which is what makes a failure here immediately diagnosable: if the
      # numerical column moves while the analytic one does not, the response machinery drifted; if
      # both move, the scaling or the driving did. No other script in the tutorial carries its own
      # exact answer in the same file.
      #
      # And it means this reference is known to be RIGHT rather than merely reproducible, which is
      # otherwise the weak point of generated reference data: across the 64 stored rows the two
      # columns agree to 8.4e-16, machine precision, with a peak response of 9.710564 in both. Most
      # entries in this file pin what pyoomph computed; this one pins what the answer is.
      #
      # (The script's oscillator.txt is empty by design - the transient route to the same answer sits
      # behind an "if False:" in the tutorial text - so there is nothing to compare in it.)
      Evolution("response.txt"),
      Fingerprint(),
  ],
  "Advanced_Linear_Dynamics/rising_bubble.py": [
      # The m=1 instability of a rising bubble against the Bond number. Bo is stepped by a FIXED
      # increment (go_to_param(Bo=Bo+dBond)), so the abscissa is exact even though everything else
      # about this script is adaptive.
      #
      # It is adaptive in an unusual way: refine_eigenfunction() refines the mesh according to the
      # EIGENFUNCTION at every step, so the mesh the eigenvalue is computed on is itself chosen by
      # the eigenvalue. That looked like the entry most likely to need excusing under MPI, for the
      # reason moffatt_eddies and heated_cylinder do - a refinement criterion evaluated per
      # partition is a different criterion - and it turned out not to: all 39361 dofs and the whole
      # Bo curve came through four ranks untouched. So an eigenfunction-driven criterion is NOT
      # rank-dependent here, which is worth knowing precisely because the spatial-error-estimator
      # ones in SpatioTemporal_PDEs are.
      Evolution("m1_instability.txt"),
      Fingerprint(),
  ],
  "Advanced_Linear_Dynamics/eigenbranch_continuation.py": [
      # Seven branches of an eigenvalue through folds: two Bond numbers, two eigenvalue indices, and
      # the stable/unstable/fold variants, each a curve of (L, ReLambda, ImLambda). The script's
      # subject is an eigenvalue tracked along a branch rather than a solution, so what matters is
      # the range the eigenvalue covers on each branch - in particular whether it crosses zero,
      # which is where the branch changes stability.
      #
      # REDUCED rather than matched row by row, and not because the rows disagreed: every one of the
      # seven comes from arclength_continuation("L", dL, max_ds=dL0), so both the L values and the
      # row count (35, 29, 57, 29, 29, 30, 29 here) are chosen by the step adaptation. This project
      # already has the measurement that says such a grid is not invariant - rayleigh_plateau.py's
      # adaptive instants shift by ~6e-7 merely between rank counts on this one machine - so pinning
      # the grid would buy a check that fails for the wrong reason. stats=("min","max") for the same
      # reason it is used on the reduced series in Moving_Mesh: an adaptive row count takes the mean,
      # the l2 and the count down with it, while the extremes are the physics.
      # Worth recording what was NOT tested: hanging_droplet.py above turned out to keep its row
      # count and to move its abscissa only by round-off, which is what makes match="rows" work
      # there, and the same might well hold for these seven. It was not measured, because these
      # loops exit on overshooting a bound (L past maxL/minL) and a different step count would then
      # fail on the row count rather than on the physics. Reducing is the conservative choice here
      # and can be strengthened to match="rows" if the extremes ever prove too weak to catch
      # something.
      FinalState("curve_Bo_0_0_std.txt", stats=("min", "max"),
                 reason="an adaptive arclength grid, so only the extremes of each column are "
                        "invariant"),
      FinalState("curve_Bo_0_1_std.txt", stats=("min", "max"),
                 reason="an adaptive arclength grid"),
      FinalState("curve_Bo_0_0_unstab.txt", stats=("min", "max"),
                 reason="an adaptive arclength grid"),
      FinalState("curve_Bo_0.0025_0_fold.txt", stats=("min", "max"),
                 reason="an adaptive arclength grid"),
      FinalState("curve_Bo_0.0025_1_fold.txt", stats=("min", "max"),
                 reason="an adaptive arclength grid"),
      FinalState("curve_Bo_0.0025_0_unstab.txt", stats=("min", "max"),
                 reason="an adaptive arclength grid"),
      FinalState("curve_Bo_0.0025_1_unstab.txt", stats=("min", "max"),
                 reason="an adaptive arclength grid"),
      Fingerprint(),
  ],
  "Advanced_Linear_Dynamics/turing_transient.py": [
      # The transient counterpart of turing_dispersion.py: the domain is built around the dominant
      # wavenumber that script predicts (kc=0.46), and this one runs to t=2000 to let the pattern
      # grow out of a perturbed homogeneous state.
      #
      # The perturbation is numpy.random.rand(ndof)*0.001 with NO seed, so every run starts from a
      # different one, and the pattern it settles into differs with it. Measured over eight runs:
      # the pattern does NOT pick a unique amplitude - u's max spreads by 13 % of the field's
      # extent and the boundary lines by up to 64 %, because several patterns (different spot
      # counts and orientations) are all stable attractors here. That is multistability rather than
      # the chaos of SpatioTemporal_PDEs/kuramoto_sivanshinsky.py, but it has the same root cause
      # and the same consequence: no instantaneous field statistic is reproducible.
      #
      # The two BULK means are the exception, and they are worth keeping. Over the same eight runs
      # u's mean stayed within 0.46707..0.48202 and v's within 0.37682..0.38185, a spread of 1.2 %
      # and 1.0 % of each field's extent - a spatial average over 6241 nodes is far less sensitive
      # to which pattern was selected than any extreme is. And they measure something a mere ndof
      # check cannot: the homogeneous stationary state this script perturbs AWAY from is
      # u=0.651007, v=0.423810 (turing_dispersion.py's 2-dof fingerprint records exactly it), so a
      # regression in which no pattern formed at all would land u's mean 0.1785 away - twelve times
      # the observed spread.
      #
      # Hence rtol=0.04, which against the field-scale rule is 0.049 absolute: 3.3x the measured
      # spread, and still 3.6x inside the deviation a failed pattern would produce.
      Fingerprint(only=["ndof", "dofs.domain/u.mean", "dofs.domain/v.mean"], rtol=0.04,
                  reason="an unseeded numpy.random.rand perturbation, so the pattern differs every "
                         "run; only the bulk means are stable enough to compare (1.2 % over eight "
                         "runs against the 14.6 % a failure to form a pattern would give)"),
  ],
  "Advanced_Linear_Dynamics/rayleigh_benard_azimuthal_stability.py": [
      # The neutral-stability curve Ra(Gamma) of Rayleigh-Benard convection in a cylinder, one curve
      # per azimuthal mode m = 0, 1, 2, 3, each found by bifurcation tracking and then continued in
      # the aspect ratio from 0.5 to 3.0.
      #
      # These are anchored to physics and not just to a previous run, which is worth spelling out
      # because it is what makes the entry meaningful rather than circular. Ra falls as Gamma rises,
      # so max(Ra) on each curve is that mode's onset at Gamma=0.5, and the four come out
      #
      #     m=0  10896.4      m=1  3773.28      m=2  9144.6      m=3  20691.7
      #
      # i.e. m=1 is the lowest, at 3773 - exactly what the script's own comment states ("at
      # Gamma=0.5 the lowest onset is the one of m=1 at Ra=3773"). All four modes therefore confirm
      # a documented claim, not merely yesterday's number.
      #
      # min(Ra) is the other end, at Gamma=3.05: 1789.05, 1773.60, 1781.05, 1784.06. A cylinder of
      # growing aspect ratio must approach the classical laterally-unbounded onset Ra_c = 1707.76
      # from above, and these do - still 4 % above it at Gamma=3, which is the right side and the
      # right order of magnitude for this aspect ratio.
      #
      # All four are pinned, not just the last: comparing the modes against each other is the
      # script. The proposer offered a single FinalState over "curve_m_*.txt", which would have
      # taken one file of the four and silently dropped the rest.
      #
      # Reduced to the extremes for the same reason as eigenbranch_continuation above - the Gamma
      # grid comes from arclength_continuation with max_ds=0.05 and the loop exits at the first
      # Gamma past 3.0, so both the grid and the row count are the step adaptation's choice.
      FinalState("curve_m_0.txt", stats=("min", "max"),
                 reason="an adaptive arclength grid in Gamma, so only each column's extremes are "
                        "invariant; min(Ra) is the critical Rayleigh number of this mode"),
      FinalState("curve_m_1.txt", stats=("min", "max"), reason="an adaptive arclength grid"),
      FinalState("curve_m_2.txt", stats=("min", "max"), reason="an adaptive arclength grid"),
      FinalState("curve_m_3.txt", stats=("min", "max"), reason="an adaptive arclength grid"),
      # The fingerprint's parameters are m=3's final state, i.e. the first Gamma past 3.0, which the
      # step adaptation picks - so it is pinned here but is the first thing to relax if another
      # platform's continuation lands elsewhere.
      Fingerprint(),
  ],
  "Advanced_Linear_Dynamics/hanging_droplet.py": [
      # The critical Bond number against droplet volume: 14 points of the curve at which a hanging
      # droplet becomes unstable. That curve IS the script's result, and each of its points is a
      # fold found by bifurcation tracking, so the check covers the tracker as well as the physics.
      #
      # match="rows" because the abscissa is not a grid - V comes out of
      # arclength_continuation("V", dV, max_ds=0.1*V) with remeshing in between, so it is a
      # computed result like the ordinate. Measured, by the first attempt at match="exact" failing:
      # 11 of the 14 stored V values recur bit-for-bit in a second run and the other three move by
      # up to 3.95e-09, i.e. 7.5e-10 of V. That is round-off in an otherwise identical step
      # sequence - the row count is the same - so matching by index compares the whole curve while
      # matching by abscissa value cannot look it up at all.
      #
      # This is a third reason for match="rows", distinct from the two already in this file: not a
      # non-monotonic abscissa and not a repeated one, but an abscissa that is itself a result.
      #
      # The risk it carries is the row count: the loop runs "while V < 5" and exits on overshooting
      # that bound, so a platform whose round-off tips the last step the other way would produce 13
      # or 15 rows and fail here. That is a signal worth getting rather than smoothing over, and
      # the fallback if it ever fires is to reduce this the way eigenbranch_continuation below is
      # reduced.
      Evolution("critical_curve.txt", match="rows",
                reason="V is produced by adaptive arclength continuation, not prescribed, so the "
                       "stored abscissa drifts by ~4e-09 and cannot be used as a lookup key"),
      Fingerprint(),
  ],

  # ===============================================================================================
  # Discontinuous_Galerkin: four variants of the same Poisson problem, plus one DG advection. The
  # chapter exists to compare discretisations, and two of the scripts measure themselves against an
  # EXACT solution and print the L2 error - which is the strongest kind of quantity in the whole
  # tutorial set, because it is an error and not just a number: it does not depend on the mesh
  # ordering, it has a known limit, and a sign error or a lost stabilisation term moves it by orders
  # of magnitude rather than by a tolerance.
  #
  # Those errors are read out of stdout because that is the only place they exist (they come from an
  # IntegralObservables that the scripts evaluate and print rather than write). Note the tolerance
  # that forces: a value printed with %.4e carries ~8e-5 of relative quantisation and one with %.3e
  # ~8e-4, so the rtol admits one quantum of the printed representation. That is not a statement
  # about the solver's reproducibility, which is far better - it is the resolution of the print.
  # ===============================================================================================

  "Discontinuous_Galerkin/poisson_weak_dirichlet.py": [
      # Nitsche's weak Dirichlet condition. vtu only, so the fingerprint's per-dof-type extent is
      # the solution; the boundary condition being weak is exactly what a wrong penalty term would
      # show up in, as a field that no longer reaches its prescribed value at the edge.
      Fingerprint(),
  ],
  "Discontinuous_Galerkin/hybrid_poisson.py": [
      # A hybridized Poisson problem: u in the elements, lam on the skeleton. report() is called
      # THREE times - on the solution, after a uniform refinement but BEFORE re-solving, and after
      # re-solving - and one pattern picks up all three in order, so the twelve captured numbers
      # cover what the chapter is actually about: the refinement rebuilds the facet skeleton from
      # scratch, and the middle report measures how well the recovery expression filled the facets
      # that the refinement created. "unfilled facets" is an integer count and is compared exactly.
      # Four patterns over the same three lines rather than one with four groups, because the four
      # quantities need four different questions asked of them. Each pattern matches all three
      # reports, in order.
      #
      # The L2 error of u is the convergence quantity: 5.798e-03 on the solution, 6.320e-03 after
      # the refinement before re-solving, 1.453e-03 after - a factor of four, which is the O(h^2)
      # a uniform refinement should buy.
      Stdout(r"L2 error of u = ([-\d.eE+]+)", rtol=2e-3,
             reason="printed with %.3e, so one quantum of the printed representation is ~8e-4 "
                    "relative; this is the resolution of the print, not of the solver"),
      # The error of lam spans 1.3e-06 to 2.9e-02 across the three reports, so it carries the same
      # relative tolerance, with an atol that keeps the smallest of them from being judged against
      # its own round-off.
      Stdout(r"error of lam = ([-\d.eE+]+)", rtol=2e-3, atol=1e-8,
             reason="printed with %.3e; the atol covers the 1.3e-06 value of the third report"),
      # |jump(u)| is SUPPOSED to vanish - a hybridized solution is continuous across its facets by
      # construction - and it comes out at 1.4e-08, i.e. round-off. Judging that relatively would
      # make the check fail whenever round-off lands elsewhere, which is the near-zero trap this
      # module hits everywhere else (see the field-scale rule on the fingerprint and on FinalState;
      # Stdout has no scale to appeal to, so the scale is stated here instead). rtol=0 with
      # atol=1e-6 asks the only question worth asking: is the jump still zero?
      Stdout(r"\|jump\(u\)\| = ([-\d.eE+]+)", rtol=0.0, atol=1e-6,
             reason="a hybridized solution is continuous across facets by construction, so this is "
                    "round-off (1.4e-08); the check is that it stays below 1e-6, not that round-off "
                    "reproduces"),
      # An integer, and the one that says the facet recovery left nothing behind.
      Stdout(r"unfilled facets = (\d+)"),
      FinalState("domain_*.txt"),
      Fingerprint(),
  ],
  "Discontinuous_Galerkin/hdg_poisson.py": [
      # The hybridizable DG form, with static condensation. The L2 error is the answer.
      Stdout(r"L2 error of u\s+:\s+([-\d.eE+]+)",
             rtol=1e-4,
             reason="printed with %.4e, so one quantum of the printed representation is ~8e-5 "
                    "relative"),
      # The element and interior-facet counts are structural and exact, and they pin the skeleton
      # the HDG form is built on - a facet mesh that came out a different size would change the
      # answer without necessarily changing ndof.
      Stdout(r"elements\s+:\s+(\d+)"),
      Stdout(r"interior facets\s+:\s+(\d+)"),
      # Deliberately NOT the static-condensation line. It reports this process's share of the
      # blocks while the dof count is the whole problem's, and the script says so in its own
      # comment by appending "on this process" under MPI - so it is a per-rank quantity and belongs
      # in no reference file.
      Fingerprint(),
  ],
  "Discontinuous_Galerkin/convection_diffusion.py": [
      # DG advection-diffusion run to t=50. The last profile is the one worth pinning: a DG scheme
      # that lost its upwinding smears or oscillates, and either shows up in min/max of c.
      FinalState("domain_*.txt"),
      Fingerprint(),
  ],
  # ===============================================================================================
  # Multicomponent_Flow. Half of this chapter does not solve anything: it DEFINES materials and
  # prints their properties, with no Problem, no mesh and no output file. The fingerprint has
  # nothing to collect there (the proposer says so rather than inventing one), so those scripts are
  # checked through stdout, which is the one place their numbers exist.
  #
  # That turns out to be the strongest kind of check in the chapter, because a material property is
  # a formula with a known answer. The pure-gas densities below are the ideal gas law, and they
  # reproduce p*M/(R*T) to 0 or 1.8e-16 at every temperature printed - so what these entries pin is
  # arithmetic that can be checked by hand, not merely yesterday's output. They also cover a
  # surprising amount of machinery on the way: the registry and its override=True, the unit algebra,
  # evaluate_at_condition, and the conversion from a symbolic expression to a float.
  #
  # The printed values are plain repr() floats, so unlike the L2 errors in Discontinuous_Galerkin
  # the print costs no precision and the default rtol applies.
  # ===============================================================================================

  "Multicomponent_Flow/materials_pure_gas.py": [
      # The density at five temperatures, with the temperature captured alongside each value so a
      # reordered loop cannot pass. Verified against p*M/(R*T) with M=28.9645 g/mol and pyoomph's
      # own gas_constant (8.314462618153239504): exact to the last bit at 10, 25 and 30 C and 1.8e-16
      # at 15 and 20 C.
      Stdout(r"DENSITY AT T\[C\]= (\d+) is ([-\d.eE+]+) kg/m\^3"),
      # The same density at 18 C, as a float. Also printed with units just above, which is where the
      # symbolic route rather than the float one would show a problem.
      Stdout(r"EVALUATED DENSITY in \(kg/m\*\*3\) ([-\d.eE+]+)"),
      # Two properties of the DEFINITION rather than of an evaluation, both read out of the leading
      # coefficient of a printed expression:
      #   - the viscosity constant, 1.813e-05 Pa s, from the first (constant) definition of air;
      #   - rho(p,T)'s coefficient, M/R = 0.003483628627635038758, which pins the molar mass and the
      #     gas constant together.
      # Only the leading number is captured, never the expression around it: GiNaC's printed form
      # (the operand order, "field(temperature,< code=0 , tags=>)") is not a contract and would make
      # a brittle check.
      Stdout(r"Dynamic viscosity: \(([-\d.eE+]+)\)"),
      Stdout(r"VARIABLE DENSITY \(([-\d.eE+]+)\)"),
      # This one checks behaviour, not a number: the script sets air.mass_density by hand to
      # 2 kg/m^3 and then loads a SECOND instance, which must still have the registered 1.225. If
      # get_pure_gas ever started handing out a shared object, this is the line that would catch it.
      Stdout(r"Mass densities \S+ \(([-\d.eE+]+)\)"),
  ],
  "Multicomponent_Flow/temperature_and_pressure_dependency.py": [
      # The same ideal-gas material as materials_pure_gas.py, reached by the route the tutorial text
      # takes here, so the same three numbers are pinned and for the same reason.
      Stdout(r"DENSITY AT T\[C\]= (\d+) is ([-\d.eE+]+) kg/m\^3"),
      Stdout(r"EVALUATED DENSITY in \(kg/m\*\*3\) ([-\d.eE+]+)"),
      Stdout(r"VARIABLE DENSITY \(([-\d.eE+]+)\)"),
  ],
  "Multicomponent_Flow/materials_gas_mixture.py": [
      # A binary gas mixture, water vapour in air at 2 % by mass. Two things are printed and both
      # are worth pinning: the mixture's initial condition (the mass fractions it was built with)
      # and its density at 20 C and 1 atm.
      #
      # On the density, note what is and is not claimed. The mass-fraction and mole-fraction mixing
      # rules are algebraically the same, 1/M_mix = sum(w_i/M_i), and with the script's own molar
      # masses (28.9645 and 18.01528 g/mol) and pyoomph's gas constant that gives
      # 1.1896284268010473 against the printed 1.1896288300208942 - agreeing to 3.4e-7, which
      # confirms the physics but is not the exact match the pure-gas case gives. The residue is
      # three orders inside this check's tolerance, so it is pinned as computed; it is noted because
      # a reader comparing this entry with the pure-gas one above would otherwise wonder why one is
      # exact and the other is not.
      Stdout(r"'massfrac_water': ([-\d.eE+]+), 'massfrac_air': ([-\d.eE+]+)"),
      # (?m) because the module matches with re.finditer and no MULTILINE flag, so a bare ^ would
      # anchor to the start of the whole output and quietly match nothing at all.
      Stdout(r"(?m)^\(([-\d.eE+]+)\)\*meter\*\*\(-3\)\*kilogram"),
  ],
  "Multicomponent_Flow/insoluble_surfactant_definition.py": [
      # Another script with no Problem: it defines a surfactant and prints its equation of state.
      # Two of the three printed lines are symbolic expressions in temperature and surface
      # concentration, and those are deliberately NOT matched - GiNaC's printed form (operand order,
      # "field(temperature,< code=0 , tags=>)", the subexpression() wrapper) is not a contract.
      # The third line is the state EVALUATED at a condition, 0.06999979028348837657 N/m, which is
      # a number and is the one thing here worth pinning: it exercises the whole chain from the
      # registered definition through the unit algebra to a float.
      Stdout(r"(?m)^\(([-\d.eE+]+)\)\*second\*\*\(-2\)\*kilogram$"),
  ],
  "Multicomponent_Flow/soluble_surfactants.py": [
      # The soluble counterpart, same shape and same reasoning: the evaluated surface tension,
      # 0.06971774159219308997 N/m. Worth having both, because the pair is the comparison the
      # tutorial is making - the soluble isotherm gives a slightly lower tension at the same
      # condition, and a regression that collapsed one onto the other would show up here.
      Stdout(r"(?m)^\(([-\d.eE+]+)\)\*second\*\*\(-2\)\*kilogram$"),
  ],
  "Multicomponent_Flow/marangoni_instability.py": [
      # Solutal Marangoni instability. The initial perturbation is a DeterministicRandomField with a
      # seed= passed, which is what makes this script checkable at all: it is the one random initial
      # condition in the tutorial set that is reproducible, in contrast to
      # SpatioTemporal_PDEs/kuramoto_sivanshinsky.py and Advanced_Linear_Dynamics/turing_transient.py
      # (see the README section on unseeded initial conditions). vtu only, so the fingerprint's
      # per-dof-type extent is the whole check.
      Fingerprint(),
  ],
  "Multicomponent_Flow/rayleigh_taylor_instability.py": [
      # The multicomponent version of the SpatioTemporal_PDEs script of the same name - a different
      # script, hence a separate entry; both are in the set.
      Fingerprint(),
  ],
  "Multicomponent_Flow/gcl_glycerol_water_capillary.py": [
      # Evaporation of a glycerol-water mixture from a capillary, 48 h of it.
      #
      # The two observable series are REDUCED rather than matched row by row, and the run() call is
      # the reason: outstep=True with temporal_error=1 writes a row per adaptive step, so both the
      # instants and the row count (80 here) belong to the machine. The extremes are the physics -
      # how much glycerol is left, how far the interface travelled - and they are invariant under
      # the grid. stats=("min","max") for the usual reason: the mean, the l2 and the count all ride
      # on the row count.
      FinalState("mass_evolution.txt", stats=("min", "max"),
                 reason="outstep=True with temporal_error=1, so the row grid is adaptive"),
      FinalState("top_interface.txt", stats=("min", "max"),
                 reason="outstep=True with temporal_error=1, so the row grid is adaptive"),
      # The final nodal profile, on the other hand, is at a prescribed end time and is compared in
      # full (reduced per column, as every nodal output is).
      FinalState("domain_*.txt"),
      Fingerprint(),
  ],
  "Multicomponent_Flow/nacl_capillary_evaporation.py": [
      # The same capillary with salt, solved twice: once with the full compositional model and once
      # with the dilute approximation. Comparing the two IS the script, so both are pinned.
      #
      # Reduced for the same reason as gcl_glycerol_water_capillary.py above - outstep=True with
      # temporal_error=1 - and the reduction is where a real physical check appears: N_salt is a
      # CONSERVED quantity, salt does not evaporate, so min and max of that column have to agree
      # with each other as well as with the reference. A leak in the compositional transport shows
      # up as the two drifting apart, which no single stored value would reveal.
      FinalState("nacl_capillary_component/bulk_evolution.txt", stats=("min", "max"),
                 reason="an adaptive row grid; and min/max of N_salt must agree, since salt is "
                        "conserved"),
      FinalState("nacl_capillary_component/evaporating_end.txt", stats=("min", "max"),
                 reason="an adaptive row grid"),
      FinalState("nacl_capillary_component/top_interface.txt", stats=("min", "max"),
                 reason="an adaptive row grid"),
      FinalState("nacl_capillary_component/domain/domain_*.txt"),
      FinalState("nacl_capillary_dilute/bulk_evolution.txt", stats=("min", "max"),
                 reason="an adaptive row grid; and min/max of N_salt must agree, since salt is "
                        "conserved"),
      FinalState("nacl_capillary_dilute/evaporating_end.txt", stats=("min", "max"),
                 reason="an adaptive row grid"),
      FinalState("nacl_capillary_dilute/top_interface.txt", stats=("min", "max"),
                 reason="an adaptive row grid"),
      FinalState("nacl_capillary_dilute/domain/domain_*.txt"),
      Fingerprint(),
      # Problem 1 serially only. Measured, and the cause is the time stepper rather than the
      # physics: at four ranks this problem ends at t=4.811532879254676 against 4.811532688712693
      # serially - 4.0e-08 relative - and the evaporative flux at that instant moves by 0.43 %
      # (4.1490782e-07 to 4.1313897e-07, with the interface velocity following it exactly, as mass
      # conservation requires). An amplification of about 1e5, because the capillary is drying out
      # and the rate is changing steeply just where the run stops.
      #
      # Problem 0 above ends at a bit-identical instant at four ranks and needs no such excuse, so
      # this is one stepper decision tipping, not a property of the script. The three reduced series
      # came through four ranks untouched, N_salt conservation included - which is the argument for
      # reducing a series rather than pinning the state it ends in, made here for the third time
      # (see Moving_Mesh/rayleigh_plateau_pinchoff.py and the README).
      Fingerprint(index=1, skip_under=("mpi",),
                  reason="the adaptive stepper ends 4.0e-08 later at four ranks and the evaporative "
                         "flux is changing steeply there, so the final state moves by 0.43 %"),
  ],
  "Multicomponent_Flow/double_layer_relaxation.py": [
      # Eight problems in one script: a polarized double layer, a sweep over bulk concentration
      # (250, 1000, 4000, 16000 nM) and a sweep over the transfer coefficient (0.1, 0.5, 2). The
      # sweeps are the script - the point is how zeta and the adsorbed amount change along them - so
      # every one of the eight is pinned rather than only the last.
      #
      # Unlike the two evaporation scripts above, these series ARE matched row by row:
      # run(12*tD, outstep=0.1*tD) prescribes the instants, so the 121 rows belong to the script and
      # not to the time stepper, and match="exact" is the strong comparison that deserves.
      #
      # Per problem: zeta and the adsorbed amount over time (interface.txt), the surface charge and
      # the dissolved amount (bulk.txt), and the final profiles on both sides of the interface - the
      # Debye layer in the liquid (c_anion, c_cation, phi, charge_density, field, ionic strength)
      # and the field in the gas, which is where the continuity of the electric field across the
      # interface would break first.
      #
      # 40 checks and 108 KB make this the largest entry in the set by a factor of three, so for the
      # record it is a considered size rather than an oversight: the 16 time series are 70 % of it,
      # which is 8 studies times 2 observables, and the whole file averages ~1 KB per check, in line
      # with every other entry. Eight separate scripts would have carried the same data in eight
      # unremarkable files. If it ever does need trimming, bulk.txt is the one to reduce to its
      # extremes - zeta on the interface is the headline quantity, the surface charge follows it.
      Evolution("dl_polarized/interface.txt"),
      Evolution("dl_polarized/bulk.txt"),
      FinalState("dl_polarized/liq/liq_*.txt"),
      FinalState("dl_polarized/gas/gas_*.txt"),
      Evolution("dl_sweep_250nM/interface.txt"),
      Evolution("dl_sweep_250nM/bulk.txt"),
      FinalState("dl_sweep_250nM/liq/liq_*.txt"),
      FinalState("dl_sweep_250nM/gas/gas_*.txt"),
      Evolution("dl_sweep_1000nM/interface.txt"),
      Evolution("dl_sweep_1000nM/bulk.txt"),
      FinalState("dl_sweep_1000nM/liq/liq_*.txt"),
      FinalState("dl_sweep_1000nM/gas/gas_*.txt"),
      Evolution("dl_sweep_4000nM/interface.txt"),
      Evolution("dl_sweep_4000nM/bulk.txt"),
      FinalState("dl_sweep_4000nM/liq/liq_*.txt"),
      FinalState("dl_sweep_4000nM/gas/gas_*.txt"),
      Evolution("dl_sweep_16000nM/interface.txt"),
      Evolution("dl_sweep_16000nM/bulk.txt"),
      FinalState("dl_sweep_16000nM/liq/liq_*.txt"),
      FinalState("dl_sweep_16000nM/gas/gas_*.txt"),
      Evolution("dl_transfer_0.1/interface.txt"),
      Evolution("dl_transfer_0.1/bulk.txt"),
      FinalState("dl_transfer_0.1/liq/liq_*.txt"),
      FinalState("dl_transfer_0.1/gas/gas_*.txt"),
      Evolution("dl_transfer_0.5/interface.txt"),
      Evolution("dl_transfer_0.5/bulk.txt"),
      FinalState("dl_transfer_0.5/liq/liq_*.txt"),
      FinalState("dl_transfer_0.5/gas/gas_*.txt"),
      Evolution("dl_transfer_2/interface.txt"),
      Evolution("dl_transfer_2/bulk.txt"),
      FinalState("dl_transfer_2/liq/liq_*.txt"),
      FinalState("dl_transfer_2/gas/gas_*.txt"),
      # One fingerprint per problem. The last three carry one dof type more than the first five
      # (12 against 11), which is the transfer studies adding the dissolved-species field - so the
      # count itself distinguishes the two halves of the script.
      Fingerprint(),
      Fingerprint(index=1),
      Fingerprint(index=2),
      Fingerprint(index=3),
      Fingerprint(index=4),
      Fingerprint(index=5),
      Fingerprint(index=6),
      Fingerprint(index=7),
  ],
  # Multicomponent_Flow/materials_liquids.py has NO entry, and deliberately: it registers liquid
  # materials and prints nothing, builds no Problem and writes no file, so it exits 0 with no number
  # anywhere to compare. The harness reports it in the NOT VALIDATED section, which is the honest
  # outcome - what it tests is that the definitions import and evaluate without raising, and that is
  # exactly what its exit status already says.
  # ===============================================================================================
  # Multiple_Domains: problems split across several meshes that talk to each other across an
  # interface. What is worth checking here is almost always the COUPLING - a temperature that stays
  # continuous across a wall, a drag that a surfactant-laden interface changes, a phase boundary
  # that moves at the rate the latent heat allows - and a broken coupling is exactly the kind of
  # defect that leaves a script exiting 0.
  #
  # Three of the nine are over 40000 dofs (melting_ice_convection 60794,
  # falling_droplet_with_surfactants 47468, falling_droplet 47339), so the MPI pass over this
  # chapter is split and those three run on their own.
  # ===============================================================================================

  "Multiple_Domains/temperature_conduction.py": [
      # The simplest coupling in the tutorial and the fastest script in the chapter at 0.15 s: heat
      # conduction through two domains joined at an interface, solved stationary. Both profiles are
      # pinned, which is what makes the check about the coupling rather than about one side of it -
      # the temperature has to be continuous there, so a broken interface condition moves one
      # profile relative to the other and cannot move both consistently.
      FinalState("domainA_*.txt"),
      FinalState("domainB_*.txt"),
      Fingerprint(),
  ],
  "Multiple_Domains/temperature_conduction_propagation.py": [
      # The transient version, with a moving ice/liquid phase boundary, run to 1000 s.
      #
      # Both nodal profiles are compared at the prescribed end time. Note the one thing to relax
      # first if another platform disagrees: this script uses spatial_adapt=1, so the node counts
      # (83 in the ice, 163 in the liquid here) are the error estimator's choice, and the reduced
      # n per column carries them. The same caveat as the adaptive scripts in Spatial_PDEs.
      FinalState("ice_*.txt"),
      FinalState("liquid_*.txt"),
      Fingerprint(),
  ],
  "Multiple_Domains/falling_droplet.py": [
      # A droplet falling under gravity, with the terminal velocity written to globals.txt. The
      # instants are prescribed - run(0.5*second, startstep=0.05*second, outstep=True) with no
      # temporal_error, so eleven rows at fixed steps - which is why match="exact" holds here while
      # the evaporation scripts in Multicomponent_Flow have to be reduced.
      #
      # UStokes is the quantity the script exists to produce, and it is the one to compare against
      # falling_droplet_with_surfactants.py below: a clean interface follows Hadamard-Rybczynski,
      # a surfactant-laden one is retarded towards the rigid-sphere Stokes limit, so the two
      # references should differ in a direction the physics dictates.
      Evolution("globals.txt"),
      # The pressure LEVEL is free here: the droplet and the surrounding fluid are both
      # incompressible and nothing pins a pressure anywhere, so the solve lands wherever it lands.
      # Measured between two runs on this machine: every pressure group's min, max and mean moved
      # by the same -6.3145163, the kinematic boundary condition's Lagrange multiplier (which
      # carries pressure units, and so rides the same constant) by -6.3145153, and each group's
      # max - min was preserved exactly - 8.111 before and after.
      #
      # gauge_free rather than skip: the level is not a result, but the pressure's SHAPE is, and
      # that is what a regression in the traction balance would change. Each of these groups is
      # therefore compared by its spread, which is invariant under the gauge.
      Fingerprint(gauge_free=["*pressure*", "*_kin_bc*"],
                  reason="nothing pins the pressure level, so it drifts between runs by a constant "
                         "(-6.3145163, identical on every pressure group and on the kinematic-BC "
                         "multiplier); the spread of each group is what carries the physics"),
  ],
  "Multiple_Domains/falling_droplet_with_surfactants.py": [
      # The same droplet with an insoluble surfactant on its interface. One more dof type than the
      # clean case (29 against 28), which is the surface concentration field, and a terminal
      # velocity that the Marangoni stresses reduce.
      Evolution("globals.txt"),
      # Same free pressure level as the clean case, measured the same way: a common shift of
      # -0.3538285 across every pressure group and the kinematic-BC multiplier, with each group's
      # spread unchanged. The constant differs from the clean case because the scales do; the
      # structure is identical.
      Fingerprint(gauge_free=["*pressure*", "*_kin_bc*"],
                  reason="nothing pins the pressure level, so it drifts between runs by a constant "
                         "(-0.3538285 here); the spread of each group carries the physics"),
  ],
  "Multiple_Domains/evaporating_water_droplet.py": [
      # An evaporating droplet coupled to the vapour in the surrounding gas, 35 dof types - the most
      # of any script in this chapter.
      #
      # EVO_droplet.txt holds the droplet volume against time, and it is REDUCED rather than matched:
      # run(100*second, startstep=10*second, outstep=True, temporal_error=1) writes a row per
      # adaptive step, which is why there are four of them rather than a round number. Reducing to
      # min and max is also the right question to ask of this particular series - the extremes are
      # the initial and the final volume, i.e. how much evaporated, which is the result. Matching
      # four adaptive rows would pin the time stepper instead.
      FinalState("EVO_droplet.txt", stats=("min", "max"),
                 reason="outstep=True with temporal_error=1, so the four rows are adaptive; min and "
                        "max are the initial and final volume, which is what the script measures"),
      Fingerprint(),
  ],
  "Multiple_Domains/two_layer_flow.py": [
      # Two immiscible layers with a free interface between them, as two coupled domains.
      Fingerprint(),
  ],
  "Multiple_Domains/two_layer_flow_single_domain.py": [
      # The same physics in ONE domain, with the interface captured instead of meshed. Comparing the
      # two formulations is the point of the pair, and both are pinned - but note that the framework
      # cannot compare them against each other: the dof types differ (13 here against 34 in the
      # two-domain version), so each reference pins its own formulation and the agreement between
      # them stays a matter for a reader, not for the check.
      Fingerprint(),
  ],
  "Multiple_Domains/melting_ice_convection.py": [
      # Ice melting into convecting water: the largest script in the chapter at 60794 dofs, and the
      # one whose answer depends on the latent-heat coupling at the moving phase boundary.
      Fingerprint(),
  ],
  "Multiple_Domains/simple_fsi.py": [
      # Fluid-structure interaction: a solid deforming under the flow it is immersed in, 26 dof
      # types across the two domains. The script has no "if __name__" guard but does use a
      # with-block, so Problem.release() is reached and the fingerprint is collected normally -
      # unlike Advanced_Linear_Dynamics/rivulet.py, which has neither.
      Fingerprint(),
  ],
  # ===============================================================================================
  # PreCICE_Coupling. Be clear about what these two entries do and do not cover: each script
  # branches on precice_participant, and with the default empty value it solves the FULL domain
  # monolithically rather than coupling anything. Coupling needs two participants alive at once
  # against a shared precice-config.xml, which the harness cannot launch - it runs one process per
  # script - so the Dirichlet/Neumann coupling, which is what the chapter is about, is NOT exercised
  # here and a green check must not be read as saying it is.
  #
  # What IS covered is worth having all the same, because the monolithic branch is a manufactured
  # solution: the source term and every Dirichlet value come from an analytic u, and the script
  # integrates (u - u_analytical)**2 into domain_IntObsv.txt. So these entries pin the
  # discretisation error of a problem with a known exact answer, at the prescribed instants of
  # run(1, outstep=0.1) - and that error is precisely the quantity a coupled run would have to
  # reproduce, which makes it the right reference for the coupling even though it does not test it.
  # ===============================================================================================

  "PreCICE_Coupling/partitioned_heat_conduction.py": [
      # The squared error against the analytic solution at eleven prescribed instants, on the
      # 22x11 full domain (903 dofs, u only).
      Evolution("domain_IntObsv.txt"),
      Fingerprint(),
  ],
  "PreCICE_Coupling/partitioned_heat_conduction_circle.py": [
      # The same manufactured solution on the circular geometry, 1657 dofs.
      Evolution("domain_IntObsv.txt"),
      Fingerprint(),
  ],

  # ===============================================================================================
  # Plotting_Interface. These scripts exist to demonstrate the plotting interface, so most of them
  # re-use a problem from an earlier chapter and differ only in what they draw. The plots themselves
  # are not checked - a PNG is not a number, and the harness's own pass/fail already says the
  # plotting ran - but the underlying problem is the same physics and is pinned the same way, with
  # the reasoning cross-referenced rather than repeated.
  #
  # Three of them are near-duplicates of entries elsewhere: rising_bubble.py of the
  # Advanced_Linear_Dynamics script of that name, evaporating_water_droplet.py of the
  # Multiple_Domains one (identical ndof, 29022), and kuramoto_sivanshinsky.py of the
  # SpatioTemporal_PDEs one. They are separate scripts in the bundle and get separate references;
  # what that buys is a check that the plotting interface does not disturb the solve.
  # ===============================================================================================

  "Plotting_Interface/one_dimensional.py": [
      # Two problems into two output directories, which is why this script is named in the harness
      # note about scripts that set their own: 121 dofs for the first, 39 for the second. vtu and
      # plots only, so the fingerprints are the whole check.
      Fingerprint(),
      Fingerprint(index=1),
  ],
  "Plotting_Interface/tracers.py": [
      # Tracer particles advected through a flow. The seed= here is a TracerSeedGrid, i.e. a
      # deterministic grid of starting positions and not a random seed - worth saying, because every
      # other "seed" in this file is about reproducible randomness.
      Fingerprint(),
  ],
  "Plotting_Interface/eigendynamics.py": [
      # An eigenmode-driven transient, with Bo and the azimuthal mode as parameters and the critical
      # eigenvalue recorded. One Problem with three run() calls, hence one fingerprint and three
      # elapsed times in the log - not three problems.
      Fingerprint(),
  ],
  "Plotting_Interface/evaporating_water_droplet.py": [
      # The Multiple_Domains droplet again, same 29022 dofs. Reduced for the reason given there:
      # outstep=True with temporal_error=1, so the four rows are the stepper's choice, and min/max
      # of the volume are the initial and final values, which is what the series is about.
      FinalState("EVO_droplet.txt", stats=("min", "max"),
                 reason="an adaptive row grid; min and max of the volume are the initial and final "
                        "values - see Multiple_Domains/evaporating_water_droplet.py"),
      Fingerprint(),
  ],
  "Plotting_Interface/plotting_evaporating_droplet.py": [
      # The same droplet with only two output rows, reduced for the same reason. With two rows the
      # extremes ARE the series, so nothing is given up by reducing it.
      FinalState("EVO_droplet.txt", stats=("min", "max"),
                 reason="an adaptive row grid; with two rows the extremes are the whole series"),
      Fingerprint(),
  ],
  "Plotting_Interface/kuramoto_sivanshinsky.py": [
      # Physically identical to SpatioTemporal_PDEs/kuramoto_sivanshinsky.py - same L=50, same
      # run(2000, outstep=True, startstep=0.1, temporal_error=1, maxstep=50), same 12800 dofs, and
      # the same DeterministicRandomField with no seed= - so it starts somewhere different on every
      # run and the measurement made there carries over without repeating it: over eight runs h's
      # l2 spreads by 23 % of itself and the boundary lines by up to 64 %.
      #
      # ndof alone, therefore, and it is meant to read as the weak check it is.
      Fingerprint(only=["ndof"],
                  reason="an unseeded DeterministicRandomField initial condition, so every run "
                         "starts from a different state - measured on the SpatioTemporal_PDEs twin"),
  ],
  "Plotting_Interface/kuramoto_sivanshinsky_bifurcation.py": [
      # The fold of the hexagonal state again, written without a header here (hence column_0,
      # column_1, column_2 as positional names) and with 29 rows against the 58 of the
      # SpatioTemporal_PDEs twin.
      Evolution("hexfold.txt", match="rows",
                reason="a continuation, so the first column does not grow"),
      # Ends in the augmented fold-tracking state, so the dof vector carries the null eigenvector:
      # the "(not described)" group is 7614 of its components plus the one continuation unknown,
      # which is why that group's max is exactly gamma (0.2738001088). ndof = 15229 = 7614 + 7614 + 1.
      #
      # A null eigenvector is defined only up to sign - that is mathematics, not a measurement - and
      # the SpatioTemporal_PDEs twin supplies the measurement that it really does flip: at four
      # ranks the sum of its components came back as -0.999993 times the serial one, magnitudes
      # unchanged. So min, max and mean of that group say which sign the eigensolver happened to
      # return. l2 and n are kept, being invariant under a flip and structural, and everything that
      # is the answer stays compared: delta and gamma at the fold, the eigenvalue, ndof, and both
      # physical fields.
      # rtol=1e-4 for the same measured reason as the twin: the family's field statistics move by
      # ~1e-05 between runs at any rank count. This particular script happened to PASS the four-rank
      # pass at the default, which is precisely the problem - a check that lands on both sides of
      # its tolerance depending on the run is one that will flake in a nightly weeks from now.
      Fingerprint(skip=["dofs.(not described).min",
                        "dofs.(not described).max",
                        "dofs.(not described).mean"],
                  rtol=1e-4,
                  reason="a null eigenvector is defined only up to sign; and this family's fields "
                         "have a ~1e-05 run-to-run spread at any rank count"),
  ],
  "Plotting_Interface/kuramoto_sivanshinsky_arclength_eigen.py": [
      # gamma, h_rms and the critical eigenvalue along the branch, 36 rows, headerless.
      #
      # Serially only, and this needs no separate investigation: at four ranks exactly one of the
      # 144 compared values moves, the critical eigenvalue at column_0=0.2457868, from
      # 0.00427606969927 to 0.00370144097136 - the same two numbers to eleven digits that the
      # SpatioTemporal_PDEs twin produced, where it was measured as 1.85 % of that column's whole
      # range (-0.0311..0.0164). It is the accuracy of a near-zero eigenvalue of a 7614-dof system
      # across rank counts, not a drift, and no tolerance covers that and still leaves a check. So
      # the strong serial comparison is kept and the MPI pass leaves this one to the fingerprint,
      # which does hold there.
      Evolution("hexdots.txt", match="rows", skip_under=("mpi",),
                reason="a continuation, so the first column does not grow; and the critical "
                       "eigenvalue moves by ~2 % of its range across rank counts, identically to "
                       "the SpatioTemporal_PDEs twin"),
      Fingerprint(),
  ],
  "Plotting_Interface/plotting_eigenmodes.py": [
      # The fold with its eigenmodes drawn, on a finer mesh than the other two: 59893 dofs, the only
      # script in this chapter over the 40000-dof MPI guidance, so it runs on its own in that pass.
      Evolution("hexfold.txt", match="rows",
                reason="a continuation, so the first column does not grow"),
      # The same augmented state and the same unnamed null-eigenvector group as
      # kuramoto_sivanshinsky_bifurcation.py above, at this mesh: n = 29947 = 29946 components plus
      # the continuation unknown, max exactly gamma (0.2789130702), ndof = 59893 = 29946 + 29946 + 1.
      #
      # And the same rtol, for the same measured reason: this family's field statistics carry a
      # run-to-run spread of ~1e-05 whatever the rank count (measured on the SpatioTemporal_PDEs
      # script). Here it showed as lapl_h's l2 at 1.02e-05 over four ranks, i.e. the default
      # tolerance missing by two percent of itself.
      Fingerprint(skip=["dofs.(not described).min",
                        "dofs.(not described).max",
                        "dofs.(not described).mean"],
                  rtol=1e-4,
                  reason="a null eigenvector is defined only up to sign; and this family's fields "
                         "have a ~1e-05 run-to-run spread at any rank count, measured over four "
                         "runs of the SpatioTemporal_PDEs script"),
  ],
  "Plotting_Interface/rising_bubble.py": [
      # The Advanced_Linear_Dynamics rising bubble again, same 39361 dofs and the same m=1
      # instability curve against a Bond number stepped by a fixed increment, so the abscissa is
      # exact. See that entry for why an eigenfunction-driven refinement turned out NOT to be
      # rank-dependent.
      Evolution("m1_instability.txt"),
      Fingerprint(),
  ],
}
