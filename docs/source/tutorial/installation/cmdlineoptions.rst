.. _installcmdlineoptions:

Command line options
----------------------

Every pyoomph driver script is a plain Python script that creates a :py:class:`~pyoomph.generic.problem.Problem` (sub)class and eventually calls e.g. :py:meth:`~pyoomph.generic.problem.Problem.run`, :py:meth:`~pyoomph.generic.problem.Problem.solve` or :py:meth:`~pyoomph.generic.problem.Problem.output`. Before doing anything else, these methods trigger :py:meth:`~pyoomph.generic.problem.Problem.initialize`, which parses ``sys.argv`` for a set of command line options that are always available, for *any* problem, without any extra code required in your driver script. This page lists all of them; a few are also discussed in more detail elsewhere in the tutorial, which is cross-referenced below.

You can always get the full list (including any additional options a particular script might add on top, see :numref:`installcmdlinecustom` below) by calling your script with ``--help``, e.g.

.. code:: bash

      python my_simulation.py --help

Solver backend selection
~~~~~~~~~~~~~~~~~~~~~~~~~

The linear solver flags are mutually exclusive (passing e.g. both ``--pardiso`` and ``--petsc`` at once is rejected by argparse with a usage error). If none is given, pyoomph falls back to its default linear solver.

``--petsc``
      Use `PETSc <https://petsc.org>`__ as linear solver, see :numref:`petscslepc`.

``--superlu``
      Use the serial SuperLU solver.

``--umfpack``
      Use the UMFPACK solver.

``--pardiso``
      Use the (MKL) Pardiso solver.

.. note::

   ``--superlu``, ``--umfpack``, ``--pardiso`` and ``--accelerate`` are not MPI-parallel. They can
   still be used under ``mpirun``: the assembled system is gathered onto rank 0 and solved there while
   the other processes wait. The assembly stays parallel, so this is a real speed-up over a serial run
   for assembly-dominated problems, but the solve itself does not scale and rank 0 needs the whole
   matrix in memory. For a genuinely distributed solve use ``--petsc_mumps``, which is also what
   pyoomph selects by default under ``mpirun`` when PETSc/MUMPS is available.

   Because the waiting processes have to leave their cores free for rank 0's threads, they sleep
   rather than spin. On most MPI implementations the processes are pinned to individual cores by
   default, which stops rank 0 from using the freed ones -- run ``mpirun --bind-to none`` and set the
   thread count with :py:meth:`~pyoomph.generic.problem.Problem.set_num_threads` if you want rank 0 to
   spread out. pyoomph prints a one-off note when it detects this.

``--petsc_mumps``
      Use PETSc as linear solver with MUMPS as the underlying factorization backend.

``--accelerate``
      Use Apple's Accelerate sparse solver framework (macOS only).

Likewise, the eigensolver flags are mutually exclusive:

``--slepc``
      Use `SLEPc <https://slepc.upv.es/>`__ as eigensolver, see :numref:`petscslepc`. You still have to specify a linear solver (e.g. together with ``--slepc_mumps``, or via :py:meth:`~pyoomph.generic.problem.Problem.set_eigensolver` in your script) for the matrix inversion during the eigensolve.

``--slepc_mumps``
      Use SLEPc as eigensolver with MUMPS as backend. This is the default whenever PETSc/SLEPc with MUMPS support is installed, so the flag is only required to override an eigensolver selected in the driver code itself.

``--arpack``
      Use scipy's ARPACK-based eigensolver. Note that this is scipy's own backend, not the ARPACK-with-Pardiso combination which pyoomph selects by default when SLEPc/MUMPS is missing but MKL Pardiso is present.

And the C compiler backend used to build the just-in-time generated code is likewise mutually exclusive between:

``--tcc``
      Use the internal TCC compiler.

``--distutils``
      Use the system C compiler, as detected by ``distutils``.

``--fast-math``
      Activate fast-math compiler flags. Only usable together with ``--distutils`` (or the default compiler), not with ``--tcc``.

      It rarely pays off. The default flags are already quite optimized. Wrapping expensive terms in :py:func:`~pyoomph.expressions.generic.subexpression` addresses that far better: the term is then evaluated once into a temporary and its derivatives are cached as well, which beats ``--fast-math`` outright. With ``subexpression`` in place, ``--fast-math`` usually adds less than a percent.

Output and code generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--outdir OUTDIR``
      Set the output directory. If not given, the output directory defaults to the name of the driver script (without the ``.py`` extension). See :numref:`secodecmdline` for an example combining this with ``-P`` below.

``--suppress_code_writing``
      Do not write the generated FEM C code to disk. Useful for debugging.

``--suppress_compilation``
      Do not compile the generated FEM C code. Useful for debugging.

``--no-cache``
      Do not use the JIT code cache (see :numref:`installenvvars`) -- always regenerate and recompile the FEM code from scratch, ignoring the ``PYOOMPH_JIT_CACHE*`` environment variables for this run.

``--distribute``
      Distribute the mesh in parallel (MPI), see :numref:`secmpimodes` for more details.
      
      Some features are not available with this flag, but pyoomph will let you know.

``--mpi-output {condensed,all,off}``
      What reaches the terminal when several ranks write to it at once, see :numref:`secmpioutput`.
      ``condensed`` (the default) lets only rank 0 print, while the stderr of every rank still gets
      through, tagged; ``all`` prints every rank, each line in one piece and tagged ``[rank N]``;
      ``off`` does not filter at all. Without ``mpirun`` the flag is accepted and does nothing.
      It can also be set with the ``PYOOMPH_MPI_OUTPUT`` environment variable
      (cf. :numref:`installenvvars`), which this flag overrides.


Threading
~~~~~~~~~

``--omp N``
      Use ``N`` threads for the element assembly and for the linear solver. Without it, pyoomph
      assembles on one thread, but might solve with OpenMP threads, which depends on the selected solver.
      
      Some features, like a finite-difference Jacobian (which perturbs data shared with the neighbouring elements) won't work with that. Pyoomph will let you know.
      Also, it is required to build with OpenMP.

      It combines with ``mpirun``: threads run inside each rank. Keep ``ranks x N`` at or below the
      number of physical cores.

      .. warning::

         ``mpirun`` pins each rank to a single core by default, which makes ``--omp N`` a no-op: the
         threads are created and the results stay correct, but they take turns on that one core
         instead of running side by side. Launch with ``mpirun --bind-to none`` (or a binding that
         gives each rank as many cores as it has threads). pyoomph prints a one-off note when it
         notices.
         
         With the binding fixed, ranks and threads are roughly interchangeable for the Jacobian, and
         ranks are clearly better for the residual. Reach for ``--omp`` under ``mpirun`` mainly for
         what more ranks cannot give you: memory headroom, or a core count beyond what the mesh
         partitioner divides sensibly.

      :py:meth:`~pyoomph.generic.problem.Problem.set_num_threads` does the same from the driver
      script. :py:class:`~pyoomph.utils.paramscan.ParallelParameterScan` runs its children
      single-threaded by default, since it is already running one simulation per core.

Run control
~~~~~~~~~~~

``--runmode {d,delete,o,overwrite,c,continue,p,replot}``
      Selects what to do with a pre-existing output directory. ``delete`` (the default) removes previous output before starting; ``overwrite`` starts without removing anything; ``continue`` (``c``) resumes a previously stopped simulation from its last written state, see :numref:`secpdecontinue`; ``replot`` (``p``) only redoes the plots of an already completed simulation, without solving anything again, see :numref:`secreplotting`.

``--recompile_on_continue``
      When using ``--runmode c`` or ``--runmode p``, code writing/compilation is normally suppressed (the existing generated code is reused). Pass this flag to force a recompile anyhow.

``--where EXPRESSION``
      A Python ``bool`` expression involving the variables ``step`` and/or ``time``, e.g. ``"step==10"`` or ``"step in [10,11,20]"``. Only used together with ``--runmode c``/``--runmode p``, to restrict which output steps are considered.

``--quick-test``
      Stop right after the first successful Newton solve (after writing that one output). Useful for quickly checking that a script runs at all, e.g. in CI.

Overriding parameters
~~~~~~~~~~~~~~~~~~~~~~

``-P NAME=VALUE [NAME=VALUE ...]``, ``--parameter NAME=VALUE [NAME=VALUE ...]``
      Override problem parameters (attributes of the :py:class:`~pyoomph.generic.problem.Problem`, of :py:class:`~pyoomph.generic.codegen.GiNaC_GlobalParam` global parameters, or of nested attributes reachable via dotted names, e.g. ``some_equation.some_property``) from the command line, without editing the script. Besides plain assignment (``=``), you can also use ``*=``, ``/=``, ``+=`` or ``-=`` to multiply, divide, increment or decrement the current value (write the operator right before the ``=``, e.g. ``spring_constant*=2``). See :numref:`secodecmdline` for a full example.

preCICE coupling
~~~~~~~~~~~~~~~~~

``--generate_precice_cfg``
      Generate parts of a `preCICE <https://precice.org/>`__ configuration file from the coupling equations defined in the problem, then exit. See :numref:`secprecice` for how to set up preCICE coupling in pyoomph.

Diagnostics
~~~~~~~~~~~

``--verbose``
      Print a lot of additional output.

``--largest_residuals N``
      Debug the ``N`` largest residual contributions after each Newton solve.

.. _installcmdlinecustom:

Adding your own options
~~~~~~~~~~~~~~~~~~~~~~~~

If your own :py:class:`~pyoomph.generic.problem.Problem` subclass needs additional command line options, override ``setup_cmd_line()`` (calling the base class implementation first, then adding further ``self.cmdlineparser.add_argument(...)`` calls) and ``parse_cmd_line()`` (again calling the base class implementation first) to read them out of ``self.cmdlineargs`` afterwards. You can also override ``cmdline_desc()`` to customize the description shown in ``--help``.

Finally, if you want to fully bypass command line parsing for a particular :py:class:`~pyoomph.generic.problem.Problem` instance (e.g. when constructing and running it programmatically from another script), set ``ignore_command_line=True`` in its constructor.
