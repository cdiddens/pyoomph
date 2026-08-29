.. _secvibecoding:

Vibe coding with pyoomph
========================

Writing a pyoomph script is largely a matter of knowing which class to reach for and how the
weak form has to be scaled. Both are things a language model can do for you, provided it
actually knows them - and no model has read pyoomph's source. The repository therefore ships a
reference written for that purpose: :download:`AGENTS.md <../../../AGENTS.md>` in the root,
together with a set of companion files in the ``agents/`` directory.

These are not the tutorial in a different format. They are condensed, code-verified, and they
concentrate on the things that are hard to guess and easy to get wrong: which keyword a
constructor actually takes, why a residual has to be dimensionless, which conditions a
coordinate system implies on a symmetry axis. Their contents have been checked by giving them
to an assistant that had no other access to pyoomph, asking for a script solving a problem with
a known answer, and running the result - so what is written there is what was needed to make
such scripts work.

How to use it
-------------

Point your assistant at ``AGENTS.md`` before asking for a script:

.. code:: none

   Read AGENTS.md, then write me a pyoomph script that ...

That single file is meant to be enough for an ordinary problem. It carries a routing table at
the end naming the companion files and when each is worth opening, and a capable assistant will
follow those links by itself when the task calls for it:

.. list-table::
    :widths: 25 75
    :header-rows: 1

    *   - File
        - Covers
    *   - ``agents/units.md``
        - Units, scales, and the rule that every residual must be dimensionless.
    *   - ``agents/physics.md``
        - The built-in equation classes and their constructor keywords.
    *   - ``agents/meshes.md``
        - Mesh templates, custom and gmsh meshes, multiple domains, moving meshes, remeshing.
    *   - ``agents/output.md``
        - Output files, observables, evaluating the solution at a point, plotting.
    *   - ``agents/examples.md``
        - Seven worked recipes to start from.
    *   - ``agents/materials.md``
        - The material library: fluids, mixtures, interfaces, surfactants, mass transfer.
    *   - ``agents/advanced.md``
        - Stability and bifurcations, custom C code, discontinuous Galerkin.
    *   - ``agents/parallel.md``
        - OpenMP and MPI.

The files are part of the repository, not of the installed package, so a ``pip install pyoomph``
does not put them on your disk. Fetch them from https://github.com/pyoomph/pyoomph if you
installed that way.

If you use a coding assistant that reads a project file automatically - the name ``AGENTS.md``
is a convention several of them follow - it will pick the file up on its own as soon as you work
inside a clone of the repository.

What to expect, and what to check
---------------------------------

An assistant working from these files will usually get the structure of a script right: the
``Equations`` subclass, the ``Problem``, the boundary conditions, the entry point. What it
cannot do is tell you whether the answer is correct.

The failure mode to worry about is not a script that crashes. Pyoomph is fairly good at
refusing malformed input, and when it does refuse, the message is normally specific enough to
hand straight back to the assistant - the complaint that *"the added residual contribution is
not dimensionless"* even prints the scales it used and the offending term. The dangerous case is
a script that runs to completion and is quietly wrong: a coefficient with the wrong units
absorbed into a scale, two domains whose meshes have come apart, a boundary condition applied to
the wrong side of a corner.

So treat the generated script as a draft to be verified, and verify it against something you
know independently of the simulation:

*   Start from a case with an analytical solution, or a limit of your problem that has one, and
    compare numbers rather than pictures.
*   Check the quantities that must hold whatever the discretisation: a conserved mass or volume,
    a symmetry, a flux balance, a value that must vanish.
*   Refine the mesh or tighten the time-stepping tolerance and confirm the answer moves the way
    it should.
*   Run once with ``--quick-test`` first (:numref:`installcmdlineoptions`), which stops after the
    first successful Newton solve and writes one output. That tells you the problem is
    well-posed and assembles, in seconds rather than hours.

Be explicit about the physics you want, in the same way you would be with a colleague. State the
geometry, the boundary conditions and the units; say whether a quantity is dimensional; and say
which limit you are in. An assistant asked for "a droplet simulation" will invent all of that,
plausibly and without telling you.

Finally, prefer the built-in equation classes. ``AGENTS.md`` says so on its first page, and it
matters more for generated code than for hand-written code: a hand-rolled Navier-Stokes weak
form is a great deal of surface for a subtle sign error, whereas
:py:class:`~pyoomph.equations.navier_stokes.NavierStokesEquations` has been used and tested for
years. If the assistant writes a weak form by hand where a class exists, ask it why.
