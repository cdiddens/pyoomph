# pyoomph — output, plotting and reading results back

Companion to [`AGENTS.md`](../AGENTS.md). How to get numbers and pictures out of a
solved problem: output equations, observables, point evaluation, and the plotters.

## Output equations

Output equations (`pyoomph/output/*.py`) are added to a domain with `+=` like any other
equation and are written whenever `problem.output()` runs:

| Class | Writes |
|---|---|
| `MeshFileOutput()` | VTU/mesh files for ParaView — the default choice for 2D/3D fields. |
| `TextFileOutput()` | Plain-text dump of the nodal values. Good for 1D and for plotting with anything. |
| `TextFileOutputAlongLine(start=, end=, N=)` | Values sampled along a line/curve through the mesh, independent of the nodes. |
| `GridFileOutput(lower, upper, N=/dx=)` | Interpolation onto a regular Cartesian grid. |
| `ODEFileOutput(first_column=...)` | For `ODEEquations` domains; `first_column=` writes e.g. a parameter instead of time. |
| `IntegralObservableOutput()` | The `IntegralObservables`/`ExtremumObservables` time series. |

For a scan or summary file not tied to the output steps, use
`problem.create_text_file_output(fname, header=[...])` and `.add_row(...)`. **Call it only
after the problem is initialised** — it opens the file in the output directory
immediately, and that directory is created by the first `solve()`/`run()`, or by an
explicit `problem.initialise()`. Calling it as the first statement of a `with` block fails
with `FileNotFoundError: 'myproblem/scan.txt'`.

Each `problem.output()` writes a new numbered file (`cavity_000000.vtu`,
`cavity_000001.vtu`, ...), so calling it repeatedly in a stationary parameter scan is
fine and nothing is overwritten. The successive files do all carry the *same* time
though, which makes ParaView overlay them when it reads the `.pvd`; pass
`problem.output(increase_time_for_PVD=True)` to advance the time stamp so a parameter
scan plays back as a series. (`output_at_increased_time()` does the same and is
deprecated.)

## Observables and reading values back

**`IntegralObservableOutput()` is how you get a scalar time series out of a PDE domain.**
Add it next to the `IntegralObservables` and it writes **one file per domain**,
`<domain-path>_IntObsv.txt` with `/` replaced by `__` (so a corner of a bulk domain lands
in `droplet__free_surface__axis_IntObsv.txt`), a `#`-comment header naming each column and
its unit (`#time[s]  Volume[m^3]`), and one row per output step — directly
`numpy.loadtxt`-able. Observables on different domains therefore land in different files;
there is no way to merge them into one. Do **not** chop `run(...)` into many short calls just to read a
value after each one: every call has to land exactly on its end time, the leftover steps
shrink, and the timestep eventually collapses and the Newton solve fails. Run once and
read the file afterwards.

`IntegralObservables(**exprs, coordsys=None)` also takes a **`coordsys=` override**, which
matters on a symmetry axis: the axisymmetric measure carries a factor `2*pi*r` (so
`IntegralObservables(Volume=1)` on an axisymmetric bulk domain really is the volume), and
that factor annihilates anything evaluated on the `r=0` boundary or on a corner point
there — pass `coordsys=cartesian` for those. A corner of a bulk boundary is addressed as
`@ "boundary_a/boundary_b"`, and when that corner holds two points (e.g. both poles of a
half-disc droplet) an expression like `maximum(var("coordinate_y"), 0*meter)` picks one. `ExtremumObservables(*fieldnames, **named_exprs)` records min/max instead of an integral:
`ExtremumObservables("u")` monitors a field by name, `ExtremumObservables(v_norm=square_root(dot(var("v"),var("v"))))`
monitors any expression. **It is read back with its own methods** —
`mesh.evaluate_maximum(name)` / `mesh.evaluate_minimum(name)`, returning dimensional
values — and is **not** written into the `_IntObsv.txt` file by
`IntegralObservableOutput()`, which silently ignores it. For a time series of an extremum,
sample it yourself, or express the same quantity as an `IntegralObservables` entry.

**Observables** are read back with `problem.get_mesh(domain).evaluate_observable(name)`.

**To read the solution at an arbitrary point** (not just an integral), use
`mesh.evaluate_at_points(coords, lagrangian=True, with_position=False)`. Coordinates are
**nondimensional** (divide by the `spatial` scale) and it returns one row per point,
`[found, <continuous fields...>, <DL fields>, <D0 fields>, <position if requested>]`, with
`found=0` and nothing else if the point is outside the mesh. The column order of the
continuous block is given by `mesh.get_nodal_field_indices()`, a `{name: index}` dict —
look it up rather than assuming. Values come back nondimensional too, so multiply by the
scale:

```python
m = problem.get_mesh("domain")
idx = m.get_nodal_field_indices()                 # e.g. {'velocity_x':0,'velocity_y':1,'temperature':2}
row = m.evaluate_at_points([[0.5, 0.25]])[0]      # nondimensional coordinates
T = row[1 + idx["temperature"]] * problem.get_scaling("temperature")
```
On a solid or moving mesh, `lagrangian=True` (the default) locates the point by its
*undeformed* coordinate, which is usually what you want; `with_position=True` appends the
current position, so a displacement is `position - lagrangian_coordinate`.

**It does not sample nodal DG fields** (`D1`/`D2`/`D1TB`/`D2TB`): the row comes back as
just `[found]`, even though `get_nodal_field_indices()` still lists the field — do not
trust that dict for a DG field, check `mesh.get_field_information()` (`{name: space}`)
instead. To read a discontinuous field back, use `TextFileOutput()`, whose dump *does*
contain the DG nodal values, or project it onto a continuous field first with
`ProjectExpression(c_cont=var("c"))`. `TextFileOutput()` writes to
`<outdir>/<domain>/<domain>_<NNNNNN>.txt` — **a subdirectory named after the domain**, not
the output directory itself — with a header like `# coordinate_x	c	@time=0.0`, one row
per node, and both coordinates and values in **physical units**. A node shared by two
elements appears twice for a DG field, once per side.
Unlike `weak(...)`, `IntegralObservables` integrates over the **physical, dimensional**
domain: `Area=1` evaluates to the real area *with units* (1e-4 m² for a 1 cm square), and
`evaluate_observable` returns a dimensional `Expression`, not a float. So

```python
eqs += IntegralObservables(Area=1, Usqr=dot(var("velocity"), var("velocity")),
                           Urms=lambda Usqr, Area: square_root(Usqr/Area))
...
Urms = float(problem.get_mesh("cavity").evaluate_observable("Urms")/(milli*meter/second))
```
Entries may reference earlier ones by name through a `lambda`, as `Urms` does here. Divide
by a unit (or by `problem.get_scaling(...)`) *before* calling `float()` — `float()` on a
value that still carries a unit raises a `RuntimeError` that names the leftover unit, and
`create_text_file_output(...).add_row(...)` calls `float()` on everything you hand it.

## Plotting

**Plotting during the run.** Subclass a plotter, override `define_plot(self)`, and assign
an instance to `problem.plotter`; it then renders at every `problem.output()`.

- `pyoomph.output.plotting.MatplotlibPlotter` — 2D (and axisymmetric) field plots.
  Inside `define_plot`: `self.set_view(xmin, ymin, xmax, ymax)`, `self.add_colorbar(...)`,
  then `self.add_plot("domain/field", colorbar=cb)` for a colour map, or with
  `mode="arrows"`/`"streamlines"` for a vector field. `"domain/boundary"` as the first
  argument draws the interface line (the interface needs at least one equation on it — an
  empty `InterfaceEquations()` is enough). `transform=["mirror_x", None]` plots mirrored
  and unmirrored copies side by side, the usual way to show an axisymmetric result.
  Also `add_arrow_key`, `add_text`, `add_time_label`, `add_scale_bar`.
- `pyoomph.output.plotting1d.MatplotlibPlotter1D` — 1D domains as ordinary x-y graphs
  (including the (x,y) curve of a 1D mesh embedded in 2D/3D).
- `pyoomph.output.plotting3d.PyVistaPlotter` — 3D rendering via PyVista.

Plots can be regenerated from written output without re-solving: `--runmode p`.

