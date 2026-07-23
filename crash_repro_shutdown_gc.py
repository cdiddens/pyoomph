# Minimal repro for the interpreter-shutdown segfault investigated on 2026-07-20/21.
#
# A Problem left as a live module-level global (no "with", no explicit .release(),
# no explicit "del problem; gc.collect()") crashes on ordinary process exit, even
# for the simplest possible mesh/equation. This is NOT specific to complex meshes
# (originally found via rivulet.py's contact-line geometry) - this trivial script
# reproduces the identical crash signature (corrupted vtable pointer read inside
# BulkElementBase::free_element_info(), destructed via Python's generic GC tp_dealloc
# chain rather than release()'s _destroy_now() path).
#
# Confirmed causes/non-causes (see project_nanobind_migration.md memory for full
# writeup):
#   - Problem.__del__ does not reliably fire at all once interpreter shutdown starts
#     clearing module dicts.
#   - When it does fire (e.g. reached via sys.exit()), ordinary operations inside it
#     (a bare "import") can fail because import machinery is already torn down.
#   - Guarding with `sys.is_finalizing(): return` inside __del__ does NOT prevent the
#     crash either - the mesh still gets destroyed later via plain GC teardown, which
#     is the actually-unsafe path (skips the explicit _destroy_now() ordering).
#   - Reproduces identically whether the script ends naturally or calls sys.exit(0).
#   - Does NOT reproduce if you explicitly do `del problem; gc.collect()` mid-script
#     (while the interpreter is still fully alive) instead of leaving it to shutdown.
#
# Run directly: python3 crash_repro_shutdown_gc.py  (expect a segfault on exit)
from pyoomph import *
from pyoomph.expressions import *
from pyoomph.meshes.simplemeshes import LineMesh

class MyEquation(Equations):
    def define_fields(self):
        self.define_scalar_field('u','C2')
    def define_residuals(self):
        u,utest=var_and_test('u')
        self.add_residual(weak(grad(u),grad(utest)))

problem=Problem()
problem+=MyEquation()@'domain'
problem.add_mesh(LineMesh(N=5))
problem.solve()
print("ending normally (no with block, no del/gc.collect(), no sys.exit)", flush=True)
