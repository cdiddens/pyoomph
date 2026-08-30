#  Profiles Problem.initialise() and attributes the time to individual C++ (nanobind) entry points.
#
#  cProfile cannot see nanobind methods at all - their cost is silently folded into the tottime of
#  whatever Python function called them - and `perf` needs a kernel.perf_event_paranoid the
#  development machines do not grant. So the entry points are wrapped by hand, on the Python
#  subclass, which shadows the inherited C++ binding.
#
#  Reported times are SELF times: a wrapper subtracts whatever its own wrapped children consumed, so
#  the column sums to the total instead of counting nested calls repeatedly.
#
#  Usage:  python3 initialisation_profile.py [N] [1|2|3]
#  See dev_docs/initialisation_cost.md for the numbers this produced and what they mean.

import sys, os, time, traceback

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DIM = int(sys.argv[2]) if len(sys.argv) > 2 else 2

from pyoomph import *
from pyoomph.equations.poisson import *
from pyoomph.generic.problem import Problem
from pyoomph.generic.codegen import FiniteElementCodeGenerator
from pyoomph.meshes.mesh import MeshFromTemplate1d, MeshFromTemplate2d, MeshFromTemplate3d, InterfaceMesh
from pyoomph.meshes.simplemeshes import LineMesh, RectangularQuadMesh, CuboidBrickMesh

SELF: dict[str, float] = {}
CNT: dict[str, int] = {}
SITES: dict[str, dict[str, int]] = {}
CHILD = [0.0]


def _wrap(orig, label):
    def f(self, *a, **k):
        c = traceback.extract_stack(limit=2)[0]
        key = "%s:%d %s" % (os.path.basename(c.filename), c.lineno, c.name)
        d = SITES.setdefault(label, {})
        d[key] = d.get(key, 0) + 1
        saved = CHILD[0]
        CHILD[0] = 0.0
        t = time.perf_counter()
        try:
            return orig(self, *a, **k)
        finally:
            dt = time.perf_counter() - t
            SELF[label] = SELF.get(label, 0.0) + dt - CHILD[0]
            CNT[label] = CNT.get(label, 0) + 1
            CHILD[0] = saved + dt

    return f


def wrap(cls, names, prefix):
    # Wrap on the CONCRETE class. MeshFromTemplateBase is not in the MRO of MeshFromTemplate1d/2d/3d
    # (they are siblings, not subclasses), so setting an attribute there silently does nothing and
    # the routine looks free.
    for m in names:
        o = getattr(cls, m, None)
        if o is None:
            continue
        setattr(cls, m, _wrap(o, prefix + m))


CPP_MESH = ["generate_from_template", "setup_tree_forest", "setup_boundary_element_info", "_set_problem",
            "setup_Dirichlet_conditions", "_pin_noncontributing_dofs", "setup_initial_conditions",
            "clear_additional_dof_constraints", "apply_additional_dof_constraints", "check_integrity",
            "assign_global_base_element_indices", "describe_global_dofs"]
PY_MESH = ["_finalise_creation", "_compile_bulk_equations", "_setup_output_scales",
           "setup_initial_conditions_with_interfaces", "_generate_interface_elements",
           "_compile_interface_equations"]

wrap(Problem, ["assign_eqn_numbers", "assign_initial_values_impulsive", "ensure_dummy_values_to_be_dummy",
               "_unpin_Dirichlet_dofs_for_matrix_manipulation", "rebuild_global_mesh", "build_global_mesh"], "C++ ")
wrap(Problem, ["_link_geometry_and_equations", "compile_meshes", "rebuild_global_mesh_from_list", "initialise",
               "_assemble_defined_field_list", "_get_jacobian_information_string", "setup_pinning", "init_output",
               "reapply_boundary_conditions", "map_nodes_on_macro_elements", "set_initial_condition",
               "relink_external_data", "before_assigning_equation_numbers", "_set_solved_residual"], "PY ")
wrap(FiniteElementCodeGenerator, ["on_apply_boundary_conditions", "_do_define_fields"], "C++ codegen.")
for _cls, _tag in ((MeshFromTemplate1d, "bulk"), (MeshFromTemplate2d, "bulk"), (MeshFromTemplate3d, "bulk"),
                   (InterfaceMesh, "iface")):
    wrap(_cls, CPP_MESH, "C++ %s." % _tag)
    wrap(_cls, PY_MESH, "PY %s." % _tag)

p = Problem()
if DIM == 1:
    p += LineMesh(N=N)
elif DIM == 2:
    p += RectangularQuadMesh(size=[1, 1], N=[N, N])
else:
    p += CuboidBrickMesh(size=[1, 1, 1], N=[N, N, N])
p += (PoissonEquation(source=1) + DirichletBC(u=0) @ "*") @ "domain"
p.initial_adaption_steps = 0
# Output goes into the CURRENT directory, not next to the script: run this from a scratch folder so
# it does not drop a mesh dump into dev_docs/examples/.
p.set_output_directory("_initprof_%dd_%d" % (DIM, N))

t0 = time.perf_counter()
p.initialise()
tot = time.perf_counter() - t0

print("\n=== %dd N=%d  ndof=%d  initialise = %.2f s  (self times)" % (DIM, N, p.ndof(), tot))
acc = 0.0
for k, v in sorted(SELF.items(), key=lambda x: -x[1]):
    if v <= 0.01:
        continue
    acc += v
    print("  %-48s %7.3f s  (%d calls)" % (k, v, CNT[k]))
    for s, c in sorted(SITES[k].items(), key=lambda x: -x[1])[:3]:
        print("        <- %s x%d" % (s, c))
print("  %-48s %7.3f s  (%.0f%% of total)" % ("[accounted for]", acc, 100 * acc / tot))
