import sys
stubfile=sys.argv[1]

with open(stubfile, 'r') as file :
	filedata = file.read()

def patch_stub(sea:str,repl:str,ignore_if_not_found:bool=False):
	global filedata
	if (not ignore_if_not_found) and filedata.count(sea)==0:
		raise RuntimeError("Cannot replace in stub file (not found): "+sea)
	filedata = filedata.replace(sea, repl)
	pass


# nanobind.stubgen (unlike the old pybind11-stubgen) already infers "| None" for
# nullable-pointer parameters and already emits "numpy.typing.NDArray[...]" directly, so
# most of the old patches for those are no longer needed. What it does NOT infer is
# nullability of *return* values, since that isn't statically knowable from the binding -
# these three methods can return None (documented as such), so patch their return type.
patch_stub("def _get_parent_domain(self) -> FiniteElementCode:","def _get_parent_domain(self) -> FiniteElementCode | None:")
patch_stub("def _get_opposite_interface(self) -> FiniteElementCode:","def _get_opposite_interface(self) -> FiniteElementCode | None:")
patch_stub("def _resolve_based_on_domain_name(self, domainname: str) -> FiniteElementCode:","def _resolve_based_on_domain_name(self, domainname: str) -> FiniteElementCode | None:")

# These bind a plain "const std::vector<double>&" C++ parameter, whose nanobind caster
# also happily accepts a 1D numpy float array at runtime (it iterates any Python sequence),
# but nanobind.stubgen only ever emits "Sequence[float]" for std::vector<double>. Widen the
# annotation for the handful of Problem methods callers commonly invoke with numpy arrays
# directly (e.g. the result of numpy.array(...)/numpy.linspace(...)), so callers don't need
# a spurious #type:ignore at every call site.
patch_stub(
    "def _update_dof_vectors_for_continuation(self, ddof: Sequence[float], current: Sequence[float]) -> None:",
    "def _update_dof_vectors_for_continuation(self, ddof: Sequence[float] | NDArray[numpy.floating], current: Sequence[float] | NDArray[numpy.floating]) -> None:")
patch_stub(
    "def _set_dof_direction_arclength(self, direction: Sequence[float]) -> None:",
    "def _set_dof_direction_arclength(self, direction: Sequence[float] | NDArray[numpy.floating]) -> None:")
patch_stub(
    "def set_current_dofs(self, values: Sequence[float]) -> None:",
    "def set_current_dofs(self, values: Sequence[float] | NDArray[numpy.floating]) -> None:")
patch_stub(
    "def set_history_dofs(self, t: int, values: Sequence[float]) -> None:",
    "def set_history_dofs(self, t: int, values: Sequence[float] | NDArray[numpy.floating]) -> None:")
patch_stub(
    "def set_current_pinned_values(self, values: Sequence[float], with_position_dofs: bool, t: int = 0) -> None:",
    "def set_current_pinned_values(self, values: Sequence[float] | NDArray[numpy.floating], with_position_dofs: bool, t: int = 0) -> None:")

# Write the file out again
with open(stubfile, 'w') as file:
  file.write(filedata)
