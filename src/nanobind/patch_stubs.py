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

# Write the file out again
with open(stubfile, 'w') as file:
  file.write(filedata)
