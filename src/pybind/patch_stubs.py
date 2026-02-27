stubfile="_pyoomph-stubs/_pyoomph.pyi"

with open(stubfile, 'r') as file :
	filedata = file.read()
    
def patch_stub(sea:str,repl:str,ignore_if_not_found:bool=False):
	global filedata
	if (not ignore_if_not_found) and filedata.count(sea)==0:
		raise RuntimeError("Cannot replace in stub file (not found): "+sea)
	filedata = filedata.replace(sea, repl)
	pass


try:
	patch_stub("GiNaC_field(arg0: str, arg1: FiniteElementCode, arg2: typing.List[str]) -> Expression","GiNaC_field(arg0: str, arg1: typing.Optional[FiniteElementCode], arg2: typing.List[str]) -> Expression")

	patch_stub("GiNaC_nondimfield(arg0: str, arg1: FiniteElementCode, arg2: typing.List[str]) -> Expression","GiNaC_nondimfield(arg0: str, arg1: typing.Optional[FiniteElementCode], arg2: typing.List[str]) -> Expression")

	patch_stub("GiNaC_scale(arg0: str, arg1: FiniteElementCode, arg2: typing.List[str]) -> Expression","GiNaC_scale(arg0: str, arg1: typing.Optional[FiniteElementCode], arg2: typing.List[str]) -> Expression")

	patch_stub("GiNaC_testscale(arg0: str, arg1: FiniteElementCode, arg2: typing.List[str]) -> Expression","GiNaC_testscale(arg0: str, arg1: typing.Optional[FiniteElementCode], arg2: typing.List[str]) -> Expression")

	patch_stub("GiNaC_dimtestfunction(arg0: str, arg1: FiniteElementCode, arg2: typing.List[str]) -> Expression","GiNaC_dimtestfunction(arg0: str, arg1: typing.Optional[FiniteElementCode], arg2: typing.List[str]) -> Expression")

	patch_stub("GiNaC_testfunction(arg0: str, arg1: FiniteElementCode, arg2: typing.List[str]) -> Expression","GiNaC_testfunction(arg0: str, arg1: typing.Optional[FiniteElementCode], arg2: typing.List[str]) -> Expression")

	patch_stub("GiNaC_eval_in_domain(arg0: Expression, arg1: FiniteElementCode, arg2: typing.List[str]) -> Expression","GiNaC_eval_in_domain(arg0: Expression, arg1: typing.Optional[FiniteElementCode], arg2: typing.List[str]) -> Expression")

	patch_stub("set_macro_element(self, arg0: MacroElement, arg1: bool) -> None","set_macro_element(self, arg0: typing.Optional[MacroElement], arg1: bool) -> None")

	patch_stub("_set_current_codegen(self, arg0: FiniteElementCode) -> None","_set_current_codegen(self, arg0: typing.Optional[FiniteElementCode]) -> None")

	patch_stub("_resolve_based_on_domain_name(self, domainname: str) -> FiniteElementCode","_resolve_based_on_domain_name(self, domainname: str) -> typing.Optional[FiniteElementCode]")
except:
	pass
	
patch_stub("set_latex_printer(self, arg0: LaTeXPrinter) -> None","set_latex_printer(self, arg0: typing.Optional[LaTeXPrinter]) -> None")

patch_stub("_get_parent_domain(self) -> FiniteElementCode","_get_parent_domain(self) -> typing.Optional[FiniteElementCode]")
patch_stub("_get_opposite_interface(self) -> FiniteElementCode","_get_opposite_interface(self) -> typing.Optional[FiniteElementCode]")

patch_stub("_set_problem(self, arg0: Problem, arg1: DynamicBulkElementInstance) -> None","_set_problem(self, arg0: Problem, arg1: typing.Optional[DynamicBulkElementInstance]) -> None")


patch_stub("import numpy","import numpy; import numpy.typing")
patch_stub("numpy.ndarray[numpy.float64]","numpy.typing.NDArray[numpy.float64]",True)
patch_stub("numpy.ndarray[numpy.int32]","numpy.typing.NDArray[numpy.int32]",True)
patch_stub("numpy.ndarray[numpy.uint64]","numpy.typing.NDArray[numpy.uint64]",True)
patch_stub("numpy.ndarray[numpy.uint32]","numpy.typing.NDArray[numpy.uint32]",True)

# Write the file out again
with open(stubfile, 'w') as file:
  file.write(filedata)

