#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
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
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================
 
from pathlib import Path

from ...typings import *
if TYPE_CHECKING:
    from ...generic.problem import Problem


class BasiliskCodeFile:
    def __init__(self,filename:Optional[str]) -> None:
        self.filename=filename

    def get_contents(self) -> str:
        return ""

    def get_include_statement(self) -> str:
        if self.filename is None:
            return self.get_contents()
        else:
            return '#include "'+self.filename+'"\n'
    

class BasiliskCodeGeneratorBase:
    def __init__(self,problem:"Problem",codedir:Optional[str]=None) -> None:
        super().__init__()
        self._problem:"Problem"=problem
        self._codedir=codedir
        self._includes:List[BasiliskCodeFile]=[]
        pass

    def get_code_directory(self) -> str:
        if self._codedir is None:
            self._codedir=self.get_problem().get_output_directory("_basilisk_src")        
        Path(self._codedir).mkdir(parents=True, exist_ok=True)
        return self._codedir

    def get_grid_size(self):
        raise RuntimeError("Implement this")

    def get_problem(self)->"Problem":
        return self._problem
    
    def define_basilisk_code(self):
        raise RuntimeError("Implement this")
    
    def compile_basilisk_code(self):
        raise RuntimeError("TODO")
    
    def run_basilisk_code(self):
        raise RuntimeError("TODO")
    

class TwoPhaseBasiliskCodeBase(BasiliskCodeGeneratorBase):
    pass