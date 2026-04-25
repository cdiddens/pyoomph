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

import sys
import _pyoomph

class _LogWrapper(object):
    
    def __init__(self,terminal,is_stderr=False):
        self.terminal = terminal     
        self.is_stderr=is_stderr   
   
    def write(self, message):        
        self.terminal.write(message)
        _pyoomph._write_to_log_file(message)

    def flush(self):        
        pass    
    
    def __del__(self):
        if self.is_stderr:
           sys.stderr=self.terminal
        else:
           sys.stdout=self.terminal
    
        
def pyoomph_activate_logging_to_file():	
    if not isinstance(sys.stdout,_LogWrapper):
        sys.stdout = _LogWrapper(sys.stdout)
        sys.stderr = _LogWrapper(sys.stderr)

