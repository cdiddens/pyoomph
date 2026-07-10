/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
Copyright (C) 2021-2026  Christian Diddens, Duarte Rocha & Maxim de Wildt

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>. 

The main author may be contacted at c.diddens@utwente.nl

================================================================================*/


#include "exception.hpp"
#include "logging.hpp"

namespace pyoomph
{
    int pyoomph_verbose = 0;

    runtime_error_with_line::runtime_error_with_line(const std::string &arg, const char *file, int line) : std::runtime_error(arg)
    {
        std::ostringstream o;
        o << file << ":" << line << ": " << arg;
        msg = o.str();
/*        if (get_logging_stream())
        {
            *get_logging_stream() << "RuntimeError: " << msg << std::endl << std::flush;
            
        }*/
    }
    
    
    const char *runtime_error_with_line::what() const throw()
    {
        
        return msg.c_str();
    }
    
    
}
