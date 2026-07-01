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

The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl

================================================================================*/


#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace pyoomph
{
    class runtime_error_with_line : public std::runtime_error
    {
    protected:
        std::string msg;

    public:
        runtime_error_with_line(const std::string &arg, const char *file, int line);
        ~runtime_error_with_line() throw() {}
        const char *what() const throw();
        
    };

    extern int pyoomph_verbose;

}
#define throw_runtime_error(arg) throw pyoomph::runtime_error_with_line(arg, __FILE__, __LINE__);
