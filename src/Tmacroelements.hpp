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


/*
 Rather dummy class to allow MacroElements on TElements
 
*/
#include "oomph_lib.hpp"
#include "exception.hpp"
namespace oomph
{

   // Generic (unimplemented) template: oomph-lib's MacroElement concept (used for curved/mapped
   // domain boundaries via Domain objects) is only specialized for 2D (triangular) T-elements
   // below; other dimensions get an empty, unusable class.
   template <int DIM>
   class TMacroElement : public MacroElement
   {
   };

   // 2d specialization allowing MacroElements to be attached to T (triangular) elements. oomph-lib's
   // own MacroElement machinery is written for quad/brick elements; this class exists purely so a
   // MacroElement pointer can be associated with a TElement-based mesh without crashing, but none of
   // the actual macro-mapping functionality is implemented (all overrides throw "Not implemented").
   template <>
   class TMacroElement<2> : public MacroElement
   {

   public:
      /// \short Constructor: Pass the pointer to the domain and the macro element's
      /// number within this domain
      TMacroElement(Domain *domain_pt, const unsigned &macro_element_number) : MacroElement(domain_pt, macro_element_number){};

      /// Default constructor (empty and broken)
      TMacroElement()
      {
         throw OomphLibError("Don't call empty constructor for TMacroElement!",
                             OOMPH_CURRENT_FUNCTION,
                             OOMPH_EXCEPTION_LOCATION);
      }

      /// Broken copy constructor
      TMacroElement(const TMacroElement &dummy)
      {
         BrokenCopy::broken_copy("TMacroElement");
      }

      /// Broken assignment operator
      void operator=(const TMacroElement &)
      {
         BrokenCopy::broken_assign("TMacroElement");
      }

      /// Empty destructor
      virtual ~TMacroElement(){};

      // Stub: not needed for triangular macro elements, calling it is a bug
      void output(const unsigned &t, std::ostream &outfile, const unsigned &nplot)
      {
         throw_runtime_error("Not implemented");
      }

      // Stub: not needed for triangular macro elements, calling it is a bug
      void output_macro_element_boundaries(std::ostream &outfile, const unsigned &nplot)
      {
         throw_runtime_error("Not implemented");
      }

      // Stub: mapping from macro-element to Eulerian coordinates is unused here
      void macro_map(const unsigned &t, const Vector<double> &S, Vector<double> &r)
      {
         throw_runtime_error("Not implemented");
      }

      // Stub: Jacobian of the macro-to-Eulerian map is unused here
      virtual void assemble_macro_to_eulerian_jacobian(const unsigned &t, const Vector<double> &s, DenseMatrix<double> &jacobian)
      {
         throw_runtime_error("Not implemented");
      }

      // Stub: second derivative of the macro-to-Eulerian map is unused here
      virtual void assemble_macro_to_eulerian_jacobian2(const unsigned &t, const Vector<double> &s, DenseMatrix<double> &jacobian2)
      {
         throw_runtime_error("Not implemented");
      }
   };

}
