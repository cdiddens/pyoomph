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



#include "mesh.hpp"
#include "nodes.hpp"
#include "meshtemplate.hpp"
#include "problem.hpp"
#include "elements.hpp"
#include "mesh3d.hpp"

#include "Telements.h"
// #include "unstructured_two_d_mesh_geometry_base.h"

#include "exception.hpp"

namespace pyoomph
{


	// True only if every element in the mesh is a brick (h-refinement via an OcTreeForest is only
	// implemented for pure-brick meshes). Also issues a one-off warning (per mesh) if adaptive
	// refinement was requested but the mesh contains non-brick (e.g. tetrahedral) elements.
	bool TemplatedMeshBase3d::refinement_possible()
	{
		bool allQ = true;
		for (unsigned int i = 0; i < this->nelement(); i++)
		{
			allQ = allQ && (dynamic_cast<oomph::BrickElementBase *>(this->element_pt(i)) != NULL);
		}
		if (allQ)
		{
			return true;
		}
		else
		{
			if (this->max_refinement_level() && !issued_tri_refinement_warning && !this->problem->is_quiet())
			{
				std::cerr << "WARNING: Found a tri or something in the mesh -> cannot be adaptive right now. Requires to implement a good tree for mixed meshes" << std::endl;
				issued_tri_refinement_warning = true;
			}
			return false;
		}
	}

	// Overload used by the generic (oomph-lib) interface: discards the documentation output.
	void TemplatedMeshBase3d::setup_boundary_element_info()
	{
		std::ostringstream oss;
		setup_boundary_element_info(oss);
	}

	// Build Boundary_element_pt / Face_index_at_boundary for a (possibly mixed brick/tet) 3d mesh.
	// For adaptive, pure-brick meshes, delegates to the generic facet-based TemplatedMeshBase
	// implementation (identication_of_boundary_elements_by_facets), which is robust under refinement.
	// Otherwise (non-adaptive or mixed meshes), uses the brick- and tet-specific helpers below.
	void TemplatedMeshBase3d::setup_boundary_element_info(std::ostream &outfile)
	{

		unsigned nbound = nboundary();

		Boundary_element_pt.clear();
		Face_index_at_boundary.clear();
		Boundary_element_pt.resize(nbound);
		Face_index_at_boundary.resize(nbound);

		if (identication_of_boundary_elements_by_facets)
        {
         if (is_adaptation_enabled() &&refinement_possible() )
         {
          identication_of_boundary_elements_by_facets=false; // For adaptive meshes, we find the facets conventionally, but for non-adaptive meshes we can use the facet information from the mesh template which is always accurate, even at mixed corners
         }
        }
        if (identication_of_boundary_elements_by_facets)
        {
         TemplatedMeshBase::setup_boundary_element_info(outfile);
		 Lookup_for_elements_next_boundary_is_setup = true;
         return;
        }

		setup_boundary_element_info_bricks(outfile);
		setup_boundary_element_info_tris(outfile);
		Lookup_for_elements_next_boundary_is_setup = true;
	}

	// Legacy (non-facet-based) boundary-element detection for the brick elements of the mesh: for every
	// brick element, tabulate which of its nodes lie on which boundaries, then use that per-node
	// boundary-membership pattern (via boundary_identifier, counting how many of a face's 4 corner
	// nodes agree on a given +/-x/y/z face indicator) to work out which local face(s) of the element
	// coincide with a boundary. Skips non-brick elements (handled separately by the tet counterpart).
	void TemplatedMeshBase3d::setup_boundary_element_info_bricks(std::ostream &outfile)
	{
		bool doc = false;
		if (outfile)
			doc = true;
		unsigned nbound = nboundary();
		if (doc)
		{
			outfile << "The number of boundaries is " << nbound << "\n";
		}
		Boundary_element_pt.clear();
		Face_index_at_boundary.clear();
		Boundary_element_pt.resize(nbound);
		Face_index_at_boundary.resize(nbound);
		oomph::Vector<oomph::Vector<oomph::FiniteElement *>> vector_of_boundary_element_pt;
		vector_of_boundary_element_pt.resize(nbound);
		// For each (boundary, element) pair, collects one signed direction indicator
		// per corner node of that element which lies on the boundary; a face is only
		// confirmed once all 4 corners belonging to it contribute the same indicator.
		oomph::MapMatrixMixed<unsigned, oomph::FiniteElement *, oomph::Vector<int> *> boundary_identifier;
		oomph::Vector<oomph::Vector<int> *> tmp_vect_pt;

		unsigned nel = nelement();
		for (unsigned e = 0; e < nel; e++)
		{
			oomph::FiniteElement *fe_pt = finite_element_pt(e);
			if (!dynamic_cast<oomph::BrickElementBase *>(fe_pt))
			{
				continue;
			} // Don't do this on tris
			if (doc)
				outfile << "Element: " << e << " " << fe_pt << std::endl;
			unsigned nnode_1d = fe_pt->nnode_1d();
			// Loop over all nodes of the element in the 3d tensor-product node ordering
			for (unsigned i0 = 0; i0 < nnode_1d; i0++)
			{
				for (unsigned i1 = 0; i1 < nnode_1d; i1++)
				{
					for (unsigned i2 = 0; i2 < nnode_1d; i2++)
					{
						unsigned j = i0 + i1 * nnode_1d + i2 * nnode_1d * nnode_1d;
						std::set<unsigned> *boundaries_pt = 0;
						fe_pt->node_pt(j)->get_boundaries_pt(boundaries_pt);
						if (boundaries_pt != 0)
						{
							for (std::set<unsigned>::iterator it = boundaries_pt->begin(); it != boundaries_pt->end(); ++it)
							{
								unsigned boundary_id = *it;
								oomph::Vector<oomph::FiniteElement *>::iterator b_el_it =
									std::find(vector_of_boundary_element_pt[*it].begin(),
											  vector_of_boundary_element_pt[*it].end(),
											  fe_pt);

								if (b_el_it == vector_of_boundary_element_pt[*it].end())
								{
									vector_of_boundary_element_pt[*it].push_back(fe_pt);
								}

								if (boundary_identifier(boundary_id, fe_pt) == 0)
								{
									oomph::Vector<int> *tmp_pt = new oomph::Vector<int>;
									tmp_vect_pt.push_back(tmp_pt);
									boundary_identifier(boundary_id, fe_pt) = tmp_pt;
								}

								// Only corner nodes (vertices of the brick) can identify a whole face;
								// each corner belongs to exactly 3 of the 6 faces, so push one signed
								// indicator per relevant local direction (+/-1 for x, +/-2 for y, +/-3 for z).
								if (((i0 == 0) || (i0 == nnode_1d - 1)) && ((i1 == 0) || (i1 == nnode_1d - 1)) && ((i2 == 0) || (i2 == nnode_1d - 1)))
								{
									(*boundary_identifier(boundary_id, fe_pt)).push_back(1 * (2 * i0 / (nnode_1d - 1) - 1));
									(*boundary_identifier(boundary_id, fe_pt)).push_back(2 * (2 * i1 / (nnode_1d - 1) - 1));
									(*boundary_identifier(boundary_id, fe_pt)).push_back(3 * (2 * i2 / (nnode_1d - 1) - 1));
								}
							}
						}
					}
				}
			}
		}

		// For each element found to touch a boundary, tally how many corner nodes
		// contributed each possible face indicator; an indicator that was contributed
		// by all 4 corners of a face confirms that face lies entirely on the boundary.
		for (unsigned i = 0; i < nbound; i++)
		{
			// Loop over elements on given boundary
			typedef oomph::Vector<oomph::FiniteElement *>::iterator IT;
			for (IT it = vector_of_boundary_element_pt[i].begin();
				 it != vector_of_boundary_element_pt[i].end();
				 it++)
			{
				oomph::FiniteElement *fe_pt = (*it);
				std::map<int, unsigned> count;
				for (int ii = 0; ii < 3; ii++)
				{
					for (int sign = -1; sign < 3; sign += 2)
					{
						count[(ii + 1) * sign] = 0;
					}
				}

				unsigned n_indicators = (*boundary_identifier(i, fe_pt)).size();
				for (unsigned j = 0; j < n_indicators; j++)
				{
					count[(*boundary_identifier(i, fe_pt))[j]]++;
				}

				int indicator = -10;

				// A face has 4 corners, so an indicator seen exactly 4 times means
				// that whole face lies on boundary i; record it as a boundary face.
				for (int ii = 0; ii < 3; ii++)
				{
					for (int sign = -1; sign < 3; sign += 2)
					{
						if (count[(ii + 1) * sign] == 4)
						{
							indicator = (ii + 1) * sign;
							Boundary_element_pt[i].push_back(*it);
							Face_index_at_boundary[i].push_back(indicator);
						}
					}
				}
			}
		}

		// Free the per-(boundary,element) indicator vectors allocated above via
		// boundary_identifier (they were collected in tmp_vect_pt since the map
		// itself does not own them).
		unsigned n = tmp_vect_pt.size();
		for (unsigned i = 0; i < n; i++)
		{
			delete tmp_vect_pt[i];
		}
	}

	// Legacy (non-facet-based) boundary-element detection for the tetrahedral elements of the mesh
	// (despite the name "tris", these are the 4-noded tets of a 3d mesh, analogous to the triangle
	// case in 2d). For each tet, each of its 4 faces is checked by intersecting the boundary-index
	// sets of its 3 corner nodes: if all three nodes share exactly one common boundary, that face is
	// recorded as lying on that boundary (a face shared by more than one boundary triggers a warning,
	// as it indicates a degenerate/too-coarse mesh).
	void TemplatedMeshBase3d::setup_boundary_element_info_tris(std::ostream &outfile)
	{
		unsigned nel = nelement();
		unsigned nbound = nboundary();
		oomph::Vector<oomph::Vector<oomph::FiniteElement *>> vector_of_boundary_element_pt;
		vector_of_boundary_element_pt.resize(nbound);
		// Matrix map for working out the fixed face for elements on boundary
		oomph::MapMatrixMixed<unsigned, oomph::FiniteElement *, int> face_identifier;
		oomph::Vector<std::set<unsigned> *> boundaries_pt(4, 0);

		for (unsigned e = 0; e < nel; e++)
		{
			// Get pointer to element
			oomph::FiniteElement *fe_pt = finite_element_pt(e);
			if (!dynamic_cast<oomph::TElementBase *>(fe_pt))
				continue; // Only on triangles
			// Only include 3D elements! Some meshes contain interface elements too.
			if (fe_pt->dim() == 3)
			{
				for (unsigned i = 0; i < 4; i++)
				{
					fe_pt->node_pt(i)->get_boundaries_pt(boundaries_pt[i]);
				}
				oomph::Vector<std::set<unsigned>> face(4);

				// Face 3 connnects points 0, 1 and 2
				if (boundaries_pt[0] && boundaries_pt[1] && boundaries_pt[2])
				{
					std::set<unsigned> aux;

					std::set_intersection(boundaries_pt[0]->begin(), boundaries_pt[0]->end(),
										  boundaries_pt[1]->begin(), boundaries_pt[1]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  aux, aux.begin()));

					std::set_intersection(aux.begin(), aux.end(),
										  boundaries_pt[2]->begin(), boundaries_pt[2]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  face[3], face[3].begin()));
				}

				if (boundaries_pt[0] && boundaries_pt[1] && boundaries_pt[3])
				{
					std::set<unsigned> aux;

					std::set_intersection(boundaries_pt[0]->begin(), boundaries_pt[0]->end(),
										  boundaries_pt[1]->begin(), boundaries_pt[1]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  aux, aux.begin()));

					std::set_intersection(aux.begin(), aux.end(),
										  boundaries_pt[3]->begin(), boundaries_pt[3]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  face[2], face[2].begin()));
				}

				// Face 1 connects points 0, 2 and 3
				if (boundaries_pt[0] && boundaries_pt[2] && boundaries_pt[3])
				{
					std::set<unsigned> aux;

					std::set_intersection(boundaries_pt[0]->begin(), boundaries_pt[0]->end(),
										  boundaries_pt[2]->begin(), boundaries_pt[2]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  aux, aux.begin()));

					std::set_intersection(aux.begin(), aux.end(),
										  boundaries_pt[3]->begin(), boundaries_pt[3]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  face[1], face[1].begin()));
				}

				// Face 0 connects points 1, 2 and 3
				if (boundaries_pt[1] && boundaries_pt[2] && boundaries_pt[3])
				{
					std::set<unsigned> aux;

					std::set_intersection(boundaries_pt[1]->begin(), boundaries_pt[1]->end(),
										  boundaries_pt[2]->begin(), boundaries_pt[2]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  aux, aux.begin()));

					std::set_intersection(aux.begin(), aux.end(),
										  boundaries_pt[3]->begin(), boundaries_pt[3]->end(),
										  std::insert_iterator<std::set<unsigned>>(
											  face[0], face[0].begin()));
				}

				// We now know whether any faces lay on the boundaries
				for (unsigned i = 0; i < 4; i++)
				{
					// How many boundaries are there
					unsigned count = 0;

					// The number of the boundary
					int boundary = -1;

					// Loop over all the members of the set and add to the count
					// and set the boundary
					for (std::set<unsigned>::iterator it = face[i].begin();
						 it != face[i].end(); ++it)
					{
						++count;
						boundary = *it;
					}

					// If we're on more than one boundary, this is weird, so die
					if (count > 1)
					{
						std::ostringstream error_stream;
						fe_pt->output(error_stream);
						error_stream << "Face " << i << " is on " << count << " boundaries.\n";
						error_stream << "This is rather strange.\n";
						error_stream << "Your mesh may be too coarse or your tet mesh\n";
						error_stream << "may be screwed up. I'm skipping the automated\n";
						error_stream << "setup of the elements next to the boundaries\n";
						error_stream << "lookup schemes.\n";
						oomph::OomphLibWarning(
							error_stream.str(),
							OOMPH_CURRENT_FUNCTION,
							OOMPH_EXCEPTION_LOCATION);
					}

					// If we have a boundary then add this to the appropriate set
					if (boundary >= 0)
					{

						// Does the pointer already exits in the vector
						oomph::Vector<oomph::FiniteElement *>::iterator b_el_it =
							std::find(vector_of_boundary_element_pt[static_cast<unsigned>(boundary)].begin(),
									  vector_of_boundary_element_pt[static_cast<unsigned>(boundary)].end(),
									  fe_pt);

						// Only insert if we have not found it (i.e. got to the end)
						if (b_el_it == vector_of_boundary_element_pt[static_cast<unsigned>(boundary)].end())
						{
							vector_of_boundary_element_pt[static_cast<unsigned>(boundary)].push_back(fe_pt);
						}

						// Also set the fixed face
						face_identifier(static_cast<unsigned>(boundary), fe_pt) = i;
					}
				}

				// Now we set the pointers to the boundary sets to zero
				for (unsigned i = 0; i < 4; i++)
				{
					boundaries_pt[i] = 0;
				}
			}
		}

		// Now copy everything across into permanent arrays
		//-------------------------------------------------

		// Loop over boundaries
		//---------------------
		for (unsigned i = 0; i < nbound; i++)
		{
			// Number of elements on this boundary (currently stored in a set)
			unsigned nel = vector_of_boundary_element_pt[i].size();
			unsigned e_count = Face_index_at_boundary[i].size();
			Face_index_at_boundary[i].resize(e_count + nel);

			typedef oomph::Vector<oomph::FiniteElement *>::iterator IT;
			for (IT it = vector_of_boundary_element_pt[i].begin();
				 it != vector_of_boundary_element_pt[i].end();
				 it++)
			{
				// Recover pointer to element
				oomph::FiniteElement *fe_pt = *it;

				// Add to permanent storage
				Boundary_element_pt[i].push_back(fe_pt);

				Face_index_at_boundary[i][e_count] = face_identifier(i, fe_pt);

				// Increment counter
				e_count++;
			}
		}
	}

}
