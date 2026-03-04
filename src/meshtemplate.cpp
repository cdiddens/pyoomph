/*================================================================================
pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha

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



#include "meshtemplate.hpp"
#include "elements.hpp"
#include "exception.hpp"
#include "codegen.hpp"
#include "ccompiler.hpp"
#include "nodes.hpp"
#include <algorithm>

namespace pyoomph
{

	MeshTemplateFacet::MeshTemplateFacet(std::vector<unsigned> &inds, MeshTemplateCurvedEntity *curved, std::vector<MeshTemplateNode *> *nodes) : nodeinds(inds), curved_entity(curved)
	{
		sorted_inds = inds;
		std::sort(sorted_inds.begin(), sorted_inds.end());
		if (curved)
		{
			parametrics.resize(inds.size(), std::vector<double>(curved->get_parametric_dimension()));
			for (unsigned int i = 0; i < nodeinds.size(); i++)
			{
				std::vector<double> nx(3);
				nx[0] = (*nodes)[nodeinds[i]]->x;
				nx[1] = (*nodes)[nodeinds[i]]->y;
				nx[2] = (*nodes)[nodeinds[i]]->z;
				(*nodes)[nodeinds[i]]->on_curved_facet = true;
				//		std::cout << "BEFORE " << i << "  " << parametrics[i][0] << std::endl;
				curved->position_to_parametric(0, nx, parametrics[i]);

				//	 			std::cout << "BEFORE " << nx[0] << "  " << nx[1] << "  " << nx[2] << std::endl;
				std::vector<double> ps(3);
				curved->parametric_to_position(0, parametrics[i], ps);
				//	 			std::cout << "AFTER " << ps[0] << "  " << ps[1] << "  " << ps[2] << std::endl;
				//		std::cout << "AFTER " << i << "  " << parametrics[i][0] << std::endl;
			}
			curved->apply_periodicity(parametrics);
		}
	}

	MeshTemplateMacroElementBase::MeshTemplateMacroElementBase(MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes) : facets(e->nfacets(), NULL), permutation(e->nfacets(), std::vector<unsigned>()), default_facet_nodes(e->nfacets())
	{
		for (unsigned int i = 0; i < e->nfacets(); i++)
		{
			MeshTemplateFacet *f = e->construct_facet(i);
			for (unsigned int j = 0; j < f->nodeinds.size(); j++)
				default_facet_nodes[i].push_back((*nodes)[f->nodeinds[j]]->oomph_node);
			delete f;
		}
	}

	void MeshTemplateMacroElementBase::set_facet(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation)
	{
		facets[ifacet] = new_facet;
		permutation[ifacet] = find_permutation(ifacet, new_facet, for_orientation);
	}

	MeshTemplateQMacroElement2::MeshTemplateQMacroElement2(MeshTemplateDomain *domain, unsigned index, MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes) : oomph::QMacroElement<2>(domain, index), MeshTemplateMacroElementBase(e, nodes)
	{
	}

	std::vector<unsigned> MeshTemplateQMacroElement2::find_permutation(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation)
	{
		if (new_facet->nodeinds[0] == for_orientation->nodeinds[0])
		{
			return std::vector<unsigned>{0, 1};
		}
		else
		{
			return std::vector<unsigned>{1, 0};
		}
	}

	void MeshTemplateQMacroElement2::macro_element_boundary(const unsigned &t, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f)
	{
		unsigned fi = i_direct - 4;
		double lambda = 0.5 * (s[0] + 1);
		if (!facets[fi] || !facets[fi]->curved_entity)
		{
			for (unsigned int i = 0; i < f.size(); i++)
			{
				f[i] = default_facet_nodes[fi][0]->x(t, i) * (1 - lambda) + default_facet_nodes[fi][1]->x(t, i) * lambda;
			}
		}
		else
		{
			std::vector<double> parametric(1);
			std::vector<double> default_f(2);
			for (unsigned int i = 0; i < 2; i++)
				default_f[i] = default_facet_nodes[fi][0]->x(t, i) * (1 - lambda) + default_facet_nodes[fi][1]->x(t, i) * lambda;
			parametric[0] = (1 - lambda) * facets[fi]->parametrics[permutation[fi][0]][0] + (lambda)*facets[fi]->parametrics[permutation[fi][1]][0];
			std::vector<double> pos(2);
			facets[fi]->curved_entity->parametric_to_position(t, parametric, pos);
			f[0] = pos[0];
			f[1] = pos[1];
		}
	}

	////

	MeshTemplateTMacroElement2::MeshTemplateTMacroElement2(MeshTemplateDomain *domain, unsigned index, MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes) : oomph::TMacroElement<2>(domain, index), MeshTemplateMacroElementBase(e, nodes)
	{
	}

	std::vector<unsigned> MeshTemplateTMacroElement2::find_permutation(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation)
	{
		if (new_facet->nodeinds[0] == for_orientation->nodeinds[0])
		{
			return std::vector<unsigned>{0, 1};
		}
		else
		{
			return std::vector<unsigned>{1, 0};
		}
	}

	void MeshTemplateTMacroElement2::macro_element_boundary(const unsigned &t, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f)
	{
		unsigned fi = i_direct - 4;
		double lambda = 0.5 * (s[0] + 1);
		if (!facets[fi] || !facets[fi]->curved_entity)
		{
			for (unsigned int i = 0; i < f.size(); i++)
			{
				f[i] = default_facet_nodes[fi][0]->x(t, i) * (1 - lambda) + default_facet_nodes[fi][1]->x(t, i) * lambda;
			}
		}
		else
		{
			std::vector<double> parametric(1);
			std::vector<double> default_f(2);
			for (unsigned int i = 0; i < 2; i++)
				default_f[i] = default_facet_nodes[fi][0]->x(t, i) * (1 - lambda) + default_facet_nodes[fi][1]->x(t, i) * lambda;
			parametric[0] = (1 - lambda) * facets[fi]->parametrics[permutation[fi][0]][0] + (lambda)*facets[fi]->parametrics[permutation[fi][1]][0];
			std::vector<double> pos(2);
			facets[fi]->curved_entity->parametric_to_position(t, parametric, pos);
			f[0] = pos[0];
			f[1] = pos[1];
		}
	}

	////

	MeshTemplateQMacroElement3::MeshTemplateQMacroElement3(MeshTemplateDomain *domain, unsigned index, MeshTemplateElement *e, std::vector<MeshTemplateNode *> *nodes) : oomph::QMacroElement<3>(domain, index), MeshTemplateMacroElementBase(e, nodes)
	{
	}

	std::vector<unsigned> MeshTemplateQMacroElement3::find_permutation(const unsigned &ifacet, MeshTemplateFacet *new_facet, MeshTemplateFacet *for_orientation)
	{
		std::vector<unsigned> perm(4);
		for (unsigned int i = 0; i < 4; i++)
			perm[i] = i;

		while (true)
		{
			for (unsigned int i = 0; i < 4; i++)
			{
				if (for_orientation->nodeinds[perm[i]] != new_facet->nodeinds[i])
				{
					break;
				}
				else if (i == 3)
				{
					return perm;
				}
			}
			if (!std::next_permutation(perm.begin(), perm.end()))
			{
				std::ostringstream oss;
				oss << std::endl
					<< "  NF :" << new_facet->nodeinds[0] << "  " << new_facet->nodeinds[1] << "  " << new_facet->nodeinds[2] << "  " << new_facet->nodeinds[3];
				oss << std::endl
					<< "  FO :" << for_orientation->nodeinds[0] << "  " << for_orientation->nodeinds[1] << "  " << for_orientation->nodeinds[2] << "  " << for_orientation->nodeinds[3] << std::endl;
				throw_runtime_error("Strange permutation: " + oss.str());
			}
		}
	}

	void MeshTemplateQMacroElement3::macro_element_boundary(const unsigned &t, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f)
	{

		unsigned fi = i_direct - 20;
		std::cout << "STARTING FACE I " << fi << std::endl;
		double lambda0 = 0.5 * (s[0] + 1);
		double lambda1 = 0.5 * (s[1] + 1);

		if (!facets[fi] || !facets[fi]->curved_entity)
		{
			for (unsigned int i = 0; i < f.size(); i++)
			{
				f[i] = (default_facet_nodes[fi][0]->x(t, i) * (1 - lambda0) + default_facet_nodes[fi][1]->x(t, i) * lambda0) * (1 - lambda1) + (default_facet_nodes[fi][2]->x(t, i) * (1 - lambda0) + default_facet_nodes[fi][3]->x(t, i) * lambda0) * lambda1;
			}
		}
		else
		{
			std::vector<double> parametric(2);
			std::vector<double> default_f(3);
			for (unsigned int i = 0; i < 3; i++)
			{
				default_f[i] = (default_facet_nodes[fi][0]->x(t, i) * (1 - lambda0) + default_facet_nodes[fi][1]->x(t, i) * lambda0) * (1 - lambda1) + (default_facet_nodes[fi][2]->x(t, i) * (1 - lambda0) + default_facet_nodes[fi][3]->x(t, i) * lambda0) * lambda1;
			}

			// return;
			std::vector<unsigned> perm = permutation[fi];
			/*for (unsigned int i=0;i<4;i++) perm[i]=i;
			std::vector<unsigned> iperm(4);
			for (unsigned int i=0;i<4;i++) iperm[perm[i]]=i;
			std::cout << "PERM IS " << " :  " << perm[0] << " " << perm[1] << " " << perm[2] << "  " << perm[3] << std::endl;
		   */
			/*   for (unsigned int i=0;i<3;i++)
			   {
				  f[i]=((1-lambda0)*facets[fi]->parametrics[iperm[0]][i]+(lambda0)*facets[fi]->parametrics[iperm[1]][i])*(1-lambda1)+((1-lambda0)*facets[fi]->parametrics[iperm[2]][i]+(lambda0)*facets[fi]->parametrics[iperm[3]][i])*lambda1;
			   }*/

			for (unsigned int i = 0; i < 2; i++)
			{
				parametric[i] = ((1 - lambda0) * facets[fi]->parametrics[perm[0]][i] + (lambda0)*facets[fi]->parametrics[perm[1]][i]) * (1 - lambda1) + ((1 - lambda0) * facets[fi]->parametrics[perm[2]][i] + (lambda0)*facets[fi]->parametrics[perm[3]][i]) * lambda1;
			}
			std::vector<double> pos(3);
			facets[fi]->curved_entity->parametric_to_position(t, parametric, pos);
			std::vector<double> test(3);
			facets[fi]->curved_entity->position_to_parametric(t, pos, test);
			std::cout << "COMPARING PARAMS " << parametric[0] << "  " << parametric[1] << "  vs " << test[0] << "  " << test[1] << "  with pos " << pos[0] << "  " << pos[1] << "  " << pos[2] << std::endl;
			f[0] = pos[0];
			f[1] = pos[1];
			f[2] = pos[2];
			/*
			std::cout << "SETTING F TO " << f[0] << "  " << f[1] << "  " << f[2] << std::endl;
			std::cout << "WHILE DEFAULT_F is " << default_f[0] << "  " << default_f[1] << "  " << default_f[2] << std::endl;
			std::vector<double> defpar(2);
			facets[fi]->curved_entity->position_to_parametric(t,default_f,defpar);
			std::cout << "PARAMETERICS " << parametric[0] << "  " << parametric[1] << "   vs def param " << defpar[0] << "  " << defpar[1] << std::endl;
			std::cout << "SUBPARAMETERICS  " << lambda0 << "  " << lambda1 << std::endl;
			//double rad=f[0]*f[0]+f[1]*f[1]+(f[2]+0.3582272)*(f[2]+0.3582272);
			//std::cout << "RAD " << rad << std::endl;
			for (unsigned int i=0;i<4;i++)
			{
			 std::cout << facets[fi]->parametrics[i][0] << "  " << facets[fi]->parametrics[i][1] << std::endl;
			}
			*/
		}
	}

	MeshTemplateDomain::MeshTemplateDomain()
	{
	}

	void MeshTemplateDomain::macro_element_boundary(const unsigned &t, const unsigned &i_macro, const unsigned &i_direct, const oomph::Vector<double> &s, oomph::Vector<double> &f)
	{
		// TODO: Remove the if via virtual base
		if (dynamic_cast<MeshTemplateQMacroElement2 *>(Macro_element_pt[i_macro]))
		{
			MeshTemplateQMacroElement2 *macro = dynamic_cast<MeshTemplateQMacroElement2 *>(Macro_element_pt[i_macro]);
			macro->macro_element_boundary(t, i_direct, s, f);
		}
		else if (dynamic_cast<MeshTemplateQMacroElement3 *>(Macro_element_pt[i_macro]))
		{
			MeshTemplateQMacroElement3 *macro = dynamic_cast<MeshTemplateQMacroElement3 *>(Macro_element_pt[i_macro]);
			macro->macro_element_boundary(t, i_direct, s, f);
		}
	}

	void MeshTemplateElement::link_nodes_with_domain(MeshTemplateElementCollection *dom)
	{
		auto *mt = dom->get_template();
		auto &nodes = mt->get_nodes();
		for (auto &nind : node_indices)
		{
			nodes[nind]->part_of_domain.insert(dom);
		}
	}


	MeshTemplateElementPoint::MeshTemplateElementPoint(const nodeindex_t &n1) : MeshTemplateElement(0)
  	{
		node_indices.push_back(n1);
  	}
	/////////////////
	MeshTemplateElementLineC1::MeshTemplateElementLineC1(const nodeindex_t &n1, const nodeindex_t &n2) : MeshTemplateElement(1)
	{
		node_indices.reserve(2);
		node_indices.push_back(n1);
		node_indices.push_back(n2);
	}

	MeshTemplateElement *MeshTemplateElementLineC1::convert_for_C2_space(MeshTemplate *templ)
	{
		nodeindex_t n3 = templ->add_intermediate_node_unique(node_indices[0], node_indices[1]);
		return new MeshTemplateElementLineC2(node_indices[0], n3, node_indices[1]);
	}

	MeshTemplateFacet *MeshTemplateElementLineC1::construct_facet(unsigned i)
	{
		unsigned ni1;
		if (i == 0)
		{
			ni1 = 0;
		}
		else if (i == 1)
		{
			ni1 = 1;
		}
		else
			return NULL;
		ni1 = node_indices[ni1];
		std::vector<unsigned> inds = {ni1};
		return new MeshTemplateFacet(inds, NULL, NULL);
	}

	/////////////////

	MeshTemplateElementLineC2::MeshTemplateElementLineC2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3) : MeshTemplateElement(2)
	{
		node_indices.reserve(3);
		node_indices.push_back(n1);
		node_indices.push_back(n2);
		node_indices.push_back(n3);
	}

	MeshTemplateFacet *MeshTemplateElementLineC2::construct_facet(unsigned i)
	{
		unsigned ni1;
		if (i == 0)
		{
			ni1 = 0;
		}
		else if (i == 1)
		{
			ni1 = 2;
		}
		else
			return NULL;
		ni1 = node_indices[ni1];
		std::vector<unsigned> inds = {ni1};
		return new MeshTemplateFacet(inds, NULL, NULL);
	}

	/////////////////////////

	MeshTemplateElementQuadC1::MeshTemplateElementQuadC1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4) : MeshTemplateElement(6)
	{
		node_indices.reserve(4);
		node_indices.push_back(n1);
		node_indices.push_back(n2);
		node_indices.push_back(n3);
		node_indices.push_back(n4);
	}

	MeshTemplateFacet *MeshTemplateElementQuadC1::construct_facet(unsigned i)
	{
		unsigned ni1, ni2;
		if (i == 0)
		{
			ni1 = 2;
			ni2 = 3;
		} // NORTH
		else if (i == 1)
		{
			ni1 = 1;
			ni2 = 3;
		} // EAST
		else if (i == 2)
		{
			ni1 = 0;
			ni2 = 1;
		} // SOUTH
		else if (i == 3)
		{
			ni1 = 0;
			ni2 = 2;
		} // WEST
		else
			return NULL;
		ni1 = node_indices[ni1];
		ni2 = node_indices[ni2];
		std::vector<unsigned> inds = {ni1, ni2};
		return new MeshTemplateFacet(inds, NULL, NULL);
	}

	MeshTemplateElement *MeshTemplateElementQuadC1::convert_for_C2_space(MeshTemplate *templ)
	{
		nodeindex_t n1 = templ->add_intermediate_node_unique(node_indices[0], node_indices[1]);
		nodeindex_t n3 = templ->add_intermediate_node_unique(node_indices[0], node_indices[2]);
		nodeindex_t n4 = templ->add_intermediate_node_unique(node_indices[0], node_indices[1], node_indices[2], node_indices[3], false);
		nodeindex_t n5 = templ->add_intermediate_node_unique(node_indices[1], node_indices[3]);
		nodeindex_t n7 = templ->add_intermediate_node_unique(node_indices[2], node_indices[3]);
		return new MeshTemplateElementQuadC2(node_indices[0], n1, node_indices[1], n3, n4, n5, node_indices[2], n7, node_indices[3]);
	}

	/////////////////////////
	MeshTemplateElementQuadC2::MeshTemplateElementQuadC2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4,
														 const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7, const nodeindex_t &n8, const nodeindex_t &n9) : MeshTemplateElement(8)
	{
		node_indices.reserve(9);
		node_indices.push_back(n1);
		node_indices.push_back(n2);
		node_indices.push_back(n3);
		node_indices.push_back(n4);
		node_indices.push_back(n5);
		node_indices.push_back(n6);
		node_indices.push_back(n7);
		node_indices.push_back(n8);
		node_indices.push_back(n9);
	}

	MeshTemplateFacet *MeshTemplateElementQuadC2::construct_facet(unsigned i)
	{
		unsigned ni1, ni2;
		if (i == 0)
		{
			ni1 = 6;
			ni2 = 8;
		} // NORTH
		else if (i == 1)
		{
			ni1 = 2;
			ni2 = 8;
		} // EAST
		else if (i == 2)
		{
			ni1 = 0;
			ni2 = 2;
		} // SOUTH
		else if (i == 3)
		{
			ni1 = 0;
			ni2 = 6;
		} // WEST
		else
			return NULL;
		ni1 = node_indices[ni1];
		ni2 = node_indices[ni2];
		std::vector<unsigned> inds = {ni1, ni2};
		return new MeshTemplateFacet(inds, NULL, NULL);
	}

	/////////////////////////////////

	MeshTemplateElementTriC1::MeshTemplateElementTriC1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3) : MeshTemplateElement(3)
	{
		node_indices.reserve(3);
		node_indices.push_back(n1);
		node_indices.push_back(n2);
		node_indices.push_back(n3);
	}

	MeshTemplateFacet *MeshTemplateElementTriC1::construct_facet(unsigned i)
	{
      unsigned ni1, ni2;
		if (i == 0)
		{
			ni1 = 0;
			ni2 = 1;
		}
		else if (i == 1)
		{
			ni1 = 1;
			ni2 = 2;
		}
		else if (i == 2)
		{
			ni1 = 2;
			ni2 = 0;
		}
		else
			return NULL;
		ni1 = node_indices[ni1];
		ni2 = node_indices[ni2];
		std::vector<unsigned> inds = {ni1, ni2};
		return new MeshTemplateFacet(inds, NULL, NULL);
	}

	MeshTemplateElement *MeshTemplateElementTriC1::convert_for_C2_space(MeshTemplate *templ)
	{
		nodeindex_t n3 = templ->add_intermediate_node_unique(node_indices[0], node_indices[1]);
		nodeindex_t n4 = templ->add_intermediate_node_unique(node_indices[1], node_indices[2]);
		nodeindex_t n5 = templ->add_intermediate_node_unique(node_indices[0], node_indices[2]);
		return new MeshTemplateElementTriC2(node_indices[0], node_indices[1], node_indices[2], n3, n4, n5);
	}
	
	MeshTemplateElement *MeshTemplateElementTriC1::convert_for_C1TB_space(MeshTemplate *templ)
	{
		nodeindex_t n3 = templ->add_intermediate_node_unique(node_indices[0], node_indices[1],node_indices[2],false);
		return new MeshTemplateElementTriC1TB(node_indices[0], node_indices[1], node_indices[2], n3);
	}	

	/////////////////////////////////

	MeshTemplateElementTriC2::MeshTemplateElementTriC2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5, const nodeindex_t &n6) : MeshTemplateElement(9)
	{
		node_indices.reserve(6);
		node_indices.push_back(n1);
		node_indices.push_back(n2);
		node_indices.push_back(n3);
		node_indices.push_back(n4);
		node_indices.push_back(n5);
		node_indices.push_back(n6);
	}

	MeshTemplateFacet *MeshTemplateElementTriC2::construct_facet(unsigned i)
	{
		unsigned ni1, ni2;
		if (i == 0)
		{
			ni1 = 0;
			ni2 = 1;
		}
		else if (i == 1)
		{
			ni1 = 1;
			ni2 = 2;
		}
		else if (i == 2)
		{
			ni1 = 2;
			ni2 = 0;
		}
		else
			return NULL;
		ni1 = node_indices[ni1];
		ni2 = node_indices[ni2];
		std::vector<unsigned> inds = {ni1, ni2};
		return new MeshTemplateFacet(inds, NULL, NULL);
	}

	MeshTemplateElement *MeshTemplateElementTriC2::convert_for_C2TB_space(MeshTemplate *templ)
	{
		nodeindex_t n7 = templ->add_intermediate_node_unique(node_indices[0], node_indices[1], node_indices[2], false);
		return new MeshTemplateElementTriC2TB(node_indices[0], node_indices[1], node_indices[2], node_indices[3], node_indices[4], node_indices[5], n7);
	}

	MeshTemplateElementTriC1TB::MeshTemplateElementTriC1TB(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4) : MeshTemplateElementTriC1(n1, n2, n3)
	{
		node_indices.push_back(n4);
	}

	MeshTemplateElementTriC2TB::MeshTemplateElementTriC2TB(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7) : MeshTemplateElementTriC2(n1, n2, n3, n4, n5, n6)
	{
		node_indices.push_back(n7);
	}

	//////////////////////////////////

	MeshTemplateElementBrickC1::MeshTemplateElementBrickC1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4,
														   const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7, const nodeindex_t &n8) : MeshTemplateElement(11)
	{
		node_indices.reserve(8);
		node_indices.push_back(n1);
		node_indices.push_back(n2);
		node_indices.push_back(n3);
		node_indices.push_back(n4);
		node_indices.push_back(n5);
		node_indices.push_back(n6);
		node_indices.push_back(n7);
		node_indices.push_back(n8);
	}

	MeshTemplateElement *MeshTemplateElementBrickC1::convert_for_C2_space(MeshTemplate *templ)
	{
		std::vector<nodeindex_t> ninds(27);
		// Bottom plate
		ninds[0] = node_indices[0];
		ninds[1] = templ->add_intermediate_node_unique(node_indices[0], node_indices[1]);
		ninds[2] = node_indices[1];
		ninds[3] = templ->add_intermediate_node_unique(node_indices[0], node_indices[2]);
		ninds[4] = templ->add_intermediate_node_unique(node_indices[0], node_indices[1], node_indices[2], node_indices[3], true);
		ninds[5] = templ->add_intermediate_node_unique(node_indices[1], node_indices[3]);
		ninds[6] = node_indices[2];
		ninds[7] = templ->add_intermediate_node_unique(node_indices[2], node_indices[3]);
		ninds[8] = node_indices[3];

		// Intermediate layer

		ninds[9] = templ->add_intermediate_node_unique(node_indices[0], node_indices[4]);
		// ninds[10]
		ninds[11] = templ->add_intermediate_node_unique(node_indices[1], node_indices[5]);
		// ninds[12]
		// ninds[13]
		// ninds[14]
		ninds[15] = templ->add_intermediate_node_unique(node_indices[2], node_indices[6]);
		// ninds[16]
		ninds[17] = templ->add_intermediate_node_unique(node_indices[3], node_indices[7]);

		// Top plate
		ninds[18] = node_indices[4];
		ninds[19] = templ->add_intermediate_node_unique(node_indices[4], node_indices[5]);
		ninds[20] = node_indices[5];
		ninds[21] = templ->add_intermediate_node_unique(node_indices[4], node_indices[6]);
		ninds[22] = templ->add_intermediate_node_unique(node_indices[4], node_indices[5], node_indices[6], node_indices[7], true);
		ninds[23] = templ->add_intermediate_node_unique(node_indices[5], node_indices[7]);
		ninds[24] = node_indices[6];
		ninds[25] = templ->add_intermediate_node_unique(node_indices[6], node_indices[7]);
		ninds[26] = node_indices[7];

		// Missing from the intermediate layer
		ninds[10] = templ->add_intermediate_node_unique(ninds[9], ninds[11]);
		ninds[12] = templ->add_intermediate_node_unique(ninds[9], ninds[15]);
		ninds[14] = templ->add_intermediate_node_unique(ninds[11], ninds[17]);
		ninds[16] = templ->add_intermediate_node_unique(ninds[15], ninds[17]);
		ninds[13] = templ->add_intermediate_node_unique(node_indices[0], node_indices[3], node_indices[5], node_indices[6], false);
		/*
		 for (unsigned int j=0;j<27;j++)
		 {
		  unsigned i=ninds[j];
		  std::cout << " D " << j << "  " << i << "   coord "  << std::flush;
		  std::cout << templ->get_nodes()[i]->x << "\t" << templ->get_nodes()[i]->y << "\t" <<templ->get_nodes()[i]->z << std::endl;
		 }


		 for (unsigned int i=0;i<27;i++)
		 {
		  std::cout << " C " << i << "  " << templ->get_nodes()[i]->x << "\t" << templ->get_nodes()[i]->y << "\t" <<templ->get_nodes()[i]->z << std::endl;
		 }



		 std::cout << "NODE 13 is avg of " << std::endl;
		 for (int j : {10,12,14,16})
		 {
		  unsigned i=ninds[j];
		  std::cout << " AV " << j << "  " << templ->get_nodes()[i]->x << "\t" << templ->get_nodes()[i]->y << "\t" <<templ->get_nodes()[i]->z << std::endl;
		 }
		*/
		return new MeshTemplateElementBrickC2(ninds);
	}

	MeshTemplateFacet *MeshTemplateElementBrickC1::construct_facet(unsigned i)
	{
		unsigned ni1, ni2, ni3, ni4;
		if (i == 0)
		{
			ni1 = 0;
			ni2 = 2;
			ni3 = 4;
			ni4 = 6;
		} // LEFT
		else if (i == 1)
		{
			ni1 = 1;
			ni2 = 3;
			ni3 = 5;
			ni4 = 7;
		} // RIGHT
		else if (i == 2)
		{
			ni1 = 0;
			ni2 = 1;
			ni3 = 4;
			ni4 = 5;
		} // DOWN
		else if (i == 3)
		{
			ni1 = 2;
			ni2 = 3;
			ni3 = 6;
			ni4 = 7;
		} // UP
		else if (i == 4)
		{
			ni1 = 0;
			ni2 = 1;
			ni3 = 2;
			ni4 = 3;
		} // BACK
		else if (i == 5)
		{
			ni1 = 4;
			ni2 = 5;
			ni3 = 6;
			ni4 = 7;
		} // FRONT
		else
			return NULL;
		ni1 = node_indices[ni1];
		ni2 = node_indices[ni2];
		ni3 = node_indices[ni3];
		ni4 = node_indices[ni4];
		std::vector<unsigned> inds = {ni1, ni2, ni3, ni4};
		// std::sort(inds.begin(),inds.end());
		return new MeshTemplateFacet(inds, NULL, NULL);
	}

	/////////////////////////////////

	MeshTemplateFacet *MeshTemplateElementBrickC2::construct_facet(unsigned i)
	{
		unsigned ni1, ni2, ni3, ni4;
		if (i == 0)
		{
			ni1 = 0;
			ni2 = 6;
			ni3 = 18;
			ni4 = 24;
		} // LEFT
		else if (i == 1)
		{
			ni1 = 2;
			ni2 = 8;
			ni3 = 20;
			ni4 = 26;
		} // RIGHT
		else if (i == 2)
		{
			ni1 = 0;
			ni2 = 2;
			ni3 = 18;
			ni4 = 20;
		} // DOWN
		else if (i == 3)
		{
			ni1 = 6;
			ni2 = 8;
			ni3 = 24;
			ni4 = 26;
		} // UP
		else if (i == 4)
		{
			ni1 = 0;
			ni2 = 2;
			ni3 = 6;
			ni4 = 8;
		} // BACK
		else if (i == 5)
		{
			ni1 = 18;
			ni2 = 20;
			ni3 = 24;
			ni4 = 26;
		} // FRONT
		else
			return NULL;
		ni1 = node_indices[ni1];
		ni2 = node_indices[ni2];
		ni3 = node_indices[ni3];
		ni4 = node_indices[ni4];
		std::vector<unsigned> inds = {ni1, ni2, ni3, ni4};
		// std::sort(inds.begin(),inds.end());
		return new MeshTemplateFacet(inds, NULL, NULL);
	}

	MeshTemplateElementBrickC2::MeshTemplateElementBrickC2(std::vector<nodeindex_t> ninds) : MeshTemplateElement(14)
	{
		if (ninds.size() != 27)
		{
			throw_runtime_error("Need 27 vertices for a brick element with C2 space");
		}
		node_indices = ninds;
	}

	/////////////////////////////////

	MeshTemplateElementTetraC1::MeshTemplateElementTetraC1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4) : MeshTemplateElement(4)
	{
		node_indices.resize(4);
		node_indices[0] = n1;
		node_indices[1] = n2;
		node_indices[2] = n3;
		node_indices[3] = n4;
	}

	MeshTemplateElement *MeshTemplateElementTetraC1::convert_for_C2_space(MeshTemplate *templ)
	{
		std::vector<nodeindex_t> ninds(10);
		ninds[0] = node_indices[0];
		ninds[1] = node_indices[1];
		ninds[2] = node_indices[2];
		ninds[3] = node_indices[3];

		ninds[4] = templ->add_intermediate_node_unique(node_indices[0], node_indices[1]);
		ninds[5] = templ->add_intermediate_node_unique(node_indices[0], node_indices[2]);
		ninds[6] = templ->add_intermediate_node_unique(node_indices[0], node_indices[3]);
		ninds[7] = templ->add_intermediate_node_unique(node_indices[1], node_indices[2]);
		ninds[8] = templ->add_intermediate_node_unique(node_indices[2], node_indices[3]);
		ninds[9] = templ->add_intermediate_node_unique(node_indices[1], node_indices[3]);

		return new MeshTemplateElementTetraC2(ninds);
	}

	MeshTemplateFacet *MeshTemplateElementTetraC1::construct_facet(unsigned i)
	{
//oomph ordering : {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {1, 2, 0}
		unsigned ni1, ni2, ni3;
		if (i == 0)
		{
			ni1 = 1; ni2 = 2; ni3 = 3;
		}
		else if (i==1)
		{
			ni1 = 0; ni2 = 2; ni3 = 3;
		}
		else if (i==2)
		{
			ni1 = 0; ni2 = 1; ni3 = 3;
		}
		else if (i==3)
		{
			ni1 = 1; ni2 = 2; ni3 = 0;
		}
		else
			return NULL;
		ni1 = node_indices[ni1];
		ni2 = node_indices[ni2];
		ni3 = node_indices[ni3];
		std::vector<unsigned> inds = {ni1, ni2, ni3};
		// std::sort(inds.begin(),inds.end());
		return new MeshTemplateFacet(inds, NULL, NULL);
	}

	/////////////////////////////////

	MeshTemplateElementTetraC2::MeshTemplateElementTetraC2(std::vector<nodeindex_t> ninds) : MeshTemplateElement(10)
	{
		if (ninds.size() != 10)
			throw_runtime_error("Need exactly 10 nodes");
		node_indices = ninds;
	}

	MeshTemplateFacet *MeshTemplateElementTetraC2::construct_facet(unsigned i)
	{
//oomph ordering : {1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {1, 2, 0}
		unsigned ni1, ni2, ni3;
		if (i == 0)
		{
			ni1 = 1; ni2 = 2; ni3 = 3;
		}
		else if (i==1)
		{
			ni1 = 0; ni2 = 2; ni3 = 3;
		}
		else if (i==2)
		{
			ni1 = 0; ni2 = 1; ni3 = 3;
		}
		else if (i==3)
		{
			ni1 = 1; ni2 = 2; ni3 = 0;
		}
		else
			return NULL;
		ni1 = node_indices[ni1];
		ni2 = node_indices[ni2];
		ni3 = node_indices[ni3];
		std::vector<unsigned> inds = {ni1, ni2, ni3};
		// std::sort(inds.begin(),inds.end());
		return new MeshTemplateFacet(inds, NULL, NULL);
	}

	MeshTemplateElement *MeshTemplateElementTetraC2::convert_for_C2TB_space(MeshTemplate *templ)
	{

		std::vector<nodeindex_t> nnodes = node_indices;
		nnodes.push_back(templ->add_intermediate_node_unique(node_indices[0], node_indices[1], node_indices[3], true));
		nnodes.push_back(templ->add_intermediate_node_unique(node_indices[0], node_indices[1], node_indices[2], true));
		nnodes.push_back(templ->add_intermediate_node_unique(node_indices[0], node_indices[2], node_indices[3], true));
		nnodes.push_back(templ->add_intermediate_node_unique(node_indices[1], node_indices[2], node_indices[3], true));
		nnodes.push_back(templ->add_intermediate_node_unique(node_indices[0], node_indices[1], node_indices[2], node_indices[3], false));

		return new MeshTemplateElementTetraC2TB(nnodes);
	}

	/////////////////////////////////

	MeshTemplateElementTetraC2TB::MeshTemplateElementTetraC2TB(std::vector<nodeindex_t> ninds) : MeshTemplateElementTetraC2(std::vector<nodeindex_t>(ninds.begin(), ninds.begin() + 10))
	{
		if (ninds.size() != 15)
			throw_runtime_error("Need exactly 15 nodes");
		for (unsigned int i = 10; i < ninds.size(); i++)
		{
			node_indices.push_back(ninds[i]);
		}
	}

	/////////////////////////////////

	std::vector<double> MeshTemplateElementCollection::get_reference_position_for_IC_and_DBC(std::set<unsigned int> boundindices)
	{
		const std::vector<MeshTemplateNode *> &nodes = mesh_template->get_nodes();
		std::vector<double> res;
		for (unsigned int ie = 0; ie < elements.size(); ie++)
		{
			auto *e = elements[ie];
			//bool hasboundnodes = false;
			for (auto ni : e->get_node_indices())
			{
				bool all_in = true;
				for (unsigned int b : boundindices)
				{
					if (!nodes[ni]->on_boundaries.count(b))
					{
						all_in = false;
						break;
					}
				}
				if (all_in)
				{
					res = {nodes[ni]->x, nodes[ni]->y, nodes[ni]->z};
					break;
				}
			}
			if (res.size())
				break;
		}
		if (res.empty())
			res = {0, 0, 0};
		return res;
	}

	std::vector<std::string> MeshTemplateElementCollection::get_adjacent_boundary_names()
	{
		std::set<unsigned int> binds;
		const std::vector<MeshTemplateNode *> &nodes = mesh_template->get_nodes();
		for (unsigned int ie = 0; ie < elements.size(); ie++)
		{
			auto *e = elements[ie];
			bool hasboundnodes = false;
			for (auto ni : e->get_node_indices())
			{
				if (!nodes[ni]->on_boundaries.empty())
				{
					hasboundnodes = true;
					break;
				}
			}
			if (hasboundnodes)
			{
				std::vector<MeshTemplateFacet *> facets;
				for (unsigned int iface = 0; iface < e->nfacets(); iface++)
				{
					MeshTemplateFacet *facet = e->construct_facet(iface);
					if (facet)
					{

						std::set<unsigned int> fbounds = nodes[facet->nodeinds[0]]->on_boundaries;
						for (unsigned int fni = 0; fni < facet->nodeinds.size(); fni++)
						{
							std::set<unsigned int> the_intersection; // Destination of intersect
							std::set_intersection(fbounds.begin(), fbounds.end(), nodes[facet->nodeinds[fni]]->on_boundaries.begin(), nodes[facet->nodeinds[fni]]->on_boundaries.end(), std::inserter(the_intersection, the_intersection.end()));
							fbounds.swap(the_intersection);
						}

						binds.insert(fbounds.begin(), fbounds.end());

						delete facet;
					}
				}
			}
		}
		std::vector<std::string> res;
		for (auto b : binds)
		{
			res.push_back(mesh_template->get_boundary_names()[b]);
		}
		return res;
	}

	int MeshTemplateElementCollection::nodal_dimension()
	{
		if (Nodal_dimension < 0 && (!elements.empty()))
		{
			Nodal_dimension = elements[0]->nodal_dimension();
		}
		return Nodal_dimension;
	}

	int MeshTemplateElementCollection::lagrangian_dimension()
	{
		if (Lagr_dimension < 0 && (!elements.empty()))
		{
			Lagr_dimension = elements[0]->nodal_dimension();
		}
		return Lagr_dimension;
	}

	MeshTemplateElementPoint *MeshTemplateElementCollection::add_point_element(const nodeindex_t &n1)
	{
		if (dim == -1)
		{
			dim = 0;
		}
		else if (dim != 0)
			throw_runtime_error("Tried to add a 0d point element to a Mesh template which has already elements of dimension " + std::to_string(mesh_template->dim));
		MeshTemplateElementPoint *res = new MeshTemplateElementPoint(n1);
		elements.push_back(res);
		res->link_nodes_with_domain(this);
		return res;
	}

	MeshTemplateElementLineC1 *MeshTemplateElementCollection::add_line_1d_C1(const nodeindex_t &n1, const nodeindex_t &n2)
	{
		if (dim == -1)
		{
			dim = 1;
		}
		else if (dim != 1)
			throw_runtime_error("Tried to add a 1d element to a Mesh template which has already elements of dimension " + std::to_string(mesh_template->dim));
		MeshTemplateElementLineC1 *res = new MeshTemplateElementLineC1(n1, n2);
		elements.push_back(res);
		res->link_nodes_with_domain(this);
		return res;
	}

	MeshTemplateElementLineC2 *MeshTemplateElementCollection::add_line_1d_C2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3)
	{
		if (dim == -1)
		{
			dim = 1;
		}
		else if (dim != 1)
			throw_runtime_error("Tried to add a 1d element to a Mesh template which has already elements of dimension " + std::to_string(mesh_template->dim));
		MeshTemplateElementLineC2 *res = new MeshTemplateElementLineC2(n1, n2, n3);
		elements.push_back(res);
		res->link_nodes_with_domain(this);
		return res;
	}

	MeshTemplateElementQuadC1 *MeshTemplateElementCollection::add_quad_2d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4)
	{
		if (dim == -1)
		{
			dim = 2;
		}
		else if (dim != 2)
			throw_runtime_error("Tried to add a 2d element to a Mesh template which has already elements of dimension " + std::to_string(mesh_template->dim));
		MeshTemplateElementQuadC1 *res = new MeshTemplateElementQuadC1(n1, n2, n3, n4);
		elements.push_back(res);
		res->link_nodes_with_domain(this);
		return res;
	}

	MeshTemplateElementQuadC2 *MeshTemplateElementCollection::add_quad_2d_C2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4,
																			 const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7, const nodeindex_t &n8, const nodeindex_t &n9)
	{
		if (dim == -1)
		{
			dim = 2;
		}
		else if (dim != 2)
			throw_runtime_error("Tried to add a 2d element to a Mesh template which has already elements of dimension " + std::to_string(mesh_template->dim));
		MeshTemplateElementQuadC2 *res = new MeshTemplateElementQuadC2(n1, n2, n3, n4, n5, n6, n7, n8, n9);
		elements.push_back(res);
		res->link_nodes_with_domain(this);
		return res;
	}

	MeshTemplateElementTriC1 *MeshTemplateElementCollection::add_tri_2d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3)
	{
		if (dim == -1)
		{
			dim = 2;
		}
		else if (dim != 2)
			throw_runtime_error("Tried to add a 2d element to a Mesh template which has already elements of dimension " + std::to_string(mesh_template->dim));
		MeshTemplateElementTriC1 *res = new MeshTemplateElementTriC1(n1, n2, n3);
		elements.push_back(res);
		res->link_nodes_with_domain(this);
		return res;
	}

   //Scott-Vogelius splitting
	std::vector<MeshTemplateElementTriC1*> MeshTemplateElementCollection::add_SV_tri_2d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3)
	{
	 std::vector<double> p1=mesh_template->get_node_position(n1);	 
	 std::vector<double> p2=mesh_template->get_node_position(n2);	 
	 std::vector<double> p3=mesh_template->get_node_position(n3);	 	 	 
	 std::vector<double> pbc(3,0.0);
	 for (unsigned int i=0;i<p1.size();i++) pbc[i]=(p1[i]+p2[i]+p3[i])/3.0;
	 nodeindex_t bc=mesh_template->add_node_unique(pbc[0],pbc[1],pbc[2]);
    return {add_tri_2d_C1(n1,n2,bc),add_tri_2d_C1(n2,n3,bc),add_tri_2d_C1(n3,n1,bc)};
	}

	MeshTemplateElementTriC2 *MeshTemplateElementCollection::add_tri_2d_C2(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, const nodeindex_t &n5, const nodeindex_t &n6)
	{
		if (dim == -1)
		{
			dim = 2;
		}
		else if (dim != 2)
			throw_runtime_error("Tried to add a 2d element to a Mesh template which has already elements of dimension " + std::to_string(mesh_template->dim));
		MeshTemplateElementTriC2 *res = new MeshTemplateElementTriC2(n1, n2, n3, n4, n5, n6);
		elements.push_back(res);
		res->link_nodes_with_domain(this);
		return res;
	}

	MeshTemplateElementBrickC1 *MeshTemplateElementCollection::add_brick_3d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4,
																			   const nodeindex_t &n5, const nodeindex_t &n6, const nodeindex_t &n7, const nodeindex_t &n8)
	{
		if (dim == -1)
		{
			dim = 3;
		}
		else if (dim != 3)
			throw_runtime_error("Tried to add a 3d element to a Mesh template which has already elements of dimension " + std::to_string(mesh_template->dim));
		MeshTemplateElementBrickC1 *res = new MeshTemplateElementBrickC1(n1, n2, n3, n4, n5, n6, n7, n8);
		elements.push_back(res);
		res->link_nodes_with_domain(this);
		return res;
	}

	MeshTemplateElementBrickC2 *MeshTemplateElementCollection::add_brick_3d_C2(const std::vector<nodeindex_t> &inds)
	{
		if (dim == -1)
		{
			dim = 3;
		}
		else if (dim != 3)
			throw_runtime_error("Tried to add a 3d element to a Mesh template which has already elements of dimension " + std::to_string(mesh_template->dim));
		MeshTemplateElementBrickC2 *res = new MeshTemplateElementBrickC2(inds);
		elements.push_back(res);
		res->link_nodes_with_domain(this);
		return res;
	}

	MeshTemplateElementTetraC1 *MeshTemplateElementCollection::add_tetra_3d_C1(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4)
	{
		if (dim == -1)
		{
			dim = 3;
		}
		else if (dim != 3)
			throw_runtime_error("Tried to add a 3d element to a Mesh template which has already elements of dimension " + std::to_string(mesh_template->dim));
		MeshTemplateElementTetraC1 *res = new MeshTemplateElementTetraC1(n1, n2, n3, n4);
		elements.push_back(res);
		res->link_nodes_with_domain(this);
		return res;
	}

	MeshTemplateElementTetraC2 *MeshTemplateElementCollection::add_tetra_3d_C2(const std::vector<nodeindex_t> &inds)
	{
		if (dim == -1)
		{
			dim = 3;
		}
		else if (dim != 3)
			throw_runtime_error("Tried to add a 3d element to a Mesh template which has already elements of dimension " + std::to_string(mesh_template->dim));
		MeshTemplateElementTetraC2 *res = new MeshTemplateElementTetraC2(inds);
		elements.push_back(res);
		res->link_nodes_with_domain(this);
		return res;
	}

	
	void MeshTemplateElementCollection::set_element_code(DynamicBulkElementInstance *code_inst)
	{
		code_instance = code_inst;
		bool has_converted_to_C2 = false;
		std::string dom_space = code_inst->get_func_table()->dominant_space;
		// Make to C2 or C1TB if not done yet
		if (dom_space == "C1TB") 
		{
    		for (unsigned int ie = 0; ie < elements.size(); ie++)
			{
				auto *e = elements[ie];
			   if (dynamic_cast<MeshTemplateElementTriC1 *>(e) && !dynamic_cast<MeshTemplateElementTriC1TB *>(e) )
				{
						elements[ie] = dynamic_cast<MeshTemplateElementTriC1 *>(e)->convert_for_C1TB_space(mesh_template);
						delete e;
						e = elements[ie];				   
				}
			   else if (dynamic_cast<MeshTemplateElementTriC2 *>(e))
				{
						elements[ie] = dynamic_cast<MeshTemplateElementTriC2 *>(e)->convert_for_C2TB_space(mesh_template);
						delete e;
						e = elements[ie];				   
				}				
			}
		}
		else if (code_inst->get_func_table()->numfields_C2 || dom_space == "C2" || code_inst->get_func_table()->numfields_C2TB || dom_space == "C2TB") 
		{
			for (unsigned int ie = 0; ie < elements.size(); ie++)
			{
				auto *e = elements[ie];
				if (e->get_nnode_C2() <= 0)
				{
					elements[ie] = e->convert_for_C2_space(mesh_template);
					if (!elements[ie])
						throw_runtime_error("Cannot cast element to C2");
					delete e;
					has_converted_to_C2 = true;
					e = elements[ie];
				}
				if (dom_space == "C2TB" || dom_space=="C1TB")
				{
					if (dynamic_cast<MeshTemplateElementTriC2 *>(e))
					{
						elements[ie] = dynamic_cast<MeshTemplateElementTriC2 *>(e)->convert_for_C2TB_space(mesh_template);
						delete e;
						e = elements[ie];
					}
					else if (dynamic_cast<MeshTemplateElementTetraC2 *>(e))
					{
						elements[ie] = dynamic_cast<MeshTemplateElementTetraC2 *>(e)->convert_for_C2TB_space(mesh_template);
						delete e;
						e = elements[ie];
					}
				}
			}
		}
		if (has_converted_to_C2)
		{
			std::vector<MeshTemplatePeriodicIntermediateNodeInfo> remaining;
			// Ensure periodicity of the new nodes
			for (unsigned int i = 0; i < mesh_template->inter_nodes_periodic.size(); i++)
			{
				// std::cout << "PER NODE " << std::endl;
				unsigned num_per_master = 0;
				auto &pm = mesh_template->inter_nodes_periodic[i];
				for (auto ncorn : pm.parent_node_ids)
				{
					if (mesh_template->nodes[ncorn]->periodic_master >= 0)
						num_per_master++;
				}
				if (num_per_master)
				{
					if (num_per_master == pm.parent_node_ids.size())
					{
						// Locate the corresponding pair indiced by the periodic master corners
						std::vector<nodeindex_t> masters;
						for (auto ncorn : pm.parent_node_ids)
						{
							masters.push_back(mesh_template->nodes[ncorn]->periodic_master);
						}
						std::sort(masters.begin(), masters.end());
						bool found = false;
						for (const auto &pm2 : mesh_template->inter_nodes_periodic)
						{
							if (masters.size() != pm2.parent_node_ids.size())
								continue;
							found = true;
							for (unsigned int k = 0; k < masters.size(); k++)
								if (masters[k] != pm2.parent_node_ids[k])
								{
									found = false;
									break;
								}
							if (found)
							{
								mesh_template->nodes[pm.myself]->periodic_master = pm2.myself;
								//	  	 	std::cout << "FOUND NODE" << std::endl;
								break;
							}
						}
						if (!found)
						{
							std::cerr << "NODE AT " << std::to_string(mesh_template->nodes[pm.myself]->x) + ", " + std::to_string(mesh_template->nodes[pm.myself]->y) + ", " + std::to_string(mesh_template->nodes[pm.myself]->z) << " with num_per_mst " << num_per_master << std::endl;
							for (auto ncorn : pm.parent_node_ids)
							{
								std::cerr << "  BETWEEN NODE " << ncorn << " @ " << std::to_string(mesh_template->nodes[ncorn]->x) + ", " + std::to_string(mesh_template->nodes[ncorn]->y) + ", " + std::to_string(mesh_template->nodes[ncorn]->z) << " with periodic master " << mesh_template->nodes[ncorn]->periodic_master << std::endl;
							}

							std::cerr << "EXISTING PERIODIC MASTER ENTRIES" << std::endl;

							throw_runtime_error("Cannot find the corresponding L2 node for periodicity. Node at " + std::to_string(mesh_template->nodes[pm.myself]->x) + ", " + std::to_string(mesh_template->nodes[pm.myself]->y) + ", " + std::to_string(mesh_template->nodes[pm.myself]->z));
						}
					}
					else if (num_per_master > 1) // One periodic master can happen in corners at the element without periodic masters
					{
						remaining.push_back(pm);
					}
				}
			}
			if (!remaining.empty())
			{
				for (auto &pm : remaining)
				{
					std::cerr << "NODE AT " << std::to_string(mesh_template->nodes[pm.myself]->x) + ", " + std::to_string(mesh_template->nodes[pm.myself]->y) + ", " + std::to_string(mesh_template->nodes[pm.myself]->z) << std::endl;
					for (auto ncorn : pm.parent_node_ids)
					{
						std::cerr << "  BETWEEN NODE " << ncorn << " @ " << std::to_string(mesh_template->nodes[ncorn]->x) + ", " + std::to_string(mesh_template->nodes[ncorn]->y) + ", " + std::to_string(mesh_template->nodes[ncorn]->z) << " with periodic master " << mesh_template->nodes[ncorn]->periodic_master << std::endl;
					}
				}
				throw_runtime_error("TODO: Handle remaining possible periodic nodes " + std::to_string(remaining.size()));
			}
		}
	}

	MeshTemplateElementCollection::~MeshTemplateElementCollection()
	{
		// if (oomph_mesh) delete  oomph_mesh;
		for (unsigned int i = 0; i < elements.size(); i++)
			delete elements[i];
	}

	/////////////////////////////////

	bool facet_order_function(const MeshTemplateFacet *a, const MeshTemplateFacet *b)
	{
		if (a->sorted_inds.size() < b->sorted_inds.size())
			return true;
		else if (a->sorted_inds.size() > b->sorted_inds.size())
			return false;
		for (unsigned int i = 0; i < a->sorted_inds.size(); i++)
		{
			if (a->sorted_inds[i] < b->sorted_inds[i])
				return true;
			else if (a->sorted_inds[i] > b->sorted_inds[i])
				return false;
		}
		return false;
	}

	MeshTemplate::MeshTemplate() : problem(NULL), dim(-1),kdtree(1), facetmap(&facet_order_function), domain(NULL)
	{
	}

	void MeshTemplate::reset()
	{
		for (std::size_t i = 0; i < nodes.size(); i++) if (nodes[i]) delete nodes[i];
		for (std::size_t i = 0; i < bulk_element_collections.size(); i++) if (bulk_element_collections[i]) delete bulk_element_collections[i];
		nodes.clear();
		bulk_element_collections.clear();
		boundary_names.clear();
		facets.clear();
		dim=-1;
		facetmap.clear();
		domain=NULL;
		inter_nodes_periodic.clear();
		kdtree.reset(1);
	}

	MeshTemplate::~MeshTemplate()
	{
		reset();
	}

	nodeindex_t MeshTemplate::add_node(double x, double y, double z)
	{
		MeshTemplateNode *res = new MeshTemplateNode(x, y, z);
		res->index = nodes.size();
		nodes.push_back(res);
		if (kdtree.add_point(x, y, z) != res->index)
		{
			throw_runtime_error("Something is wrong with the KDTree");
		}
		return res->index;
	}

	void MeshTemplate::add_periodic_node_pair(const nodeindex_t &n1, const nodeindex_t &n2)
	{
		nodes[n2]->periodic_master = n1; // Just put the link here
		// nodeindex_t nsrc=n1;
		// unsigned cnt=0;
		/*
		while (nodes[nsrc]->periodic_master>=0)
		{
		 nsrc=nodes[nsrc]->periodic_master;
		 cnt++;
		 if (cnt>10000) {throw_runtime_error("Probably a circular periodic mapping");}
		}
		*/
		// nodes[n2]->periodic_master=nsrc;
		/*
		if (!nodes[n2]->periodic_master_of.empty())
		{
		//	throw_runtime_error("Moooop");
		 for (auto i : nodes[n2]->periodic_master_of)
		 {
		   nodes[i]->periodic_master=nsrc;
		  nodes[nsrc]->periodic_master_of.push_back(i);
		 }
		 nodes[n2]->periodic_master_of.clear();
		}
		nodes[nsrc]->periodic_master_of.push_back(n2);
		*/
	}

	nodeindex_t MeshTemplate::add_node_unique(double x, double y, double z)
	{
		int present = kdtree.point_present(x, y, z);
		if (present >= 0)
		{
			// std::cout << "NODE " << x << " " << y << "  " << z << std::endl;
			// auto v=this->get_node_position(present);
			// std::cout  << "  found " << v[0] << " " << v[1] << "  " << v[2] << std::endl;
			return present;
		}

		MeshTemplateNode *res = new MeshTemplateNode(x, y, z);
		res->index = nodes.size();
		nodes.push_back(res);
		if (kdtree.add_point(x, y, z) != res->index)
		{
			throw_runtime_error("Something is wrong with the KDTree");
		}

		return res->index;
	}

	nodeindex_t MeshTemplate::add_intermediate_node_unique(const nodeindex_t &n1, const nodeindex_t &n2)
	{
		nodeindex_t ni = add_node_unique(0.5 * (nodes[n1]->x + nodes[n2]->x), 0.5 * (nodes[n1]->y + nodes[n2]->y), 0.5 * (nodes[n1]->z + nodes[n2]->z));
		// Merge the boundary information!
		if (nodes[ni]->on_boundaries.empty())
		{
			std::set_intersection(nodes[n1]->on_boundaries.begin(), nodes[n1]->on_boundaries.end(), nodes[n2]->on_boundaries.begin(), nodes[n2]->on_boundaries.end(), std::inserter(nodes[ni]->on_boundaries, nodes[ni]->on_boundaries.begin()));
		}
		if (nodes[ni]->part_of_domain.empty())
		{
			std::set_intersection(nodes[n1]->part_of_domain.begin(), nodes[n1]->part_of_domain.end(), nodes[n2]->part_of_domain.begin(), nodes[n2]->part_of_domain.end(), std::inserter(nodes[ni]->part_of_domain, nodes[ni]->part_of_domain.begin()));
		}
		if (nodes.size() == (ni + 1) && (!nodes[ni]->on_boundaries.empty())) // NODE WAS ADDED, store the information for later on patching the periodicity
		{
			inter_nodes_periodic.push_back(MeshTemplatePeriodicIntermediateNodeInfo(ni, {n1, n2}));
		}

		return ni;
	}

	nodeindex_t MeshTemplate::add_intermediate_node_unique(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, bool boundary_possible)
	{
		nodeindex_t ni = add_node_unique((nodes[n1]->x + nodes[n2]->x + nodes[n3]->x) / 3, (nodes[n1]->y + nodes[n2]->y + nodes[n3]->y) / 3, (nodes[n1]->z + nodes[n2]->z + nodes[n3]->z) / 3);

		if (nodes[ni]->part_of_domain.empty())
		{
			std::set<MeshTemplateElementCollection *> part_of_domain, old;
			std::set_intersection(nodes[n1]->part_of_domain.begin(), nodes[n1]->part_of_domain.end(), nodes[n2]->part_of_domain.begin(), nodes[n2]->part_of_domain.end(), std::inserter(part_of_domain, part_of_domain.begin()));
			old = part_of_domain;
			part_of_domain.clear();
			std::set_intersection(nodes[n3]->part_of_domain.begin(), nodes[n3]->part_of_domain.end(), old.begin(), old.end(), std::inserter(part_of_domain, part_of_domain.begin()));
			nodes[ni]->part_of_domain = part_of_domain;
		}

		if (!boundary_possible)
			return ni;
		// Check boundary
		if (nodes[ni]->on_boundaries.empty())
		{
			std::set<unsigned int> on_boundaries, old;
			std::set_intersection(nodes[n1]->on_boundaries.begin(), nodes[n1]->on_boundaries.end(), nodes[n2]->on_boundaries.begin(), nodes[n2]->on_boundaries.end(), std::inserter(on_boundaries, on_boundaries.begin()));
			old = on_boundaries;
			on_boundaries.clear();
			std::set_intersection(nodes[n3]->on_boundaries.begin(), nodes[n3]->on_boundaries.end(), old.begin(), old.end(), std::inserter(on_boundaries, on_boundaries.begin()));
			nodes[ni]->on_boundaries = on_boundaries;
		}

		if (nodes.size() == (ni + 1) && (!nodes[ni]->on_boundaries.empty())) // NODE WAS ADDED, store the information for later on patching the periodicity
		{
			inter_nodes_periodic.push_back(MeshTemplatePeriodicIntermediateNodeInfo(ni, {n1, n2, n3}));
		}

		return ni;
	}

	nodeindex_t MeshTemplate::add_intermediate_node_unique(const nodeindex_t &n1, const nodeindex_t &n2, const nodeindex_t &n3, const nodeindex_t &n4, bool boundary_possible)
	{
		nodeindex_t ni = add_node_unique(0.25 * (nodes[n1]->x + nodes[n2]->x + nodes[n3]->x + nodes[n4]->x), 0.25 * (nodes[n1]->y + nodes[n2]->y + nodes[n3]->y + nodes[n4]->y),
										 0.25 * (nodes[n1]->z + nodes[n2]->z + nodes[n3]->z + nodes[n4]->z));

		if (nodes[ni]->part_of_domain.empty())
		{
			std::set<MeshTemplateElementCollection *> part_of_domain, old;
			std::set_intersection(nodes[n1]->part_of_domain.begin(), nodes[n1]->part_of_domain.end(), nodes[n2]->part_of_domain.begin(), nodes[n2]->part_of_domain.end(), std::inserter(part_of_domain, part_of_domain.begin()));
			old = part_of_domain;
			part_of_domain.clear();
			std::set_intersection(nodes[n3]->part_of_domain.begin(), nodes[n3]->part_of_domain.end(), old.begin(), old.end(), std::inserter(part_of_domain, part_of_domain.begin()));
			old = part_of_domain;
			part_of_domain.clear();
			std::set_intersection(nodes[n4]->part_of_domain.begin(), nodes[n4]->part_of_domain.end(), old.begin(), old.end(), std::inserter(part_of_domain, part_of_domain.begin()));
			nodes[ni]->part_of_domain = part_of_domain;
		}

		if (!boundary_possible)
			return ni;
		// Check boundary
		if (nodes[ni]->on_boundaries.empty())
		{
			std::set<unsigned int> on_boundaries, old;
			std::set_intersection(nodes[n1]->on_boundaries.begin(), nodes[n1]->on_boundaries.end(), nodes[n2]->on_boundaries.begin(), nodes[n2]->on_boundaries.end(), std::inserter(on_boundaries, on_boundaries.begin()));
			old = on_boundaries;
			on_boundaries.clear();
			std::set_intersection(nodes[n3]->on_boundaries.begin(), nodes[n3]->on_boundaries.end(), old.begin(), old.end(), std::inserter(on_boundaries, on_boundaries.begin()));
			old = on_boundaries;
			on_boundaries.clear();
			std::set_intersection(nodes[n4]->on_boundaries.begin(), nodes[n4]->on_boundaries.end(), old.begin(), old.end(), std::inserter(on_boundaries, on_boundaries.begin()));
			nodes[ni]->on_boundaries = on_boundaries;
		}

		if (nodes.size() == (ni + 1) && (!nodes[ni]->on_boundaries.empty())) // NODE WAS ADDED, store the information for later on patching the periodicity
		{
			inter_nodes_periodic.push_back(MeshTemplatePeriodicIntermediateNodeInfo(ni, {n1, n2, n3, n4}));
		}

		return ni;
	}

	unsigned int MeshTemplate::get_boundary_index(const std::string &boundname) const
	{
		unsigned int bi = boundary_names.size();
		for (unsigned int i = 0; i < boundary_names.size(); i++)
			if (boundary_names[i] == boundname)
			{
				bi = i;
				break;
			}
		if (bi == boundary_names.size())
			throw_runtime_error("Boundary " + boundname + " not in mesh template");
		return bi;
	}

	void MeshTemplate::add_node_to_boundary(const std::string &boundname, const nodeindex_t &ni)
	{
		unsigned int bi = boundary_names.size();
		for (unsigned int i = 0; i < boundary_names.size(); i++)
			if (boundary_names[i] == boundname)
			{
				bi = i;
				break;
			}
		if (bi == boundary_names.size())
		{
			boundary_names.push_back(boundname);
		}
		nodes[ni]->on_boundaries.insert(bi);
	}

	void MeshTemplate::add_nodes_to_boundary(const std::string &boundname, const std::vector<nodeindex_t> &ni)
	{
		unsigned int bi = boundary_names.size();
		for (unsigned int i = 0; i < boundary_names.size(); i++)
			if (boundary_names[i] == boundname)
			{
				bi = i;
				break;
			}
		if (bi == boundary_names.size())
		{
			boundary_names.push_back(boundname);
		}
		for (unsigned int i = 0; i < ni.size(); i++)
			nodes[ni[i]]->on_boundaries.insert(bi);
	}

	void MeshTemplate::add_facet_to_curve_entity(std::vector<unsigned> &facetnodes, MeshTemplateCurvedEntity *curved)
	{
		MeshTemplateFacet *nf = new MeshTemplateFacet(facetnodes, curved, &this->nodes);

		if (facetmap.count(nf))
		{
			MeshTemplateFacet *old = facets[facetmap[nf]];
			if (old->curved_entity && old->curved_entity != nf->curved_entity)
				throw_runtime_error("Cannot add a facet on two different curved entities");
			delete nf;
			return;
		}

		facetmap[nf] = facets.size();
		facets.push_back(nf);
	}

	MeshTemplateElementCollection *MeshTemplate::new_bulk_element_collection(std::string name)
	{
		MeshTemplateElementCollection *res = new MeshTemplateElementCollection(this, name);
		bulk_element_collections.push_back(res);
		return res;
	}

	MeshTemplateElementCollection *MeshTemplate::get_collection(std::string name)
	{
		for (unsigned int i = 0; i < bulk_element_collections.size(); i++)
			if (bulk_element_collections[i]->name == name)
				return bulk_element_collections[i];
		return NULL;
	}

	void MeshTemplate::link_periodic_nodes()
	{
		for (unsigned int i = 0; i < nodes.size(); i++)
		{
			if (nodes[i]->periodic_master >= 0 && nodes[i]->oomph_node)
			{
				int current_mst = nodes[i]->periodic_master;
				int cnt = 0;
				while (nodes[current_mst]->periodic_master >= 0)
				{
					current_mst = nodes[current_mst]->periodic_master;
					cnt++;
					if (cnt > 1000)
					{
						throw_runtime_error("Probably a circular periodic nodal mapping");
					}
				}
				//  	 throw_runtime_error("PERIODIC NODES MIGHT BE DEFUCT");
				nodes[i]->oomph_node->make_periodic(nodes[current_mst]->oomph_node);
			}
		}
	}

	void MeshTemplate::flush_oomph_nodes()
	{
		for (unsigned int i = 0; i < nodes.size(); i++)
		{
			nodes[i]->oomph_node = NULL; // TODO: Store them somewhere to have a nice interface connection
		}
	}

	void MeshTemplate::_find_opposite_interface_connections()
	{
		for (unsigned int bi = 0; bi < boundary_names.size(); bi++)
		{
			std::set<MeshTemplateElementCollection *> bound_domains;
			for (auto *d : bulk_element_collections)
			{
				bound_domains.insert(d);
			}
			for (auto *n : nodes)
			{
				if (n->on_boundaries.count(bi))
				{
					std::set<MeshTemplateElementCollection *> the_intersection; // Destination of intersect
					std::set_intersection(bound_domains.begin(), bound_domains.end(), n->part_of_domain.begin(), n->part_of_domain.end(), std::inserter(the_intersection, the_intersection.end()));
					bound_domains.swap(the_intersection);
					if (bound_domains.size() < 2)
					{
						break;
					}
				}
			}
			if (bound_domains.size() == 2)
			{
				std::vector<std::string> connnames;
				for (auto *d : bound_domains)
				{
					connnames.push_back(d->name + "/" + boundary_names[bi]);
				}
				this->_add_opposite_interface_connection(connnames[0], connnames[1]);
			}
		}
	}

	std::set<std::string> MeshTemplate::_find_interface_intersections()
	{
		std::set<std::string> all_intersects;
		for (auto *d : bulk_element_collections)
		{		  
		  std::vector<int> adjbinds;
		  for (auto &bname : d->get_adjacent_boundary_names()) adjbinds.push_back(get_boundary_index(bname));
		  // Now find nodes that are at least on two of these boundaries
		  for (auto *e : d->get_elements())
		  {
			std::vector<nodeindex_t> ninds = e->get_node_indices();
			for (unsigned int i = 0; i < ninds.size(); i++)
			{
				nodeindex_t ni = ninds[i];
				if (this->nodes[ni]->on_boundaries.size()>=2)
				{
					std::set<int> intersect;
					std::set_intersection(this->nodes[ni]->on_boundaries.begin(), this->nodes[ni]->on_boundaries.end(),adjbinds.begin(), adjbinds.end(),std::inserter(intersect, intersect.begin()));					
					if (intersect.size()>=2)
					{
						std::vector<int> intersectv(intersect.begin(), intersect.end());
						std::sort(intersectv.begin(), intersectv.end());
						std::vector<std::vector<int>> permutations;
						std::sort(intersectv.begin(), intersectv.end());						
						do {
							permutations.push_back(std::vector<int>(intersectv.begin(), intersectv.end()));
						} while (std::next_permutation(intersectv.begin(), intersectv.end()));
						for (auto & p : permutations)
						{
							std::string intername=d->name;
							for (auto &b : p) intername+="/"+boundary_names[b];
							all_intersects.insert(intername);
						}
						
					}					
				}
		  	}
		  }
		}
		return all_intersects;
	}

	BulkElementBase *MeshTemplate::factory_element(MeshTemplateElement *el, MeshTemplateElementCollection *coll)
	{
		// Generate all nodes if not present
		const JITFuncSpec_Table_FiniteElement_t *functable = coll->code_instance->get_func_table();
		BulkElementBase::__CurrentCodeInstance = coll->code_instance;
		unsigned nC1 = functable->numfields_C1;
		unsigned nC2 = functable->numfields_C2;
		unsigned nC1TB = functable->numfields_C1TB;		
		unsigned nC2TB = functable->numfields_C2TB;
		unsigned ntot = nC1 +nC1TB+ nC2 + nC2TB;

		unsigned n_lagrangian_type = 1;

		unsigned nodal_dim = coll->nodal_dimension();
		unsigned n_lagrangian = coll->lagrangian_dimension();
		bool require_macro_elem = false;

		std::vector<nodeindex_t> nodeindices = el->get_node_indices();
		std::string domspace = coll->code_instance->get_func_table()->dominant_space;
		if (el->get_geometric_type_index() == 8 && (domspace == "C1" || domspace=="C1TB")) // QC2 -> QC1
		{
			// Reduce the element
			nodeindices = {nodeindices[0], nodeindices[2], nodeindices[6], nodeindices[8]};
		}
		else if (el->get_geometric_type_index() == 9 && domspace == "C1") // TC2 -> TC1
		{
			// Reduce the element
			nodeindices = {nodeindices[0], nodeindices[1], nodeindices[2]};
		}
		else if (el->get_geometric_type_index() == 2 && (domspace == "C1" || domspace == "C1TB")) // LC2->LC1
		{
			// Reduce the element
			nodeindices = {nodeindices[0], nodeindices[2]};
		}		
		else if (el->get_geometric_type_index()==14 && (domspace == "C1" || domspace == "C1TB") ) //BC2 ->BC1
		{
		
			nodeindices = {nodeindices[0], nodeindices[2], nodeindices[6], nodeindices[8],nodeindices[18],nodeindices[20],nodeindices[24],nodeindices[26]};
		}
		else if (el->get_geometric_type_index()==0 ) //BC2 ->BC1
		{
		
			nodeindices = {nodeindices[0]};
		}
		

		std::vector<bool> constructed_oomph_node(nodeindices.size(), false);
		for (unsigned int ni = 0; ni < nodeindices.size(); ni++)
		{
			unsigned nii = nodeindices[ni];
			if (nodes[nii]->on_curved_facet)
				require_macro_elem = true;
			//  std::cout << ni << "  " << nii << "  " << nodes.size() << std::endl;
			//  std::cout << nodes[nii] << std::endl;
			if (!nodes[nii]->oomph_node)
			{
				//	std::cout << "ADDING WITH NODAL DIM " << nodal_dim << std::endl;
				constructed_oomph_node[ni] = true;
				if (nodes[nii]->on_boundaries.empty() && !coll->all_nodes_as_boundary_nodes)
				{
					nodes[nii]->oomph_node = new pyoomph::Node(this->problem->time_stepper_pt(), n_lagrangian, n_lagrangian_type, nodal_dim, 1, ntot);
				}
				else
				{
					nodes[nii]->oomph_node = new pyoomph::BoundaryNode(this->problem->time_stepper_pt(), n_lagrangian, n_lagrangian_type, nodal_dim, 1, ntot);
				}

				if (nodal_dim > 0)
				{
					nodes[nii]->oomph_node->x(0) = nodes[nii]->x;
					if (nodal_dim > 1)
					{
						nodes[nii]->oomph_node->x(1) = nodes[nii]->y;
						if (nodal_dim > 2)
						{
							nodes[nii]->oomph_node->x(2) = nodes[nii]->z;
						}
					}
				}
				//      	std::cout << "POS " << nodes[nii]->oomph_node->x(0)<< " = " << nodes[nii]->x << " and " << nodes[nii]->oomph_node->x(1) <<" = " << nodes[nii]->y << " and " <<  nodes[nii]->oomph_node->x(2) <<" = " << nodes[nii]->z <<std::endl;

				for (unsigned int i = 0; i < n_lagrangian; i++)
					nodes[nii]->oomph_node->xi(i) = nodes[nii]->oomph_node->x(i);
					
				for (unsigned int t = 1; t < 2; t++)
				{
				  for (unsigned int i=0;i<nodes[nii]->oomph_node->ndim();i++)
				  {
					nodes[nii]->oomph_node->x(t,i) = nodes[nii]->oomph_node->x(i);					
				  }
			   }
			}
		}
		BulkElementBase *res = BulkElementBase::create_from_template(this, el);

		// Remove unused oomph-nodes... happens, when having e.g. second order elements in the template, but due to the selected spaces (e.g. only linear) a lower order element was generated
		for (unsigned int ni = 0; ni < nodeindices.size(); ni++)
		{
			if (!constructed_oomph_node[ni])
				continue;
			unsigned nii = nodeindices[ni];
			if (nodes[nii]->oomph_node)
			{
				bool found = false;
				for (unsigned int nj = 0; nj < res->nnode(); nj++)
				{
					if (res->node_pt(nj) == nodes[nii]->oomph_node)
					{
						found = true;
						break;
					}
				}
				if (!found)
				{
					delete nodes[nii]->oomph_node;
					nodes[nii]->oomph_node = NULL;
				}
			}
		}

		if (pyoomph_verbose)
			std::cout << "MACRO ELEM: MESH REQUIRES MACRO " << require_macro_elem << std::endl;
		if (require_macro_elem)
		{
			if (!domain)
				domain = new MeshTemplateDomain();
			if (dynamic_cast<BulkElementQuad2dC1 *>(res) || dynamic_cast<BulkElementQuad2dC2 *>(res))
			{
				MeshTemplateQMacroElement2 *macro = NULL;
				for (unsigned int ifacet = 0; ifacet < el->nfacets(); ifacet++)
				{
					MeshTemplateFacet *test_facet = el->construct_facet(ifacet);
					if (pyoomph_verbose)
					{
						std::cout << "LOOKING FOR FACET ";
						for (auto &i : test_facet->sorted_inds)
							std::cout << "  " << i;
						std::cout << std::endl;
					}
					for (auto &fm : facetmap)
					{
						if (pyoomph_verbose)
						{
							std::cout << "	DEFINED ";
							for (auto &i : fm.first->sorted_inds)
								std::cout << "  " << i;
							std::cout << std::endl;
						}
					}

					if (facetmap.count(test_facet))
					{
						if (pyoomph_verbose)
							std::cout << "MACRO ELEM: FACET IDENTIFIED" << std::endl;
						MeshTemplateFacet *actual_facet = facets[facetmap[test_facet]];
						if (actual_facet->curved_entity)
						{
							if (pyoomph_verbose)
								std::cout << "MACRO ELEM: CURVED ENTITY PRESENT : MACRO " << macro << std::endl;
							if (!macro)
							{
								macro = new MeshTemplateQMacroElement2(domain, domain->nmacro_element(), el, &nodes);
								domain->push_back_macro_element(macro);
							}
							macro->set_facet(ifacet, actual_facet, test_facet);
						}
					}
					delete test_facet;
				}
				if (macro)
				{
					if (pyoomph_verbose)
						std::cout << "ADDING MACRO ELEMENT " << std::endl;
					res->set_macro_elem_pt(macro);
					res->map_nodes_on_macro_element();
				}
			}
			else if (dynamic_cast<BulkElementTri2dC1 *>(res) || dynamic_cast<BulkElementTri2dC2 *>(res))
			{
				MeshTemplateTMacroElement2 *macro = NULL;
				for (unsigned int ifacet = 0; ifacet < el->nfacets(); ifacet++)
				{
					MeshTemplateFacet *test_facet = el->construct_facet(ifacet);
					if (pyoomph_verbose)
					{
						std::cout << "LOOKING FOR FACET ";
						for (auto &i : test_facet->sorted_inds)
							std::cout << "  " << i;
						std::cout << std::endl;
					}
					for (auto &fm : facetmap)
					{
						if (pyoomph_verbose)
						{
							std::cout << "	DEFINED ";
							for (auto &i : fm.first->sorted_inds)
								std::cout << "  " << i;
							std::cout << std::endl;
						}
					}

					if (facetmap.count(test_facet))
					{
						if (pyoomph_verbose)
							std::cout << "MACRO ELEM: FACET IDENTIFIED" << std::endl;
						MeshTemplateFacet *actual_facet = facets[facetmap[test_facet]];
						if (actual_facet->curved_entity)
						{
							if (pyoomph_verbose)
								std::cout << "MACRO ELEM: CURVED ENTITY PRESENT : MACRO " << macro << std::endl;
							if (!macro)
							{
								macro = new MeshTemplateTMacroElement2(domain, domain->nmacro_element(), el, &nodes);
								domain->push_back_macro_element(macro);
							}
							macro->set_facet(ifacet, actual_facet, test_facet);
						}
					}
					delete test_facet;
				}
				if (macro)
				{
					if (pyoomph_verbose)
						std::cout << "ADDING MACRO ELEMENT " << std::endl;
					res->set_macro_elem_pt(macro);
					res->map_nodes_on_macro_element();
				}
			}
			else if (dynamic_cast<BulkElementBrick3dC1 *>(res) || dynamic_cast<BulkElementBrick3dC2 *>(res))
			{
				MeshTemplateQMacroElement3 *macro = NULL;
				for (unsigned int ifacet = 0; ifacet < el->nfacets(); ifacet++)
				{
					MeshTemplateFacet *test_facet = el->construct_facet(ifacet);
					if (pyoomph_verbose)
					{
						std::cout << "LOOKING FOR FACET ";
						for (auto &i : test_facet->sorted_inds)
							std::cout << "  " << i;
						std::cout << std::endl;
					}
					for (auto &fm : facetmap)
					{
						if (pyoomph_verbose)
						{
							std::cout << "	DEFINED ";
							for (auto &i : fm.first->sorted_inds)
								std::cout << "  " << i;
							std::cout << std::endl;
						}
					}

					if (facetmap.count(test_facet))
					{
						if (pyoomph_verbose)
							std::cout << "MACRO ELEM: FACET IDENTIFIED" << std::endl;
						MeshTemplateFacet *actual_facet = facets[facetmap[test_facet]];
						if (actual_facet->curved_entity)
						{
							if (pyoomph_verbose)
								std::cout << "MACRO ELEM: CURVED ENTITY PRESENT : MACRO " << macro << std::endl;
							if (!macro)
							{
								macro = new MeshTemplateQMacroElement3(domain, domain->nmacro_element(), el, &nodes);
								domain->push_back_macro_element(macro);
							}
							macro->set_facet(ifacet, actual_facet, test_facet);
						}
					}
					delete test_facet;
				}
				if (macro)
				{
					if (pyoomph_verbose)
						std::cout << "ADDING MACRO ELEMENT " << std::endl;
					res->set_macro_elem_pt(macro);
					res->map_nodes_on_macro_element();
				}
			}
			else
				throw_runtime_error("MacroElements not implement for this element type");
		}

		BulkElementBase::__CurrentCodeInstance = NULL;
		return res;
	}

	std::map<int, MeshTemplateCurvedEntity *> MeshTemplateCurvedEntity::load_from_strings(const std::vector<std::string> &s, size_t &currline)
	{
		throw_runtime_error("IMPLEM");
		return std::map<int, MeshTemplateCurvedEntity *>();
	}

	CurvedEntityCatmullRomSpline::CurvedEntityCatmullRomSpline(const std::vector<std::vector<double>> &_pts) : MeshTemplateCurvedEntity(1), pts(_pts)
	{
		N = pts.size();
		std::vector<double> vs = pts[0];
		for (unsigned int i = 0; i < vs.size(); i++)
			vs[i] = 2.0 * vs[i] - pts[1][i];
		std::vector<double> ve = pts.back();
		for (unsigned int i = 0; i < ve.size(); i++)
			ve[i] = 2.0 * ve[i] - pts[pts.size() - 2][i];
		//	# TODO: Allow closed
		pts.insert(pts.begin(), vs);
		pts.push_back(ve);
		gen_samples(100);
	}

	std::string CurvedEntityCatmullRomSpline::get_information_string()
	{
		std::ostringstream oss;
		oss << pts.size() << std::endl;
		for (unsigned int i = 0; i < pts.size(); i++)
			write_vector_information(pts[i], oss);
		return oss.str();
	}

	void CurvedEntityCatmullRomSpline::gen_samples(unsigned num)
	{
		samples.resize(num);
		for (unsigned int i = 0; i < samples.size(); i++)
			samples[i] = (N - 1) * i / (samples.size() - 1.0);
		samplepos.resize(samples.size(), std::vector<double>(pts[0].size()));
		for (unsigned int i = 0; i < samplepos.size(); i++)
		{
			interpolate(samples[i], samplepos[i]);
		}
	}

	void CurvedEntityCatmullRomSpline::interpolate(double t, std::vector<double> &pos)
	{
		t = std::min(std::max(t, 0.0), N - 1.0);
		unsigned offs = t;
		t = t - int(t);
		if (offs + 1 == N)
		{
			offs--;
			t = 1.0;
		}
		double t2 = t * t;
		double t3 = t2 * t;
		double s0 = -.5 * t3 + t2 - .5 * t;
		double s1 = 1.5 * t3 - 2.5 * t2 + 1;
		double s2 = -1.5 * t3 + 2 * t2 + .5 * t;
		double s3 = 0.5 * t3 - 0.5 * t2;
		pos.resize(pts.front().size());
		for (unsigned int i = 0; i < pos.size(); i++)
		{
			pos[i] = s0 * pts[offs][i] + s1 * pts[offs + 1][i] + s2 * pts[offs + 2][i] + s3 * pts[offs + 3][i];
		}
	}

	void CurvedEntityCatmullRomSpline::dinterpolate(double t, std::vector<double> &dpos)
	{
		t = std::min(std::max(t, 0.0), N - 1.0);
		unsigned offs = t;
		t = t - int(t);
		if (offs + 1 == N)
		{
			offs--;
			t = 1.0;
		}
		double t2 = 2 * t;
		double t3 = 3 * t * t;
		double s0 = -.5 * t3 + t2 - .5;
		double s1 = 1.5 * t3 - 2.5 * t2;
		double s2 = -1.5 * t3 + 2 * t2 + .5;
		double s3 = 0.5 * t3 - 0.5 * t2;
		dpos.resize(pts.front().size());
		for (unsigned int i = 0; i < dpos.size(); i++)
		{
			dpos[i] = s0 * pts[offs][i] + s1 * pts[offs + 1][i] + s2 * pts[offs + 2][i] + s3 * pts[offs + 3][i];
		}
	}

	void CurvedEntityCatmullRomSpline::parametric_to_position(const unsigned &t, const std::vector<double> &parametric, std::vector<double> &position)
	{
		interpolate(parametric[0], position);
	}
	void CurvedEntityCatmullRomSpline::position_to_parametric(const unsigned &t, const std::vector<double> &position, std::vector<double> &parametric)
	{
		double mindist = 1.0e20;
		int minind = -1;
		for (unsigned i = 0; i < samplepos.size(); i++)
		{
			double dist = 0.0;
			const std::vector<double> &ps = samplepos[i];
			for (unsigned int j = 0; j < std::min(position.size(), ps.size()); j++)
				dist += (position[j] - ps[j]) * (position[j] - ps[j]);
			if (dist < mindist)
			{
				mindist = dist;
				minind = i;
			}
		}
		double t0 = samples[minind];
		std::vector<double> tang;
		dinterpolate(t0, tang);
		int maxind = -1;
		double besttang = -1.0;
		for (unsigned int i = 0; i < std::min(position.size(), tang.size()); i++)
		{
			if (tang[i] * tang[i] > besttang)
			{
				besttang = tang[i] * tang[i];
				maxind = i;
			}
		}
		// Optimize via newton
		double troot = t0;
		mindist = sqrt(mindist);
		unsigned iter = 0;
		double eps = 1e-10;
		std::vector<double> current, J;

		while (mindist > eps)
		{
			interpolate(troot, current);
			mindist = current[maxind] - position[maxind];
			dinterpolate(troot, J);
			troot -= mindist / J[maxind];
			mindist = fabs(mindist);
			if (iter++ > 1000)
				break;
		}
		if (mindist > eps)
		{
			if (samples.size() < 10000)
			{
				gen_samples(2 * samples.size());
				position_to_parametric(t, position, parametric);
				return;
			}
			else
			{
				throw_runtime_error("Cannot invert spline");
			}
		}
		else
		{
			interpolate(troot, current);
			double dist = 0.0;
			for (unsigned int j = 0; j < std::min(position.size(), current.size()); j++)
				dist += (position[j] - current[j]) * (position[j] - current[j]);
			if (sqrt(dist) > 100 * eps)
			{
				if (samples.size() < 10000)
				{
					gen_samples(2 * samples.size());
					position_to_parametric(t, position, parametric);
					return;
				}
				else
				{
					throw_runtime_error("Cannot invert spline");
				}
			}
		}
		// TODO: Also a dist check
		parametric[0] = troot;
	}

	/*
	void MeshTemplate::finalise_creation()
	{
	 for (unsigned int i=0;i<bulk_element_collections.size();i++)
	 {

	   bulk_element_collections[i]->oomph_mesh=new pyoomph::Mesh(this->problem,this,bulk_element_collections[i]->name);

	   //Aquire fields on the nodes
	   for (unsigned int ie=0;ie<bulk_element_collections[i]->elements.size();ie++)
	   {
			 const MeshTemplateElement * e=bulk_element_collections[i]->elements[ie];
		 const DynamicBulkElementInstance * code=e->get_dynamic_code();

			 for (unsigned int iC2=0;iC2<code->get_num_fields_C2();iC2++)
			 {
					 int global_field=code->get_global_field_index_C2(iC2);
					 if (global_field<0) {throw_runtime_error("Field of element not bound to a global field");}
					 for (unsigned int in=0;in<e->get_nnode_C2();in++)
					 {
						MeshTemplateNode * n=nodes[e->get_node_index_C2(in)];
						bool found=false;
						for (unsigned int fi=0;fi<n->global_field_at_value_index.size();fi++)
						{
							if (n->global_field_at_value_index[fi]==(unsigned int)global_field)
							{
								found=true; break;
							}
						}
						if (!found)
						{
							n->global_field_at_value_index.push_back(global_field);
							dynamic_cast<PyNodeBase*>(n->oomph_node)->global_field_at_value_index.push_back(global_field);
							n->oomph_node->resize(n->oomph_node->nvalue()+1);
							n->oomph_node->set_value(n->oomph_node->nvalue()-1,0.0);
						}
					 }
			 }


			 for (unsigned int iC1=0;iC1<code->get_num_fields_C1();iC1++)
			 {
					 int global_field=code->get_global_field_index_C1(iC1);
					 if (global_field<0) {throw_runtime_error("Field of element not bound to a global field");}
					 for (unsigned int in=0;in<e->get_nnode_C1();in++)
					 {
						MeshTemplateNode * n=nodes[e->get_node_index_C1(in)];
						bool found=false;
						for (unsigned int fi=0;fi<n->global_field_at_value_index.size();fi++)
						{
							if (n->global_field_at_value_index[fi]==(unsigned int)global_field)
							{
								found=true; break;
							}
						}
						if (!found)
						{
							n->global_field_at_value_index.push_back(global_field);
							dynamic_cast<PyNodeBase*>(n->oomph_node)->global_field_at_value_index.push_back(global_field);
							n->oomph_node->resize(n->oomph_node->nvalue()+1);
							n->oomph_node->set_value(n->oomph_node->nvalue()-1,0.0);

						}
					 }
			 }
	   }




	   problem->add_sub_mesh(bulk_element_collections[i]->oomph_mesh);
	 }




	 if (problem->mesh_pt())
	 {
		 problem->rebuild_global_mesh();
	 }
	 else
	 {
		 problem->build_global_mesh();
	 }

	}
	*/
	// pyoomph::Mesh * mesh_inner=new pyoomph::Mesh(meshtemplate,"inner");
	// problem.add_sub_mesh(mesh_inner);
	// problem.build_global_mesh();

}
