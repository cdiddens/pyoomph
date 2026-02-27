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


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
namespace py = pybind11;

#include "../mesh.hpp"
#include "../nodes.hpp"
#include "../meshtemplate.hpp"
#include "../problem.hpp"
#include "../elements.hpp"
#include "../mesh1d.hpp"
#include "../mesh2d.hpp"
#include "../mesh3d.hpp"
#include "../tracers.hpp"

namespace pyoomph
{

	class PyMeshTemplateCurvedEntity : public MeshTemplateCurvedEntity
	{
	public:
		using MeshTemplateCurvedEntity::MeshTemplateCurvedEntity;
		virtual void parametric_to_pos(const unsigned &t, const py::array_t<double> &param, py::array_t<double> &pos)
		{
			throw_runtime_error("parametric_to_pos not specialised");
		}
		virtual void pos_to_parametric(const unsigned &t, const py::array_t<double> &pos, py::array_t<double> &param)
		{
			throw_runtime_error("pos_to_parametric not specialised");
		}
		void parametric_to_position(const unsigned &t, const std::vector<double> &parametric, std::vector<double> &position)
		{
			py::array_t<double> parr(position.size(), position.data());
			parametric_to_pos(t, py::cast(parametric), parr);
			py::buffer_info buff = parr.request();
			for (unsigned int i = 0; i < position.size(); i++)
			{
				position[i] = ((double *)(buff.ptr))[i];
			}
		}
		void position_to_parametric(const unsigned &t, const std::vector<double> &position, std::vector<double> &parametric)
		{
			py::array_t<double> parr(parametric.size(), parametric.data());
			pos_to_parametric(t, py::cast(position), parr);
			py::buffer_info buff = parr.request();
			for (unsigned int i = 0; i < parametric.size(); i++)
			{
				parametric[i] = ((double *)(buff.ptr))[i];
			}
		}

		virtual void ensure_periodicity(py::array_t<double> &parametrics)
		{
		}

		void apply_periodicity(std::vector<std::vector<double>> &parametric)
		{
			if (parametric.empty())
				return;
			py::array_t<double> parr(parametric.size() * parametric[0].size());
			py::buffer_info buff = parr.request();
			for (unsigned int i = 0; i < parametric.size(); i++)
			{
				for (unsigned int j = 0; j < parametric[i].size(); j++)
				{
					((double *)(buff.ptr))[i * parametric[i].size() + j] = parametric[i][j];
				}
			}
			parr.resize({parametric.size(), parametric[0].size()});
			ensure_periodicity(parr);
			for (unsigned int i = 0; i < parametric.size(); i++)
			{
				for (unsigned int j = 0; j < parametric[i].size(); j++)
				{
					parametric[i][j] = ((double *)(buff.ptr))[i * parametric[i].size() + j];
				}
			}
		};
	};

	class PyMeshTemplateCurvedEntityTrampoline : public PyMeshTemplateCurvedEntity
	{
	public:
		using PyMeshTemplateCurvedEntity::PyMeshTemplateCurvedEntity;

		void pos_to_parametric(const unsigned &t, const py::array_t<double> &pos, py::array_t<double> &param) override
		{
			PYBIND11_OVERLOAD(void, PyMeshTemplateCurvedEntity, pos_to_parametric, t, pos, param);
		}

		void parametric_to_pos(const unsigned &t, const py::array_t<double> &param, py::array_t<double> &pos) override
		{
			PYBIND11_OVERLOAD(void, PyMeshTemplateCurvedEntity, parametric_to_pos, t, param, pos);
		}

		void ensure_periodicity(py::array_t<double> &parametrics) override
		{
			PYBIND11_OVERLOAD(void, PyMeshTemplateCurvedEntity, ensure_periodicity, parametrics);
		}
	};

	class PyMeshTemplateTrampoline : public MeshTemplate
	{
	public:
		using MeshTemplate::MeshTemplate;
		void _add_opposite_interface_connection(const std::string &sideA, const std::string &sideB) override
		{
			PYBIND11_OVERLOAD(void, MeshTemplate, _add_opposite_interface_connection, sideA, sideB);
		}
	};

	

}

using namespace pybind11::literals;

static py::class_<oomph::Data> *py_decl_OomphData = NULL;
static py::class_<oomph::Mesh> *py_decl_OomphMesh = NULL;
static py::class_<pyoomph::Mesh, oomph::Mesh> *py_decl_PyoomphMesh = NULL;
static py::class_<oomph::GeneralisedElement> * py_decl_GeneralisedElement =NULL;
void PyDecl_Mesh(py::module &m)
{
	py_decl_OomphData = new py::class_<oomph::Data>(m, "OomphData");
	py_decl_OomphMesh = new py::class_<oomph::Mesh>(m, "OomphMesh");
	py_decl_PyoomphMesh = new py::class_<pyoomph::Mesh, oomph::Mesh>(m, "Mesh");
	py_decl_GeneralisedElement=new py::class_<oomph::GeneralisedElement>(m, "OomphGeneralisedElement");
}

void PyReg_Mesh(py::module &m)
{

	py::class_<pyoomph::MeshTemplateCurvedEntity>(m, "MeshTemplateCurvedEntityBase")
		.def_static("load_from_strings", &pyoomph::MeshTemplateCurvedEntity::load_from_strings)
		.def("get_information_string", &pyoomph::MeshTemplateCurvedEntity::get_information_string)
		.def("get_pos_from_parametric", &pyoomph::MeshTemplateCurvedEntity::parametric_to_position)
		.def("get_parametric_from_pos", &pyoomph::MeshTemplateCurvedEntity::position_to_parametric)
		.doc()="A generic class representing a relation for a curved boundary representation";

	py::class_<pyoomph::CurvedEntityCircleArc, pyoomph::MeshTemplateCurvedEntity>(m, "CurvedEntityCircleArc")
		.def(py::init<const std::vector<double> &, const std::vector<double> &, const std::vector<double> &>());

	py::class_<pyoomph::CurvedEntityCylinderArc, pyoomph::MeshTemplateCurvedEntity>(m, "CurvedEntityCylinderArc")
		.def(py::init<const std::vector<double> &, const std::vector<double> &, const std::vector<double> &>());

	py::class_<pyoomph::CurvedEntityCatmullRomSpline, pyoomph::MeshTemplateCurvedEntity>(m, "CurvedEntityCatmullRomSpline")
		.def(py::init<const std::vector<std::vector<double>> &>());

	py::class_<pyoomph::CurvedEntitySpherePart, pyoomph::MeshTemplateCurvedEntity>(m, "CurvedEntitySpherePart")
		.def(py::init<const std::vector<double> &, const std::vector<double> &, const std::vector<double> &>()); // center,onsphere, tangent

	py::class_<pyoomph::PyMeshTemplateCurvedEntity, pyoomph::PyMeshTemplateCurvedEntityTrampoline, pyoomph::MeshTemplateCurvedEntity>(m, "MeshTemplateCurvedEntity")
		.def(py::init<unsigned>())
      .def("pos_to_parametric",&pyoomph::PyMeshTemplateCurvedEntity::pos_to_parametric, py::arg("t"),py::arg("pos"),py::arg("param"))
      .def("parametric_to_pos",&pyoomph::PyMeshTemplateCurvedEntity::parametric_to_pos,py::arg("t"),py::arg("param"),py::arg("pos"))
      .def("ensure_periodicity",&pyoomph::PyMeshTemplateCurvedEntity::ensure_periodicity,py::arg("param"));

		

	py::class_<pyoomph::LagrZ2ErrorEstimator>(m, "Z2ErrorEstimator")
		.def(py::init<>())
		.def_readwrite("use_Lagrangian", &pyoomph::LagrZ2ErrorEstimator::use_Lagrangian);

	py_decl_OomphData->def("set_time_stepper", &oomph::Data::set_time_stepper)
		.def("pin", &oomph::Data::pin)
		.def("unpin", &oomph::Data::unpin)
		.def("is_pinned", &oomph::Data::is_pinned)
		.def("eqn_number", (long(oomph::Data::*)(const unsigned &) const) & oomph::Data::eqn_number)
		.def("nvalue", (unsigned(oomph::Data::*)() const) & oomph::Data::nvalue)
		.def("value", (double(oomph::Data::*)(unsigned const &) const) & oomph::Data::value)
		.def("value_at_t", (double(oomph::Data::*)(unsigned const &, unsigned const &) const) & oomph::Data::value)
		.def("ntstorage", &oomph::Data::ntstorage)
		.def("set_value", (void(oomph::Data::*)(unsigned const &, double const &)) & oomph::Data::set_value)
		.def("set_value_at_t", (void(oomph::Data::*)(unsigned const &, unsigned const &, double const &)) & oomph::Data::set_value);

	py::class_<pyoomph::Node, oomph::Data>(m, "Node")
		.def("x", (const double &(pyoomph::Node::*)(const unsigned int &) const) & pyoomph::Node::x)
		.def("x_at_t", (const double &(pyoomph::Node::*)(const unsigned int &,const unsigned int &) const) & pyoomph::Node::x)
		.def("x_lagr", (const double &(pyoomph::Node::*)(const unsigned int &) const) & pyoomph::Node::xi)
		.def("ndim", (unsigned(pyoomph::Node::*)() const) & pyoomph::Node::ndim)
		.def("set_x", [](pyoomph::Node *n, unsigned const &ind, double const &x)
			 { n->x(ind) = x; })
		.def("set_x_at_t", [](pyoomph::Node *n, unsigned const &t,unsigned const &ind, double const &x)
			 { n->x(t,ind) = x; })
		.def("set_x_lagr", [](pyoomph::Node *n, unsigned const &ind, double const &x)
			 { n->xi(ind) = x; })
		.def("pin_position", (void(pyoomph::Node::*)(const unsigned &)) & pyoomph::Node::pin_position)
		.def("unpin_position", (void(pyoomph::Node::*)(const unsigned &)) & pyoomph::Node::unpin_position)
		.def("position_is_pinned", (bool(pyoomph::Node::*)(const unsigned &)) & pyoomph::Node::position_is_pinned)
		.def("is_hanging", (bool(pyoomph::Node::*)(const int &) const) & pyoomph::Node::is_hanging, "index"_a = -1)
		.def("variable_position_pt", &pyoomph::Node::variable_position_pt, py::return_value_policy::reference)
		.def("is_on_boundary", (bool(pyoomph::Node::*)() const) & pyoomph::Node::is_on_boundary)
		.def("set_obsolete", &pyoomph::Node::set_obsolete)
		.def("is_obsolete", &pyoomph::Node::is_obsolete)
		.def("remove_from_boundary",&pyoomph::Node::remove_from_boundary)
		.def("add_to_boundary", &pyoomph::Node::add_to_boundary)						
		//	.def("is_on_boundary_index",(bool (pyoomph::Node::*,const unsigned )() const) &pyoomph::Node::is_on_boundary)
		.def("get_boundary_indices", [](pyoomph::Node *n)
			 {std::set<unsigned> *pt; n->get_boundaries_pt(pt); std::set<unsigned> res; if (pt) {for (auto i : *pt) res.insert(i);}  return res; })
		.def("additional_value_index", &pyoomph::Node::additional_value_index)
		.def("set_coordinates_on_boundary",[](pyoomph::Node *self,unsigned boundary_index, std::vector<double> &zeta) {
			oomph::Vector<double> zeta_prime(zeta.size()); for(unsigned int i=0;i<zeta.size();i++){zeta_prime[i]=zeta[i];};
			self->set_coordinates_on_boundary(boundary_index, zeta_prime);
		})
		.def("get_coordinates_on_boundary",[](pyoomph::Node *self,unsigned boundary_index) {
			oomph::Vector<double> zeta_prime(self->ncoordinates_on_boundary(boundary_index),0);
			self->get_coordinates_on_boundary(boundary_index, zeta_prime);
			std::vector<double> zeta(zeta_prime.size(),0);
			for(unsigned int i=0;i<zeta.size();i++){zeta[i]=zeta_prime[i];};
			return zeta;
		})
		.def("_make_periodic", [](pyoomph::Node *slv, pyoomph::Node *mst, pyoomph::Mesh *mesh)
			 {
	    pyoomph::Node * imst=mst;
	    if (mst->is_a_copy())
	    {
	     mst=mesh->resolve_copy_master(mst);
	     if (!mst) {throw_runtime_error("Strange.. the master node is already a copy, but it cannot be resolved");}
	    }
	    if (slv->is_a_copy())
	    {
	     pyoomph::Node * omst=mesh->resolve_copy_master(slv);
	     if (mst!=omst)  {
	       if (omst!=slv)
	       {
	        std::ostringstream oss;
	        oss<<std::endl;
	        oss << "SLAVE "; for (unsigned int i=0;i<slv->ndim();i++) oss << slv->x(i) << "  " ; oss<< std::endl;
	        oss << "IMST "; for (unsigned int i=0;i<imst->ndim();i++) oss << imst->x(i) << "  " ; oss<< std::endl;
	        oss << "OMST "; for (unsigned int i=0;i<omst->ndim();i++) oss << omst->x(i) << "  " ; oss<< std::endl;
	        oss << "MST "; for (unsigned int i=0;i<mst->ndim();i++) oss << mst->x(i) << "  " ; oss<< std::endl;	        	        	        	        	        	        	        
	        throw_runtime_error("Inconsistent periodic boundaries:"+oss.str());
	       }
	       else
	       {
	        return;
	       }	      
	      }
	    }
	    slv->make_periodic(mst);
	    mesh->store_copy_master(slv,mst); })
		.def("_nullify_residual_contribution", [](pyoomph::Node *n, pyoomph::DynamicBulkElementInstance *for_ci, int index)
			 {
		 throw_runtime_error("Nullified dofs are deactivated for now... Never used so far");
		 /*
		 pyoomph::BoundaryNode* bn=dynamic_cast<pyoomph::BoundaryNode*>(n);
		 if (bn)
		 {
		  if (!bn->nullified_dofs.count(for_ci)) bn->nullified_dofs[for_ci]=std::set<int>();
		  bn->nullified_dofs[for_ci].insert(index);
		 }
		 else throw_runtime_error("Cannot nullify non-boundary nodes");
		 */ })
		.def("is_on_boundary", (bool(pyoomph::Node::*)(unsigned const &) const) & pyoomph::Node::is_on_boundary);

	auto &decl_GeneralisedElement = (*py_decl_GeneralisedElement);
	decl_GeneralisedElement 	
		.def("_debug_hessian", [](oomph::GeneralisedElement *self, std::vector<double> Y, std::vector<std::vector<double>> C, double epsilon)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) { throw_runtime_error("Not a BulkelementBase"); return;	   }
			be->debug_hessian(Y,C,epsilon); })
		.def("get_meshio_type_index", [](oomph::GeneralisedElement *self) -> unsigned int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return -1;
			return be->get_meshio_type_index(); })
		.def("assemble_hessian_and_mass_hessian",[](oomph::GeneralisedElement *self)
		   {
			  pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			  if (!be) { throw_runtime_error("Not a BulkelementBase"); 	   }		   
			  oomph::RankThreeTensor<double> hess,mhess;
			  be->assemble_hessian_and_mass_hessian(hess,mhess);
			  unsigned n=hess.nindex1();
			  auto hdata=py::array_t<double>({n,n,n});
			  double * hdest=(double*)hdata.request().ptr;
			  auto mdata=py::array_t<double>({n,n,n});
			  double * mdest=(double*)mdata.request().ptr;			  
			  for (unsigned int i=0;i<n;i++) for (unsigned int j=0;j<n;j++) for (unsigned int k=0;k<n;k++) 
			   {
			     hdest[i*n*n+j*n+k]=hess(i,j,k);
			     mdest[i*n*n+j*n+k]=mhess(i,j,k);
			   }
			  return std::make_tuple(hdata,mdata);
			  
		   }
		  )
		.def("refinement_level", [](oomph::GeneralisedElement *self) -> unsigned int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0;
			return be->refinement_level(); })
		.def("non_halo_proc_ID", [](oomph::GeneralisedElement *self) -> int
			 {
			#ifdef OOMPH_HAS_MPI
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return -1;			
			return be->non_halo_proc_ID(); 
			#else
			return -1;
			#endif
			})
		.def("describe_my_dofs", [](oomph::GeneralisedElement *self, std::string in)
			 {
	   std::ostringstream oss;
   	pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	   if (be) be->describe_my_dofs(std::cout,in);
	   return oss.str(); })
		.def("get_nodal_index_by_name", [](oomph::GeneralisedElement *self, pyoomph::Node *n, std::string name) -> int
			 {
	   std::ostringstream oss;
   	pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	   if (!be) return -1; 
	   return be->get_nodal_index_by_name(n,name); })
		.def(
			"get_code_instance", [](oomph::GeneralisedElement *self) -> pyoomph::DynamicBulkElementInstance *
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return NULL;
			return be->get_code_instance(); },
			py::return_value_policy::reference)
		.def("get_macro_element_coordinate_at_s",[](oomph::GeneralisedElement *self, std::vector<double> s) -> std::vector<double>
		   {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return {};
			oomph::Vector<double> so(s.size()); for (unsigned int i=0;i<s.size();i++) so[i]=s[i];
			return be->get_macro_element_coordinate_at_s(so);			
		   })
		.def("evalulate_local_expression_at_s", [](oomph::GeneralisedElement *self, int index, std::vector<double> s) -> double
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0.0;
			oomph::Vector<double> so(s.size()); for (unsigned int i=0;i<s.size();i++) so[i]=s[i];
			return be->eval_local_expression_at_s(index,so); })
		.def("evalulate_local_expression_at_midpoint", [](oomph::GeneralisedElement *self, int index) -> double
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0.0;
			return be->eval_local_expression_at_midpoint(index); })
		.def("evalulate_local_expression_at_node_index", [](oomph::GeneralisedElement *self, int index, unsigned node_index) -> double
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0.0;
			return be->eval_local_expression_at_node(index,node_index); })
		.def(
			"node_pt", [](oomph::GeneralisedElement *self, unsigned int i) -> pyoomph::Node *
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return NULL;
			return dynamic_cast<pyoomph::Node*>(be->node_pt(i)); },
			py::return_value_policy::reference)
		.def("nodes",[](oomph::GeneralisedElement *self) 
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);			
			std::vector<pyoomph::Node*> nodes;
			if (be)
			{			
				for (unsigned int i=0;i<be->nnode();i++) nodes.push_back(dynamic_cast<pyoomph::Node*>(be->node_pt(i)));
			}
			return nodes;
			},py::return_value_policy::reference)
		.def("boundary_nodes",[](oomph::GeneralisedElement *self,int boundary_index) 
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);			
			std::vector<pyoomph::Node*> nodes;
			if (be)
			{		
				if (boundary_index<0)	
				{
					for (unsigned int i=0;i<be->nnode();i++) if (be->node_pt(i)->is_on_boundary()) nodes.push_back(dynamic_cast<pyoomph::Node*>(be->node_pt(i)));
				}
				else
				{
					for (unsigned int i=0;i<be->nnode();i++) if (be->node_pt(i)->is_on_boundary(boundary_index)) nodes.push_back(dynamic_cast<pyoomph::Node*>(be->node_pt(i)));
				}
			}
			return nodes;
			},py::return_value_policy::reference,py::arg("boundary_index")=-1)
		.def("boundary_vertex_nodes",[](oomph::GeneralisedElement *self,int boundary_index) 
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);			
			std::vector<pyoomph::Node*> nodes;
			if (be)
			{		
				for (unsigned int i=0;i<be->nvertex_node();i++) if (be->vertex_node_pt(i)->is_on_boundary(boundary_index)) nodes.push_back(dynamic_cast<pyoomph::Node*>(be->vertex_node_pt(i)));				
			}
			return nodes;
			},py::return_value_policy::reference)
		.def("_connect_periodic_tree",[](oomph::GeneralisedElement *self,oomph::GeneralisedElement *other, int mydir, int otherdir)
			{
				dynamic_cast<pyoomph::BulkElementBase*>(self)->connect_periodic_tree(dynamic_cast<pyoomph::BulkElementBase*>(other),mydir,otherdir);
			})
		//Returns oomph::Data and value indices for a fields. If use_elemental_indices, it will return (NULL,-1) for elemental node indices that do not have data associated
		.def("get_field_data_list",[](oomph::GeneralisedElement *self, std::string name,bool use_elemental_indices) -> std::vector<std::pair<oomph::Data *,int> >
		   {
			 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			 if (!be) return {};
			 return be->get_field_data_list(name,use_elemental_indices);
		   },py::return_value_policy::reference)
		.def(
			"opposite_node_pt", [](oomph::GeneralisedElement *self, unsigned int i) -> pyoomph::Node *
			{
			pyoomph::InterfaceElementBase * ie=dynamic_cast<pyoomph::InterfaceElementBase*>(self);
			if (!ie) return NULL;
			return ie->opposite_node_pt(i); },
			py::return_value_policy::reference)
		.def("get_attached_element_equation_mapping", [](oomph::GeneralisedElement *self, std::string which)
		    {
			   pyoomph::InterfaceElementBase * ie=dynamic_cast<pyoomph::InterfaceElementBase*>(self);
			   if (!ie) { throw_runtime_error("Not an interface element"); }		    
			   return ie->get_attached_element_equation_mapping(which);
		    }
		    )
		.def("set_opposite_interface_element", [](oomph::GeneralisedElement *self, oomph::GeneralisedElement *opp,std::vector<double> offs)
			 {
			pyoomph::InterfaceElementBase * ie=dynamic_cast<pyoomph::InterfaceElementBase*>(self);
			pyoomph::InterfaceElementBase * io=dynamic_cast<pyoomph::InterfaceElementBase*>(opp);
			if (!ie || !io) { throw_runtime_error("Can only connect interface elements this way"); }
			ie->set_opposite_interface_element(io,offs); })
		.def(
			"vertex_node_pt", [](oomph::GeneralisedElement *self, unsigned int i) -> pyoomph::Node *
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return NULL;
			return dynamic_cast<pyoomph::Node*>(be->vertex_node_pt(i)); },
			py::return_value_policy::reference)
		.def("ndof", [](oomph::GeneralisedElement *self) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) {
			  pyoomph::BulkElementODE0d * ode=dynamic_cast<pyoomph::BulkElementODE0d*>(self);			
			  if (ode) return ode->ndof();
			  else return self->ndof();
			}
			return be->ndof(); })
		.def("eqn_number", [](oomph::GeneralisedElement *self, unsigned int i) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return -1;
			return be->eqn_number(i); })
		.def("nvertex_node", [](oomph::GeneralisedElement *self) -> unsigned
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0;
			return be->nvertex_node(); })
		.def_property(
			"_elemental_error_max_override", [](oomph::GeneralisedElement *self)
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0.0;
			return be->elemental_error_max_override; },
			[](oomph::GeneralisedElement *self, double val)
			{
				pyoomph::BulkElementBase *be = dynamic_cast<pyoomph::BulkElementBase *>(self);
				if (!be)
					return;
				be->elemental_error_max_override = val;
			})
		.def("num_Z2_flux_terms", [](oomph::GeneralisedElement *self) -> unsigned int
			 {
	  	pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		if (!be) return 0;
		else return be->num_Z2_flux_terms(); })
		.def(
			"get_bulk_element", [](oomph::GeneralisedElement *self)
			{
	    pyoomph::InterfaceElementBase * ie=dynamic_cast<pyoomph::InterfaceElementBase*>(self);
	    if (!ie) return (oomph::GeneralisedElement*)NULL;
	    else return dynamic_cast<oomph::GeneralisedElement*>(ie->bulk_element_pt()); },
			py::return_value_policy::reference)
		.def(
			"get_opposite_bulk_element", [](oomph::GeneralisedElement *self)
			{
	    pyoomph::InterfaceElementBase * ie=dynamic_cast<pyoomph::InterfaceElementBase*>(self);
	    if (!ie) return (oomph::GeneralisedElement*)NULL;
	    else {
	     pyoomph::InterfaceElementBase * oi=ie->get_opposite_side();
	     if (!oi) return (oomph::GeneralisedElement*)NULL;
	     else return dynamic_cast<oomph::GeneralisedElement*>(oi->bulk_element_pt());
	    } },
			py::return_value_policy::reference)
		.def(
			"get_opposite_interface_element", [](oomph::GeneralisedElement *self)
			{
	    pyoomph::InterfaceElementBase * ie=dynamic_cast<pyoomph::InterfaceElementBase*>(self);
	    if (!ie) return (oomph::GeneralisedElement*)NULL;
	    else {
	     pyoomph::InterfaceElementBase * oi=ie->get_opposite_side();
	     if (!oi) return (oomph::GeneralisedElement*)NULL;
	     else return dynamic_cast<oomph::GeneralisedElement*>(oi);
	    } },
			py::return_value_policy::reference)
		.def("get_outline", [](oomph::GeneralisedElement *self,bool lagrangian) -> py::array_t<double>
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be || !be->nnode()) return py::array_t<double>();
			auto outl=be->get_outline(lagrangian);
			unsigned ndim=be->node_pt(0)->ndim();
			unsigned nnode=outl.size()/ndim;
			auto data=py::array_t<double>({ndim,nnode});
			double * dest=(double*)data.request().ptr;
			for (unsigned int i=0;i<outl.size();i++) dest[i]=outl[i];
			return data; },py::arg("lagrangian")=false)
		.def("get_debug_jacobian_info", [](oomph::GeneralisedElement *self)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			oomph::Vector<double> R;
			oomph::DenseMatrix<double> J;
			std::vector<std::string> dofnames;
			be->get_debug_jacobian_info(R,J,dofnames);
			std::vector<double> Rv(R.size()); for (unsigned int i=0;i<R.size();i++) Rv[i]=R[i];
			std::vector<double> Jv(J.ncol()*J.nrow()); for (unsigned int i=0;i<J.nrow();i++) for (unsigned int j=0;j<J.ncol();j++) Jv[J.ncol()*i+j]=J(i,j);
			return std::make_tuple(Rv,Jv,dofnames); })
		.def("nnode", [](oomph::GeneralisedElement *self) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0;
			return be->nnode(); })
		.def("get_quality_factor", [](oomph::GeneralisedElement *self) -> double
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 1.0;
			return be->get_quality_factor(); })
		.def("get_initial_cartesian_nondim_size", [](oomph::GeneralisedElement *self) -> double
			 {
	  	 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		 if (!be) return 0.0;
		 return be->initial_cartesian_nondim_size; })
		.def("set_initial_cartesian_nondim_size", [](oomph::GeneralisedElement *self, double s)
			 {
	  	 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		 return be->initial_cartesian_nondim_size=s; })
		.def("get_initial_quality_factor", [](oomph::GeneralisedElement *self) -> double
			 {
	  	 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		 if (!be) return 1.0;
		 return be->initial_quality_factor; })
		.def("set_initial_quality_factor", [](oomph::GeneralisedElement *self, double s)
			 {
	  	 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		 return be->initial_quality_factor=s; })
		.def("get_Eulerian_midpoint", [](oomph::GeneralisedElement *self)
			 {
	  pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	  std::vector<double> res;
	  if (!be) return res;
	  oomph::Vector<double> ores=be->get_Eulerian_midpoint_from_local_coordinate();
	  res.resize(ores.size()); for (unsigned int i=0;i<ores.size();i++) res[i]=ores[i];
	  return res; })
		.def("get_Lagrangian_midpoint", [](oomph::GeneralisedElement *self)
			 {
	  pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	  std::vector<double> res;
	  if (!be) return res;
	  oomph::Vector<double> ores=be->get_Lagrangian_midpoint_from_local_coordinate();
	  res.resize(ores.size()); for (unsigned int i=0;i<ores.size();i++) res[i]=ores[i];
	  return res; })	  
		.def("get_current_cartesian_nondim_size", [](oomph::GeneralisedElement *self) -> double
			 {
	  	 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		 if (!be) return 0.0;
		 return be->size(); })
		.def("nnode_1d", [](oomph::GeneralisedElement *self) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0;
			return be->nnode_1d(); })
		.def(
			"boundary_node_pt", [](oomph::GeneralisedElement *self, int dir, int index) -> pyoomph::Node *
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0;
			return dynamic_cast<pyoomph::Node*>(be->boundary_node_pt(dir,index)); },
			py::return_value_policy::reference)

		.def("ninternal_data", [](oomph::GeneralisedElement *self) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) {
				pyoomph::BulkElementODE0d * ode=dynamic_cast<pyoomph::BulkElementODE0d *>(self);
				if (ode) return ode->ninternal_data(); else return self->ninternal_data();
			}
			return be->ninternal_data(); })
		.def("nexternal_data", [](oomph::GeneralisedElement *self) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) {
				pyoomph::BulkElementODE0d * ode=dynamic_cast<pyoomph::BulkElementODE0d *>(self);
				if (ode) return ode->ninternal_data(); else return self->nexternal_data();
			}
			return be->nexternal_data(); })
		.def("get_dof_names", [](oomph::GeneralisedElement *self)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return std::vector<std::string>();
			else return be->get_dof_names(); })
		.def(
			"get_father_element", [](oomph::GeneralisedElement *self) -> oomph::GeneralisedElement *
			{
				pyoomph::BulkElementBase *be = dynamic_cast<pyoomph::BulkElementBase *>(self);
				if (!be)
					return NULL;
				return dynamic_cast<pyoomph::BulkElementBase *>(be->father_element_pt()); },
			py::return_value_policy::reference)
		.def(
			"get_macro_element", [](oomph::GeneralisedElement *self) -> oomph::MacroElement *
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return NULL;
			return be->macro_elem_pt(); },
			py::return_value_policy::reference)
		.def("set_macro_element", [](oomph::GeneralisedElement *self, oomph::MacroElement *m, bool map_nodes)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return ;
			be->set_macro_elem_pt(m);
			if (map_nodes) be->map_nodes_on_macro_element(); })
		.def("create_interpolated_node",[](oomph::GeneralisedElement *self, const std::vector<double> & s,bool as_boundary_node) -> pyoomph::Node *
		   {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return NULL;			
			oomph::Vector<double> soomph(s.size());
			for (unsigned int i=0;i<s.size();i++) soomph[i]=s[i];
			return be->create_interpolated_node(soomph,as_boundary_node);
		   },py::return_value_policy::reference)
		.def("local_coordinate_of_node", [](oomph::GeneralisedElement *self, unsigned int l) -> std::vector<double>
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return std::vector<double>();
			oomph::Vector<double> s;
			be->local_coordinate_of_node(l, s);
			std::vector<double> res(s.size());
			for (unsigned int i=0;i<s.size();i++) res[i]=s[i];
			return res; })
		.def("set_undeformed_macro_element", [](oomph::GeneralisedElement *self, oomph::MacroElement *m)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return ;
			be->set_undeformed_macro_elem_pt(m); })
		.def("map_nodes_on_macro_element", [](oomph::GeneralisedElement *self)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return ;
			be->map_nodes_on_macro_element(); })
		.def("locate_zeta", [](oomph::GeneralisedElement *self, const std::vector<double> &_zeta, const std::vector<double> &_s, const bool &use_coordinate_as_initial_guess)
			 {
	  pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	  if (!be) return std::vector<double>();
	  oomph::Vector<double> zeta(_zeta.size());
	  for (unsigned i=0;i<_zeta.size();i++) zeta[i]=_zeta[i];
	  oomph::Vector<double> s(_s.size());
	  for (unsigned i=0;i<_s.size();i++) s[i]=_s[i];
	  oomph::GeomObject* geom_object_pt=be;
	  be->locate_zeta(zeta,geom_object_pt, s, use_coordinate_as_initial_guess);
	  if (!geom_object_pt) return std::vector<double>();
  	  std::vector<double> res(s.size());
  	  for (unsigned int i=0;i<s.size();i++) res[i]=s[i];
  	  return res; })
		.def("get_interpolated_nodal_values_at_s", [](oomph::GeneralisedElement *self, unsigned t, const std::vector<double> &_s)
			 {
	  pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	  if (!be) return std::vector<double>();
  	  oomph::Vector<double> s(_s.size());
	  for (unsigned i=0;i<_s.size();i++) s[i]=_s[i];
  	  oomph::Vector<double> vals;
	  be->get_interpolated_values(t,s,vals);
	  std::vector<double> res(vals.size());
	  for (unsigned int i=0;i<vals.size();i++) res[i]=vals[i];
	  return res; })
		.def("get_interpolated_position_at_s", [](oomph::GeneralisedElement *self, unsigned t, const std::vector<double> &_s, bool lagr)
			 {
	  pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	  if (!be) return std::vector<double>();
  	  oomph::Vector<double> s(_s.size());
	  for (unsigned i=0;i<_s.size();i++) s[i]=_s[i];
  	  oomph::Vector<double> vals(be->nodal_dimension());
  	  if (!lagr) be->interpolated_x(t,s,vals);
  	  else  be->interpolated_xi(s,vals);
	  std::vector<double> res(vals.size());
	  for (unsigned int i=0;i<vals.size();i++) res[i]=vals[i];
	  return res; })
		.def("get_interpolated_discontinuous_at_s", [](oomph::GeneralisedElement *self, unsigned t, const std::vector<double> &_s)
			 {
	  pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	  if (!be) return std::vector<double>();
  	  oomph::Vector<double> s(_s.size());
	  for (unsigned i=0;i<_s.size();i++) s[i]=_s[i];
  	  oomph::Vector<double> vals(be->nodal_dimension());
  	  be->get_interpolated_discontinuous_values(t,s,vals);
	  std::vector<double> res(vals.size());
	  for (unsigned int i=0;i<vals.size();i++) res[i]=vals[i];
	  return res; })
		.def("dim", [](oomph::GeneralisedElement *self) -> int
			 {
		  pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	     if (!be) return -1;
	     return be->dim(); })
		.def("external_data_pt", (oomph::Data * &(oomph::GeneralisedElement::*)(const unsigned &)) & oomph::GeneralisedElement::external_data_pt, py::return_value_policy::reference)
		.def("internal_data_pt", (oomph::Data * &(oomph::GeneralisedElement::*)(const unsigned &)) & oomph::GeneralisedElement::internal_data_pt, py::return_value_policy::reference);

	auto &decl_OomphMesh = (*py_decl_OomphMesh);
	decl_OomphMesh
		.def(
			"as_pyoomph_mesh", [](oomph::Mesh *self)
			{ return dynamic_cast<pyoomph::Mesh *>(self); },
			py::return_value_policy::reference)
		.def("add_node_to_mesh",[](oomph::Mesh *self,pyoomph::Node *n)
			 {
				self->add_node_pt(n);
			 })
		.def("prune_dead_nodes",[](oomph::Mesh *self,bool with_bounds) 
			{
				if (with_bounds) self->prune_dead_nodes();
				else dynamic_cast<pyoomph::TemplatedMeshBase*>(self)->prune_dead_nodes_without_respecting_boundaries();				
			})
		.def("output_paraview", [](oomph::Mesh *self, const std::string &fname, const unsigned &order)
			 { std::ofstream f(fname); self->output_paraview(f,order); })
		.def("nelement", &oomph::Mesh::nelement)
		.def(
			"element_pt", [](oomph::Mesh &self, const unsigned &ei) -> oomph::GeneralisedElement *
			{ if (ei>=self.nelement()) return (pyoomph::BulkElementBase *)(NULL); else return  dynamic_cast<pyoomph::BulkElementBase *>(self.element_pt(ei)); },
			py::return_value_policy::reference)
		.def(
			"boundary_element_pt", [](oomph::Mesh &self, const unsigned &bi, const unsigned &ei) -> oomph::GeneralisedElement *
			{ return dynamic_cast<pyoomph::BulkElementBase *>(self.boundary_element_pt(bi, ei)); },
			py::return_value_policy::reference)
		.def("face_index_at_boundary", [](oomph::Mesh &self, const unsigned &bi, const unsigned &ei)
			 { return self.face_index_at_boundary(bi, ei); })
		.def("nboundary", &oomph::Mesh::nboundary)
		.def("nboundary_node", &oomph::Mesh::nboundary_node)
		.def("nboundary_element", &oomph::Mesh::nboundary_element)
		.def("resolve_copy_master_node", [](oomph::Mesh *self, pyoomph::Node *n)->pyoomph::Node *
			 { 
				if (!dynamic_cast<pyoomph::Mesh*>(self)) return NULL;
				return dynamic_cast<pyoomph::Node *>(dynamic_cast<pyoomph::Mesh*>(self)->resolve_copy_master(n)); 
			},py::return_value_policy::reference
			)
		.def("_disable_adaptation", [](oomph::Mesh *self)
			 {
    oomph::RefineableMeshBase* refmesh =dynamic_cast<oomph::RefineableMeshBase*>(self);
    if (refmesh) refmesh->disable_adaptation(); })
		.def("_enable_adaptation", [](oomph::Mesh *self)
			 {
    oomph::RefineableMeshBase* refmesh =dynamic_cast<oomph::RefineableMeshBase*>(self);
    if (refmesh) refmesh->enable_adaptation(); })
		.def("nnode", &oomph::Mesh::nnode)
		.def("_set_interpolate_lagrangian_on_remeshing", [](oomph::Mesh *self, bool lagr)
			 {
			pyoomph::Mesh* mesh =dynamic_cast<pyoomph::Mesh*>(self);
			if (mesh) mesh->interpolated_lagrangian_coordinates_at_remeshing=lagr;
			 })
		.def("get_elemental_errors", [](oomph::Mesh *self)
			 {
			oomph::Vector<double> elerrs(self->nelement(),0.0);
			oomph::RefineableMeshBase* refmesh =dynamic_cast<oomph::RefineableMeshBase*>(self);
			if (refmesh)
			{
			  if (refmesh->is_adaptation_enabled())
			  {
				 oomph::ErrorEstimator* error_estimator_pt=refmesh->spatial_error_estimator_pt();
				 if (error_estimator_pt)
				 {
					error_estimator_pt->get_element_errors(self,elerrs);
				 }
			  }
			}
			std::vector<double> res(elerrs.size());
			for (unsigned int i=0;i<elerrs.size();i++) res[i]=elerrs[i];
			return res; })
		.def(
			"node_pt", [](oomph::Mesh *self, unsigned int i) -> pyoomph::Node *
			{ return dynamic_cast<pyoomph::Node *>(self->node_pt(i)); },
			py::return_value_policy::reference)
		.def(
			"boundary_node_pt", [](oomph::Mesh &self, const unsigned &b, const unsigned &n)
			{ return dynamic_cast<pyoomph::Node *>(self.boundary_node_pt(b, n)); },
			py::return_value_policy::reference);

	auto &decl_PyoomphMesh = (*py_decl_PyoomphMesh);
	decl_PyoomphMesh
		.def("activate_duarte_debug", &pyoomph::Mesh::activate_duarte_debug)
		.def("check_integrity",&pyoomph::Mesh::check_integrity)
		.def("prepare_zeta_interpolation", [](pyoomph::Mesh *self, pyoomph::Mesh *old_mesh){self->prepare_zeta_interpolation(old_mesh);})
		.def("remove_boundary_nodes",[](pyoomph::Mesh *self) {self->remove_boundary_nodes();})		
		.def("remove_boundary_nodes_of_bound",[](pyoomph::Mesh *self,unsigned b) {self->remove_boundary_nodes(b);})		
		.def("add_interpolated_nodes_at",&pyoomph::Mesh::add_interpolated_nodes_at,py::return_value_policy::reference)
		.def("add_boundary_node",[](pyoomph::Mesh *self,unsigned bind,pyoomph::Node *n) {self->add_boundary_node(bind,n);})
		.def("flush_element_storage", [](pyoomph::Mesh *self){self->flush_element_storage();})
		.def("_set_time_level_for_projection", [](pyoomph::Mesh *self, unsigned time_level){self->set_time_level_for_projection(time_level);})
		.def("get_field_information", [](pyoomph::Mesh *self)
			 { return self->get_field_information(); })
		.def("describe_my_dofs", [](pyoomph::Mesh *self, std::string in)
			 {std::ostringstream oss;self->describe_my_dofs(oss,in);return oss.str(); })
		.def("_pin_all_my_dofs", [](pyoomph::Mesh *self, std::set<std::string> only_dofs, std::set<std::string> ignore_dofs, std::set<unsigned> ignore_continuous_at_interfaces)
			 { self->pin_all_my_dofs(only_dofs, ignore_dofs, ignore_continuous_at_interfaces); })
		.def("generate_interface_elements", [](pyoomph::Mesh *m, const std::string &bn, pyoomph::Mesh *im, pyoomph::DynamicBulkElementInstance *jitcode)
			 { m->generate_interface_elements(bn, im, jitcode); })
		.def("is_mesh_distributed", [](pyoomph::Mesh *m)
			 { return m->is_mesh_distributed(); })
		.def("_save_state", [](pyoomph::Mesh *m)
			 {std::vector<double> data; m->_save_state(data); return data; })
		.def("_setup_information_from_old_mesh", &pyoomph::Mesh::_setup_information_from_old_mesh)
		.def("_load_state", &pyoomph::Mesh::_load_state)
		.def("_pin_noncontributing_dofs", &pyoomph::Mesh::pin_noncontributing_dofs)
		.def("has_interface_dof_id", &pyoomph::Mesh::has_interface_dof_id)
		.def("list_integral_functions", &pyoomph::Mesh::list_integral_functions)
		.def("list_local_expressions", &pyoomph::Mesh::list_local_expressions)
		.def("prepare_interpolation", &pyoomph::Mesh::prepare_interpolation)
		.def("nodal_interpolate_from", &pyoomph::Mesh::nodal_interpolate_from)
		.def("nodal_interpolate_along_boundary", &pyoomph::Mesh::nodal_interpolate_along_boundary)
		.def("_evaluate_integral_function", [](pyoomph::Mesh *m, const std::string &n)
			 { return m->evaluate_integral_function(n); })
		.def("_evaluate_extremum", [](pyoomph::Mesh *m, const std::string &n,int sign,unsigned flags)
			 {  
				pyoomph::BulkElementBase * extreme_element;
			    oomph::Vector<double> extreme_local_coords;
				GiNaC::ex resval=m->evaluate_extremum(n,sign,extreme_element,extreme_local_coords,flags); 
				std::vector<double> s(extreme_local_coords.size());
				for (unsigned i=0;i<extreme_local_coords.size();i++) s[i]=extreme_local_coords[i];
				//return std::make_tuple(resval,s,std::unique_ptr<oomph::GeneralisedElement,py::nodelete>(dynamic_cast<oomph::GeneralisedElement*>(extreme_element)));
				return std::make_tuple(resval,s,dynamic_cast<oomph::GeneralisedElement*>(extreme_element));
			 
			 },py::return_value_policy::reference)			
		.def("ensure_external_data", [](pyoomph::Mesh *m)
			 { m->ensure_external_data(); })
		.def("ensure_halos_for_periodic_boundaries", [](pyoomph::Mesh *m)
			 { m->ensure_halos_for_periodic_boundaries(); })
		.def("nrefined", [](pyoomph::Mesh *m)
			 { return m->nrefined(); })
		.def("nunrefined", [](pyoomph::Mesh *m)
			 { return m->nunrefined(); })
		.def("invalidate_lagrangian_kdtree", [](pyoomph::Mesh *m)
			 { m->invalidate_lagrangian_kdtree(); })
		.def_property(
			"min_permitted_error", [](pyoomph::Mesh *m)
			{ return m->min_permitted_error(); },
			[](pyoomph::Mesh *m, double e)
			{ m->min_permitted_error() = e; })
		.def_property(
			"max_permitted_error", [](pyoomph::Mesh *m)
			{ return m->max_permitted_error(); },
			[](pyoomph::Mesh *m, double e)
			{ m->max_permitted_error() = e; })
		.def_property(
			"max_refinement_level", [](pyoomph::Mesh *m)
			{ return dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->max_refinement_level(); },
			[](pyoomph::Mesh *m, unsigned l)
			{ dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->max_refinement_level() = l; })
		.def_property(
			"min_refinement_level", [](pyoomph::Mesh *m)
			{ return dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->min_refinement_level(); },
			[](pyoomph::Mesh *m, unsigned l)
			{ dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->min_refinement_level() = l; })
		.def_property(
			"max_keep_unrefined", [](pyoomph::Mesh *m)
			{ return dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->max_keep_unrefined(); },
			[](pyoomph::Mesh *m, unsigned l)
			{ dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->max_keep_unrefined() = l; })
		.def("boundary_coordinate_bool", &pyoomph::Mesh::boundary_coordinates_bool)
		.def("is_boundary_coordinate_defined", &pyoomph::Mesh::is_boundary_coordinate_defined)
		.def("fill_node_index_to_node_map",[](pyoomph::Mesh *self) 
			{std::map<oomph::Node *, unsigned> node2index; 
			self->fill_node_map(node2index);
			std::vector<pyoomph::Node *> index2node(node2index.size(),NULL);; 
			for (auto & n2i : node2index){index2node[n2i.second]=dynamic_cast<pyoomph::Node*>(n2i.first);}
			return index2node;
			}, py::return_value_policy::reference)
		.def("setup_interior_boundary_elements", [](pyoomph::Mesh *self, unsigned bindex)
			 {
	 pyoomph::TemplatedMeshBase * templmesh=dynamic_cast<pyoomph::TemplatedMeshBase*>(self);
	 if (!templmesh) return;
	 templmesh->setup_interior_boundary_elements(bindex); })
		.def("fill_dof_types", [](pyoomph::Mesh *self, py::array_t<int> &desc)
			 { self->fill_dof_types((int *)desc.request().ptr); })
		.def("set_lagrangian_nodal_coordinates", &pyoomph::Mesh::set_lagrangian_nodal_coordinates)
		.def("get_refinement_pattern", [](pyoomph::Mesh *self)
			 {
	  oomph::TreeBasedRefineableMeshBase * tbself=dynamic_cast<oomph::TreeBasedRefineableMeshBase*>(self);
	  if (!tbself)
	  {
	   return std::vector<py::array_t<unsigned>>();
	  }
	  unsigned milev,malev;
	  tbself->get_refinement_levels(milev,malev);
	  if (malev==0) 	   return std::vector<py::array_t<unsigned>>();
	  oomph::Vector<oomph::Vector<unsigned>> ref;
	  tbself->get_refinement_pattern(ref);
	  std::vector<py::array_t<unsigned>> result;
	  result.resize(ref.size());
	  for (unsigned int i=0;i<ref.size();i++)
	  {
	   result[i]=py::array_t<unsigned>({static_cast<unsigned>(ref[i].size())});
	   unsigned * dest=(unsigned*)result[i].request().ptr;
	   for (unsigned int j=0;j<ref[i].size();j++) dest[j]=ref[i][j];
	  }
	  return result; })
		.def("refine_base_mesh", [](pyoomph::Mesh *self, const std::vector<std::vector<unsigned>> &refine)
			 {
	  oomph::TreeBasedRefineableMeshBase * tbself=dynamic_cast<oomph::TreeBasedRefineableMeshBase*>(self);
	  if (!tbself)
	  {
	   return;
	  }
	  oomph::Vector<oomph::Vector<unsigned>> ref(refine.size());
	  for (unsigned int i=0;i<refine.size();i++)
	  {
	   ref[i].resize(refine[i].size());
	   for (unsigned int j=0;j<ref[i].size();j++) ref[i][j]=refine[i][j];
	  }
	  tbself->refine_base_mesh(ref); })
		.def("reorder_nodes", [](pyoomph::Mesh *self, bool old_ordering)
			 { self->reorder_nodes(); })
		.def(
			"get_node_reordering", [](pyoomph::Mesh *self, bool old_ordering)
			{
	    oomph::Vector<oomph::Node*> nodes;
	    self->get_node_reordering(nodes,old_ordering);
	    std::vector<pyoomph::Node*> result(nodes.size());
	    for (unsigned int i=0;i<result.size();i++) result[i]=dynamic_cast<pyoomph::Node*>(nodes[i]);
	    return result; },
			py::return_value_policy::reference)
		.def("evaluate_local_expression_at_nodes", [](pyoomph::Mesh *self, unsigned index, bool nondimensional,bool discontinuous)
			 { return self->evaluate_local_expression_at_nodes(index, nondimensional,discontinuous); })
		.def("to_numpy", [](pyoomph::Mesh *self, bool tesselate_tri, bool nondimensional, unsigned history_index,bool discontinuous)
			 {
				/*std::cout << self << std::endl;
				std::cout << self->communicator_pt() << std::endl;*/
			 unsigned nnode=self->count_nnode(discontinuous);
			 pyoomph::Node* node0=self->get_some_node();
			 unsigned nodal_dim=(node0 ? node0->ndim() : 0);
			 pyoomph::BulkElementBase* be=NULL;
			 if (self->nelement()>0) 
			 {
				be=dynamic_cast<pyoomph::BulkElementBase*>(self->element_pt(0));
			 }
			 #ifdef OOMPH_HAS_MPI
			 else
			 {
			 	int my_rank = self->communicator_pt()->my_rank();
      			int n_proc = self->communicator_pt()->nproc();
				for (int nrnk=0;nrnk<n_proc;nrnk++)
				{
					std::cout << "INFO MY RANK " << my_rank << " NRNK " << nrnk << " NROOT " << self->nroot_haloed_element(nrnk) << std::endl;
					if (nrnk!=my_rank) 
					{
						if (self->nroot_haloed_element(nrnk)>0) 
						{
							be=dynamic_cast<pyoomph::BulkElementBase*>(self->root_haloed_element_pt(nrnk,0));
							break;
						}
					}
				}
				
			 }
			 #endif
			 if (!be) throw std::runtime_error("No elements in mesh. Cannot convert to numpy.");
			 
 			 unsigned nlagrange=(node0 ? node0 ->nlagrangian() : 0);
			 unsigned ncontfields=(be ? be->ncont_interpolated_values() : 0);
			 unsigned nDGfields=(be ? be->num_DG_fields(false) :0);
			 unsigned naddC2TB=(be ? be->nadditional_fields_C2TB() : 0);			 			 
			 unsigned naddC2=(be ? be->nadditional_fields_C2() : 0);
			 unsigned naddC1TB=(be ? be->nadditional_fields_C1TB() : 0);			 			 
			 unsigned naddC1=(be ? be->nadditional_fields_C1() : 0);
			 unsigned nnormal=0;
			 if (be->nodal_dimension()==be->dim()+1) {nnormal=be->nodal_dimension();} //TODO: >= ? But what is a normal of a 1d line in 3d. XXX MAKE SURE TO ADJUST IT ALSO IN Mesh::to_numpy
			 auto nodal_data=py::array_t<double>({nnode,nodal_dim+nlagrange+ncontfields+nDGfields+naddC2TB+naddC2+naddC1TB+naddC1+nnormal});
			 unsigned nelem;
			 unsigned numelem_indices=self->get_num_numpy_elemental_indices(tesselate_tri,nelem,discontinuous);
			 auto elemtypes=py::array_t<int>({nelem});
			 auto elem_node_inds=py::array_t<int>({nelem,numelem_indices});
			 unsigned numD0=be->get_code_instance()->get_func_table()->numfields_D0;
			 unsigned numDL=be->get_code_instance()->get_func_table()->numfields_DL;
			 unsigned DL_stride=(be->dim()+1);
			 auto D0_data=py::array_t<double>({(discontinuous ? nnode : nelem),numD0});
			 py::array_t<double> DL_data;
			 if (discontinuous)
			 {
			   DL_data=py::array_t<double>({nnode,numDL});
			 }
			 else
			 {
			   DL_data=py::array_t<double>({nelem,numDL,DL_stride});			 
			 }
			 self->to_numpy((double*)nodal_data.request().ptr,(int*)elem_node_inds.request().ptr,numelem_indices,(int*)elemtypes.request().ptr,tesselate_tri,nondimensional,(double*)D0_data.request().ptr,(double*)DL_data.request().ptr,history_index,discontinuous);
			 std::map<std::string,unsigned> nodal_field_desc;
			 if (nodal_dim>0) {nodal_field_desc["coordinate_x"]=0; if (nodal_dim>1) {nodal_field_desc["coordinate_y"]=1;} if (nodal_dim>2) {nodal_field_desc["coordinate_z"]=2;}}
			 if (nlagrange>0) {nodal_field_desc["lagrangian_x"]=nodal_dim; if (nlagrange>1) {nodal_field_desc["lagrangian_y"]=nodal_dim+1; if (nlagrange>2) {nodal_field_desc["lagrangian_z"]=nodal_dim+2; }}}
			 auto  nfd=be->get_code_instance()->get_nodal_field_indices();
			 for (auto & nf : nfd)
			 {
				nodal_field_desc[nf.first]=nlagrange+nodal_dim+nf.second;
			 }
			 for (unsigned int nn=0;nn<nnormal;nn++)
			 {
			   const std::vector<std::string> dir{"x","y","z"};
			   nodal_field_desc["normal_"+dir[nn]]=nlagrange+nodal_dim+nfd.size()+nn;
			 }
			 std::map<std::string,unsigned> elemental_field_desc;
			 auto  efd=be->get_code_instance()->get_elemental_field_indices();
			 for (auto & ef : efd)
			 {
				elemental_field_desc[ef.first]=ef.second;
			 }
			 return std::make_tuple(nodal_data,elem_node_inds,elemtypes,nodal_field_desc,D0_data,DL_data,elemental_field_desc); },
			 	py::arg("tesselate_tri"),py::arg("nondimensional"),py::arg("history_index")=0,py::arg("discontinuous")=false)
		.def("get_values_at_zetas", [](pyoomph::Mesh *self, const py::array_t<double> &coords, bool with_scales)
			 {
			py::buffer_info buf = coords.request();
			double *ptr = (double *)buf.ptr;
			size_t N = buf.shape[0];
			size_t D = buf.shape[1];
			std::vector<std::vector<double>> zetas(N,std::vector<double>(D));
			for (unsigned int i=0;i<N;i++) for (unsigned int j=0;j<D;j++) zetas[i][j]=ptr[j + i*D];
			std::vector<bool> masked_lines;
			std::vector<std::vector<double>> values=self->get_values_at_zetas(zetas,masked_lines,with_scales);

		   pyoomph::Node* node0=self->get_some_node();
			unsigned nodal_dim=(node0 ? node0->ndim() : 0);
		   pyoomph::BulkElementBase* el=dynamic_cast<pyoomph::BulkElementBase*>(self->element_pt(0));
			std::map<std::string,unsigned> descs;
			if (nodal_dim>0) {descs["coordinate_x"]=0; if (nodal_dim>1) {descs["coordinate_y"]=1;} if (nodal_dim>2) {descs["coordinate_z"]=2;}}
			auto  nfd=el->get_code_instance()->get_nodal_field_indices();
			unsigned offset=nodal_dim;
			for (auto & nf : nfd)
			{
				descs[nf.first]=nodal_dim+nf.second;
				if (nodal_dim+nf.second>=offset) offset=nodal_dim+nf.second+1;
			}
			auto  efd=el->get_code_instance()->get_elemental_field_indices();
			for (auto & ef : efd)
			{
				descs[ef.first]=offset+ef.second;
			}

			return std::make_tuple(values,masked_lines,descs); })
		.def("describe_global_dofs", [](pyoomph::Mesh *self)
			 {
	 std::vector<int> types;
	 std::vector<std::string> names;
	 self->describe_global_dofs(types,names);
	 return std::make_tuple(types,names); })
		.def("set_output_scale", &pyoomph::Mesh::set_output_scale)
		.def("get_output_scale", &pyoomph::Mesh::get_output_scale)
		.def("get_element_dimension", &pyoomph::Mesh::get_element_dimension)
		.def("set_initial_condition", &pyoomph::Mesh::set_initial_condition, py::keep_alive<1, 3>())
		.def("setup_Dirichlet_conditions", &pyoomph::Mesh::setup_Dirichlet_conditions)
		.def("_set_dirichlet_active", &pyoomph::Mesh::set_dirichlet_active)
		.def("_get_dirichlet_active", &pyoomph::Mesh::get_dirichlet_active)
		.def("setup_initial_conditions", &pyoomph::Mesh::setup_initial_conditions)
		.def("get_boundary_index", &pyoomph::Mesh::get_boundary_index)
		.def("get_boundary_names", &pyoomph::Mesh::get_boundary_names)
		.def("set_spatial_error_estimator_pt", &pyoomph::Mesh::set_spatial_error_estimator_pt, py::keep_alive<1, 2>())
		.def("_enlarge_elemental_error_max_override_to_only_nodal_connected_elems", &pyoomph::Mesh::enlarge_elemental_error_max_override_to_only_nodal_connected_elems)
		.def("adapt_by_elemental_errors", [](pyoomph::Mesh *self, const std::vector<double> &errs)
			 {
 	   oomph::Vector<double> oerrs(errs.size());
 	   for (unsigned int i=0;i<errs.size();i++) oerrs[i]=errs[i];
 	   self->adapt(oerrs); });

	py::class_<pyoomph::BulkElementODE0d, oomph::GeneralisedElement>(m, "BulkElementODE0d")
      .def("_debug",[](pyoomph::BulkElementODE0d * self)
      {
        std::cout << "ODE DEBUG " << dynamic_cast<pyoomph::BulkElementODE0d*>(self) << "  " <<  dynamic_cast<pyoomph::BulkElementBase*>(self) << std::endl;
      }) 
		.def("_debug_hessian", [](pyoomph::BulkElementODE0d *self, std::vector<double> Y, std::vector<std::vector<double>> C, double epsilon)
			 {
			pyoomph::BulkElementODE0d * be=dynamic_cast<pyoomph::BulkElementODE0d*>(self);
			if (!be) { throw_runtime_error("Not a BulkelementBase"); return;	   }
			be->debug_hessian(Y,C,epsilon); })
		.def("ninternal_data", [](pyoomph::BulkElementODE0d *self)
			 { return self->ninternal_data(); })
		.def(
			"internal_data_pt", [](pyoomph::BulkElementODE0d *self, unsigned i)
			{ return self->internal_data_pt(i); },
			py::return_value_policy::reference)
		.def("to_numpy", [](pyoomph::BulkElementODE0d *self)
			 {
			 unsigned ndata=self->get_code_instance()->get_func_table()->numfields_D0;
			 auto data=py::array_t<double>({ndata});
			 self->to_numpy((double*)data.request().ptr);
			 std::map<std::string,unsigned> field_desc;
			 auto  nfd=self->get_code_instance()->get_elemental_field_indices();
			 for (auto & nf : nfd)
			 {
				field_desc[nf.first]=nf.second;
			 }
			 return std::make_tuple(data,field_desc); })
		.def_static("construct_new", &pyoomph::BulkElementODE0d::construct_new, py::return_value_policy::reference)
		.def(py::init<pyoomph::DynamicBulkElementInstance *, oomph::TimeStepper *>()); // Constructor does not work

	py::class_<pyoomph::ODEStorageMesh, pyoomph::Mesh, oomph::Mesh>(m, "ODEStorageMesh")
		.def(py::init<>())
		.def("_set_problem", [](pyoomph::ODEStorageMesh *self, pyoomph::Problem *p, pyoomph::DynamicBulkElementInstance *inst)
			 { self->_set_problem(p, inst); })
		.def(
			"_add_ODE", [](pyoomph::ODEStorageMesh *self, std::string name, pyoomph::BulkElementODE0d *ode)
			{ return self->add_ODE(name, ode); },
			py::keep_alive<1, 3>())
		.def(
			"_get_ODE", [](pyoomph::ODEStorageMesh *self, std::string name)
			{ return dynamic_cast<pyoomph::BulkElementODE0d *>(self->get_ODE(name)); },
			py::return_value_policy::reference);

	py::class_<pyoomph::MeshTemplateElementPoint>(m, "MeshTemplateElementPoint");
	py::class_<pyoomph::MeshTemplateElementLineC1>(m, "MeshTemplateElementLineC1");
	py::class_<pyoomph::MeshTemplateElementLineC2>(m, "MeshTemplateElementLineC2");
	py::class_<pyoomph::MeshTemplateElementQuadC1>(m, "MeshTemplateElementQuadC1");
	py::class_<pyoomph::MeshTemplateElementQuadC2>(m, "MeshTemplateElementQuadC2");
	py::class_<pyoomph::MeshTemplateElementTriC1>(m, "MeshTemplateElementTriC1");
	py::class_<pyoomph::MeshTemplateElementTriC2>(m, "MeshTemplateElementTriC2");
	py::class_<pyoomph::MeshTemplateElementBrickC1>(m, "MeshTemplateElementBrickC1");
	py::class_<pyoomph::MeshTemplateElementBrickC2>(m, "MeshTemplateElementBrickC2");
	py::class_<pyoomph::MeshTemplateElementTetraC1>(m, "MeshTemplateElementTetraC1");
	py::class_<pyoomph::MeshTemplateElementTetraC2>(m, "MeshTemplateElementTetraC2");

	py::class_<pyoomph::MeshTemplateElementCollection>(m, "MeshTemplateElementCollection")
		.def("_get_reference_position_for_IC_and_DBC", &pyoomph::MeshTemplateElementCollection::get_reference_position_for_IC_and_DBC)
		.def("add_point_element", &pyoomph::MeshTemplateElementCollection::add_point_element, py::return_value_policy::reference,"Adds a single point element to the domain")
		.def("add_line_1d_C1", &pyoomph::MeshTemplateElementCollection::add_line_1d_C1, py::return_value_policy::reference,"Adds a line element by two node indices")
		.def("add_line_1d_C2", &pyoomph::MeshTemplateElementCollection::add_line_1d_C2, py::return_value_policy::reference,"Adds a second order line element by three node indices")
		.def("add_quad_2d_C1", &pyoomph::MeshTemplateElementCollection::add_quad_2d_C1, py::return_value_policy::reference,"Adds a quadrilateral element by four node indices")
		.def("add_quad_2d_C2", &pyoomph::MeshTemplateElementCollection::add_quad_2d_C2, py::return_value_policy::reference,"Adds a second-order quadrilateral element by nine node indices")
		.def("add_tri_2d_C1", &pyoomph::MeshTemplateElementCollection::add_tri_2d_C1, py::return_value_policy::reference,"Adds a triangular element by three node indices")
		.def("add_SV_tri_2d_C1", &pyoomph::MeshTemplateElementCollection::add_SV_tri_2d_C1, py::return_value_policy::reference)		
		.def("add_tri_2d_C2", &pyoomph::MeshTemplateElementCollection::add_tri_2d_C2, py::return_value_policy::reference,"Adds a second-order triangular element by six node indices")
		.def("add_brick_3d_C1", &pyoomph::MeshTemplateElementCollection::add_brick_3d_C1, py::return_value_policy::reference,"Adds a hexahedral element by eight node indices")
		.def("add_brick_3d_C2", &pyoomph::MeshTemplateElementCollection::add_brick_3d_C2, py::return_value_policy::reference,"Adds a second-order hexahedral element by 27 node indices")
		.def("add_tetra_3d_C1", &pyoomph::MeshTemplateElementCollection::add_tetra_3d_C1, py::return_value_policy::reference,"Adds a tetrahedral element by four node indices")
		.def("add_tetra_3d_C2", &pyoomph::MeshTemplateElementCollection::add_tetra_3d_C2, py::return_value_policy::reference,"Adds a second-order tetrahedral element by ten node indices")
		.def("nodal_dimension", &pyoomph::MeshTemplateElementCollection::nodal_dimension,"Returns the dimension of the Eulerian coordinates")
		.def("lagrangian_dimension", &pyoomph::MeshTemplateElementCollection::lagrangian_dimension,"Returns the dimension of the Lagrangian coordinates")
		.def("set_nodal_dimension", &pyoomph::MeshTemplateElementCollection::set_nodal_dimension,"Sets the dimension of the Eulerian coordinates")
		.def("set_lagrangian_dimension", &pyoomph::MeshTemplateElementCollection::set_lagrangian_dimension,"Sets the dimension of the Lagrangian coordinates")
		.def("get_element_dimension", &pyoomph::MeshTemplateElementCollection::get_element_dimension)
		.def("get_adjacent_boundary_names", &pyoomph::MeshTemplateElementCollection::get_adjacent_boundary_names)
		.def("set_all_nodes_as_boundary_nodes",&pyoomph::MeshTemplateElementCollection::set_all_nodes_as_boundary_nodes)
		.def("set_element_code", &pyoomph::MeshTemplateElementCollection::set_element_code).
		doc()="A collection of bulk elements, i.e. a bulk domain of a mesh. Must be created as part of a :py:class:`~pyoomph.meshes.mesh.MeshTemplate` by :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.new_domain`";

	py::class_<pyoomph::MeshTemplate, pyoomph::PyMeshTemplateTrampoline>(m, "MeshTemplate")
		.def(py::init<>())
		.def("_set_problem", &pyoomph::MeshTemplate::_set_problem)
		.def("get_node_position", &pyoomph::MeshTemplate::get_node_position)
		.def("new_bulk_element_collection", &pyoomph::MeshTemplate::new_bulk_element_collection, py::return_value_policy::reference)
		.def("add_nodes_to_boundary", &pyoomph::MeshTemplate::add_nodes_to_boundary,"Adds a list of nodes, i.e. a facet, to a boundary")
		.def("add_facet_to_curve_entity", &pyoomph::MeshTemplate::add_facet_to_curve_entity,"Adds a facet to a curved boundary so that e.g. additional nodes of refined meshes will be exactly on this curve")
		.def("_find_opposite_interface_connections", &pyoomph::MeshTemplate::_find_opposite_interface_connections)
		.def("_find_interface_intersections", &pyoomph::MeshTemplate::_find_interface_intersections)
		.def("add_periodic_node_pair", &pyoomph::MeshTemplate::add_periodic_node_pair, "n_mst"_a, "n_slv"_a)
		.def("add_node_unique", &pyoomph::MeshTemplate::add_node_unique, "x"_a, "y"_a = 0.0, "z"_a = 0.0,"Adds a node at the given position. If there is already a node at this position,no new node is created")
		.def("add_node", &pyoomph::MeshTemplate::add_node, "x"_a, "y"_a = 0.0, "z"_a = 0.0,"Adds a node at the given position. Creates overlapping nodes, if there is already a node at this position.")
		.doc()="A base class for the MeshTemplate class in pyoomph";

	py::class_<pyoomph::TemplatedMeshBase1d, pyoomph::Mesh, oomph::Mesh>(m, "TemplatedMeshBase1d")
		.def(py::init<>())
		.def("_set_problem", [](pyoomph::TemplatedMeshBase1d *self, pyoomph::Problem *p, pyoomph::DynamicBulkElementInstance *inst)
			 { self->_set_problem(p, inst); })
		.def(
			"_get_problem", [](pyoomph::TemplatedMeshBase1d *self)
			{ return self->get_problem(); },
			py::return_value_policy::reference)
		.def("refinement_possible", [](pyoomph::TemplatedMeshBase1d *self)
			 { return true; })
		.def(
			"refine_uniformly", [](pyoomph::TemplatedMeshBase1d *self, unsigned int num)
			{for (unsigned int i=0;i<num;i++) self->refine_uniformly(); },
			"num"_a = 1)
		.def("generate_from_template", &pyoomph::TemplatedMeshBase1d::generate_from_template)
		.def("refine_selected_elements", [](pyoomph::TemplatedMeshBase1d *m, std::vector<unsigned int> inds)
			 { oomph::Vector<unsigned> oomphvec(inds.size()); for (unsigned int i=0;i<inds.size();i++) oomphvec[i]=inds[i]; m->refine_selected_elements(oomphvec); })
		.def("setup_boundary_element_info", [](pyoomph::TemplatedMeshBase1d *self)
			 { std::ostringstream oss; self->setup_boundary_element_info(oss); })
		.def("setup_tree_forest", &pyoomph::TemplatedMeshBase1d::setup_tree_forest);

	py::class_<pyoomph::TemplatedMeshBase2d, pyoomph::Mesh, oomph::Mesh>(m, "TemplatedMeshBase2d")
		.def(py::init<>())
		.def("_set_problem", [](pyoomph::TemplatedMeshBase2d *self, pyoomph::Problem *p, pyoomph::DynamicBulkElementInstance *inst)
			 { self->_set_problem(p, inst); })
		.def(
			"_get_problem", [](pyoomph::TemplatedMeshBase2d *self)
			{ return self->get_problem(); },
			py::return_value_policy::reference)
		.def("set_max_neighbour_finding_tolerance", [](pyoomph::TemplatedMeshBase2d *self, double tol)
			 {  oomph::Tree::max_neighbour_finding_tolerance() = tol; })
		.def("generate_from_template", &pyoomph::TemplatedMeshBase2d::generate_from_template)
		.def("add_tri_C1",[](pyoomph::TemplatedMeshBase2d *self,pyoomph::Node *n1,pyoomph::Node *n2,pyoomph::Node *n3){self->add_tri_C1(n1,n2,n3);})
		.def("add_tri_C1TB",[](pyoomph::TemplatedMeshBase2d *self,pyoomph::Node *n1,pyoomph::Node *n2,pyoomph::Node *n3,pyoomph::Node *n4=NULL){self->add_tri_C1TB(n1,n2,n3,n4);})
		.def("refinement_possible", &pyoomph::TemplatedMeshBase2d::refinement_possible)
		.def(
			"refine_uniformly", [](pyoomph::TemplatedMeshBase2d *self, unsigned int num)
			{for (unsigned int i=0;i<num;i++) self->refine_uniformly(); },
			"num"_a = 1)
		.def("setup_tree_forest", &pyoomph::TemplatedMeshBase2d::setup_tree_forest)
		.def("refine_selected_elements", [](pyoomph::TemplatedMeshBase2d *m, std::vector<unsigned int> inds)
			 { oomph::Vector<unsigned> oomphvec(inds.size()); for (unsigned int i=0;i<inds.size();i++) oomphvec[i]=inds[i]; m->refine_selected_elements(oomphvec); })
		.def("setup_boundary_element_info", [](pyoomph::TemplatedMeshBase2d *self)
			 { std::ostringstream oss; self->setup_boundary_element_info(oss); });

	py::class_<pyoomph::TemplatedMeshBase3d, pyoomph::Mesh, oomph::Mesh>(m, "TemplatedMeshBase3d")
		.def(py::init<>())
		.def("_set_problem", [](pyoomph::TemplatedMeshBase3d *self, pyoomph::Problem *p, pyoomph::DynamicBulkElementInstance *inst)
			 { self->_set_problem(p, inst); })
		.def(
			"_get_problem", [](pyoomph::TemplatedMeshBase3d *self)
			{ return self->get_problem(); },
			py::return_value_policy::reference)
		.def("generate_from_template", &pyoomph::TemplatedMeshBase3d::generate_from_template)
		.def("refinement_possible", &pyoomph::TemplatedMeshBase3d::refinement_possible)
		.def(
			"refine_uniformly", [](pyoomph::TemplatedMeshBase3d *self, unsigned int num)
			{for (unsigned int i=0;i<num;i++) self->refine_uniformly(); },
			"num"_a = 1)
		.def("setup_tree_forest", &pyoomph::TemplatedMeshBase3d::setup_tree_forest)
		.def("refine_selected_elements", [](pyoomph::TemplatedMeshBase3d *m, std::vector<unsigned int> inds)
			 { oomph::Vector<unsigned> oomphvec(inds.size()); for (unsigned int i=0;i<inds.size();i++) oomphvec[i]=inds[i]; m->refine_selected_elements(oomphvec); })
		.def("setup_boundary_element_info", [](pyoomph::TemplatedMeshBase3d *self)
			 { std::ostringstream oss; self->setup_boundary_element_info(oss); });

	py::class_<pyoomph::InterfaceMesh, pyoomph::Mesh, oomph::Mesh>(m, "InterfaceMesh")
		.def("clear_before_adapt", &pyoomph::InterfaceMesh::clear_before_adapt)
		.def("nullify_selected_bulk_dofs", &pyoomph::InterfaceMesh::nullify_selected_bulk_dofs)
		.def("_connect_interface_elements_by_kdtree", &pyoomph::InterfaceMesh::connect_interface_elements_by_kdtree)
		.def("rebuild_after_adapt", &pyoomph::InterfaceMesh::rebuild_after_adapt)
		.def("set_opposite_interface_offset_vector",&pyoomph::InterfaceMesh::set_opposite_interface_offset_vector)
		.def("get_opposite_interface_offset_vector",&pyoomph::InterfaceMesh::get_opposite_interface_offset_vector)		
		.def("update_zeta_in_buffer",&pyoomph::InterfaceMesh::update_zeta_in_buffer)
		.def("get_bulk_mesh", &pyoomph::InterfaceMesh::get_bulk_mesh)
		.def(
			"_get_problem", [](pyoomph::InterfaceMesh *self)
			{ return self->get_problem(); },
			py::return_value_policy::reference)
		.def("_set_problem", [](pyoomph::InterfaceMesh *self, pyoomph::Problem *p, pyoomph::DynamicBulkElementInstance *inst)
			 { self->_set_problem(p, inst); })
		//  .def(py::init<pyoomph::Problem*>());
		.def(py::init<>());

	m.def("set_tolerance_for_singular_jacobian", [](double tol)
		  { oomph::FiniteElement::Tolerance_for_singular_jacobian = tol; });
	m.def("set_interpolate_new_interface_dofs", [](bool on)
		  { pyoomph::InterfaceElementBase::interpolate_new_interface_dofs = on; });
	m.def("set_use_eigen_Z2_error_estimators", [](bool on)
		  { pyoomph::BulkElementBase::use_eigen_error_estimators = on; });

	py::class_<pyoomph::TracerCollection>(m, "TracerCollection")
		.def(py::init<std::string>())
		.def("_set_mesh", &pyoomph::TracerCollection::set_mesh)
		.def("_advect_all", &pyoomph::TracerCollection::advect_all)
		.def("_prepare_advection", &pyoomph::TracerCollection::prepare_advection)
		.def("_locate_elements", &pyoomph::TracerCollection::locate_elements)
		.def("_save_state", [](pyoomph::TracerCollection *t)
			 {std::vector<double> pos; std::vector<int> tags; t->_save_state(pos,tags); return std::make_tuple(pos,tags); })
		.def("_load_state", [](pyoomph::TracerCollection *t, std::vector<double> pos, std::vector<int> tags)
			 { t->_load_state(pos, tags); })
		.def("_set_transfer_interface", &pyoomph::TracerCollection::set_transfer_interface)
		.def(
			"add_tracer", [](pyoomph::TracerCollection *coll, const std::vector<double> &pos, int tag = 0)
			{ coll->add_tracer(pos, tag); },
			py::arg("position"), py::arg("tag") = 0)
		.def("get_positions", [](pyoomph::TracerCollection *coll)
			 {
    unsigned nd=coll->get_coordinate_dimension();
    if (!nd) { return py::array_t<double>({0}); }
    std::vector<double> pos=coll->get_positions();
    auto data=py::array_t<double>({(unsigned)(pos.size()/nd),nd});
	 double * dest=(double*)data.request().ptr;
	 for (unsigned int i=0;i<pos.size();i++) dest[i]=pos[i];
	 return data; });
	 
	 
	 delete py_decl_OomphData;
	 delete py_decl_OomphMesh;
	 delete py_decl_PyoomphMesh;
	 delete py_decl_GeneralisedElement;
}
