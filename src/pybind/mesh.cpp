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

	// Base class for a curved boundary/interface entity that can be subclassed from Python
	// (see pyoomph::MeshTemplateCurvedEntity in ../meshtemplate.hpp). The C++ side works with
	// oomph::Vector<double>, but Python users typically want to work with numpy arrays, so this
	// class converts between std::vector<double> (used by parametric_to_position/position_to_parametric,
	// which is what the C++ core calls) and py::array_t<double> (used by the virtual functions
	// parametric_to_pos/pos_to_parametric/ensure_periodicity, which are the ones a Python subclass overrides).
	class PyMeshTemplateCurvedEntity : public MeshTemplateCurvedEntity
	{
	public:
		using MeshTemplateCurvedEntity::MeshTemplateCurvedEntity;
		// Maps a parametric coordinate (e.g. arc length) to a Cartesian position. To be overridden in Python.
		virtual void parametric_to_pos(const unsigned &t, const py::array_t<double> &param, py::array_t<double> &pos)
		{
			throw_runtime_error("parametric_to_pos not specialised");
		}
		// Maps a Cartesian position back to a parametric coordinate. To be overridden in Python.
		virtual void pos_to_parametric(const unsigned &t, const py::array_t<double> &pos, py::array_t<double> &param)
		{
			throw_runtime_error("pos_to_parametric not specialised");
		}
		// Converts the std::vector<double> position used on the C++ side into a numpy array before
		// delegating to the (potentially Python-overridden) parametric_to_pos.
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
		// Converts the std::vector<double> position used on the C++ side into a numpy array before
		// delegating to the (potentially Python-overridden) pos_to_parametric.
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

		// Adjusts a batch of parametric coordinates so that they lie within a canonical periodic range
		// (relevant e.g. for closed curves where the parametrisation wraps around). To be overridden in Python.
		virtual void ensure_periodicity(py::array_t<double> &parametrics)
		{
		}

		// Converts a batch of parametric coordinates into a 2d numpy array, calls the (potentially
		// Python-overridden) ensure_periodicity, and writes the result back.
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

	// pybind11 trampoline forwarding the virtual functions of PyMeshTemplateCurvedEntity to
	// Python method overrides (this is what actually gets registered as the base class for
	// "MeshTemplateCurvedEntity" exposed to Python).
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

	// pybind11 trampoline allowing Python subclasses of MeshTemplate (the base class used to define
	// a mesh geometry, see pyoomph.meshes.mesh.MeshTemplate) to override the connection-finding logic
	// for opposite interfaces and the reset routine used when the mesh is rebuilt.
	class PyMeshTemplateTrampoline : public MeshTemplate
	{
	public:
		using MeshTemplate::MeshTemplate;
		void _add_opposite_interface_connection(const std::string &sideA, const std::string &sideB) override
		{
			PYBIND11_OVERLOAD(void, MeshTemplate, _add_opposite_interface_connection, sideA, sideB);
		}
		void reset() override
		{
			PYBIND11_OVERLOAD(void, MeshTemplate, reset );
		}
	};

	

}

using namespace pybind11::literals;

static py::class_<oomph::Data> *py_decl_OomphData = NULL;
static py::class_<oomph::Mesh> *py_decl_OomphMesh = NULL;
static py::class_<pyoomph::Mesh, oomph::Mesh> *py_decl_PyoomphMesh = NULL;
static py::class_<oomph::GeneralisedElement> * py_decl_GeneralisedElement =NULL;
// Pre-declares the classes that are mutually referenced further down (e.g. Node derives from
// OomphData, and several methods on OomphGeneralisedElement/OomphMesh return Node/Mesh pointers),
// so that these forward declarations can be filled in with their methods later in PyReg_Mesh.
void PyDecl_Mesh(py::module &m)
{
	py_decl_OomphData = new py::class_<oomph::Data>(m, "OomphData", "Base class of oomph-lib for a container of nodal/internal/external double values that can be pinned (Dirichlet-constrained) or free (unknowns of the linear system)");
	py_decl_OomphMesh = new py::class_<oomph::Mesh>(m, "OomphMesh", "Base class of oomph-lib for a mesh, i.e. a collection of elements and nodes");
	py_decl_PyoomphMesh = new py::class_<pyoomph::Mesh, oomph::Mesh>(m, "Mesh", "pyoomph's extended mesh class, adding e.g. boundary handling, error estimation control and (de)serialization on top of oomph-lib's OomphMesh");
	py_decl_GeneralisedElement=new py::class_<oomph::GeneralisedElement>(m, "OomphGeneralisedElement", "Base class of oomph-lib for any element (bulk, interface, ODE, ...) that contributes degrees of freedom and residuals/Jacobians to a problem");
}

void PyReg_Mesh(py::module &m)
{

	py::class_<pyoomph::MeshTemplateCurvedEntity>(m, "MeshTemplateCurvedEntityBase")
		.def_static("load_from_strings", &pyoomph::MeshTemplateCurvedEntity::load_from_strings)
		.def("get_information_string", &pyoomph::MeshTemplateCurvedEntity::get_information_string)
		.def("get_pos_from_parametric", &pyoomph::MeshTemplateCurvedEntity::parametric_to_position)
		.def("get_parametric_from_pos", &pyoomph::MeshTemplateCurvedEntity::position_to_parametric)
		.doc()="A generic class representing a relation for a curved boundary representation";

	py::class_<pyoomph::CurvedEntityCircleArc, pyoomph::MeshTemplateCurvedEntity>(m, "CurvedEntityCircleArc", "A curved entity representing an arc of a circle, defined by its center, a point on the circle and a tangent direction")
		.def(py::init<const std::vector<double> &, const std::vector<double> &, const std::vector<double> &>(), py::arg("center"), py::arg("point_on_circle"), py::arg("tangent"));

	py::class_<pyoomph::CurvedEntityCylinderArc, pyoomph::MeshTemplateCurvedEntity>(m, "CurvedEntityCylinderArc", "A curved entity representing an arc on the mantle of a cylinder, defined by its axis, a point on the cylinder and a tangent direction")
		.def(py::init<const std::vector<double> &, const std::vector<double> &, const std::vector<double> &>(), py::arg("axis"), py::arg("point_on_cylinder"), py::arg("tangent"));

	py::class_<pyoomph::CurvedEntityCatmullRomSpline, pyoomph::MeshTemplateCurvedEntity>(m, "CurvedEntityCatmullRomSpline", "A curved entity interpolating a given set of points by a Catmull-Rom spline")
		.def(py::init<const std::vector<std::vector<double>> &>(), py::arg("points"));

	py::class_<pyoomph::CurvedEntitySpherePart, pyoomph::MeshTemplateCurvedEntity>(m, "CurvedEntitySpherePart", "A curved entity representing a part of a sphere, defined by its center, a point on the sphere and a tangent direction")
		.def(py::init<const std::vector<double> &, const std::vector<double> &, const std::vector<double> &>(), py::arg("center"), py::arg("point_on_sphere"), py::arg("tangent")); // center,onsphere, tangent

	// Trampoline-backed base class allowing users to implement custom curved boundary shapes directly in Python
	// by subclassing MeshTemplateCurvedEntity and overriding pos_to_parametric/parametric_to_pos/ensure_periodicity.
	py::class_<pyoomph::PyMeshTemplateCurvedEntity, pyoomph::PyMeshTemplateCurvedEntityTrampoline, pyoomph::MeshTemplateCurvedEntity>(m, "MeshTemplateCurvedEntity", "Base class to implement a custom curved boundary representation in Python")
		.def(py::init<unsigned>(), py::arg("num_parameters"))
      .def("pos_to_parametric",&pyoomph::PyMeshTemplateCurvedEntity::pos_to_parametric, py::arg("t"),py::arg("pos"),py::arg("param"), "Maps a Cartesian position to a parametric coordinate. To be implemented by the Python subclass")
      .def("parametric_to_pos",&pyoomph::PyMeshTemplateCurvedEntity::parametric_to_pos,py::arg("t"),py::arg("param"),py::arg("pos"), "Maps a parametric coordinate to a Cartesian position. To be implemented by the Python subclass")
      .def("ensure_periodicity",&pyoomph::PyMeshTemplateCurvedEntity::ensure_periodicity,py::arg("param"), "Adjusts a batch of parametric coordinates to lie within a canonical periodic range. Can be implemented by the Python subclass if the parametrisation is periodic");



	py::class_<pyoomph::LagrZ2ErrorEstimator>(m, "Z2ErrorEstimator", "Zienkiewicz-Zhu spatial error estimator used to drive adaptive mesh refinement")
		.def(py::init<>())
		.def_readwrite("use_Lagrangian", &pyoomph::LagrZ2ErrorEstimator::use_Lagrangian, "If True, the error is estimated based on the Lagrangian (undeformed) coordinates instead of the Eulerian (current) ones");

	py_decl_OomphData->def("set_time_stepper", &oomph::Data::set_time_stepper, py::arg("time_stepper"), py::arg("preserve_existing_data"), "Assigns a time stepper to this data object, allocating the required storage for the time history of its values")
		.def("pin", &oomph::Data::pin, py::arg("value_index"), "Pins (Dirichlet-constrains) the given value index, i.e. removes it from the unknowns of the linear system")
		.def("unpin", &oomph::Data::unpin, py::arg("value_index"), "Unpins the given value index, i.e. makes it an unknown of the linear system again")
		.def("is_pinned", &oomph::Data::is_pinned, py::arg("value_index"), "Returns whether the given value index is pinned (Dirichlet-constrained)")
		.def("eqn_number", (long(oomph::Data::*)(const unsigned &) const) & oomph::Data::eqn_number, py::arg("value_index"), "Returns the global equation number associated with the given value index, or a negative number if it is pinned")
		.def("nvalue", (unsigned(oomph::Data::*)() const) & oomph::Data::nvalue, "Returns the number of values stored in this data object")
		.def("value", (double(oomph::Data::*)(unsigned const &) const) & oomph::Data::value, py::arg("value_index"), "Returns the current value at the given value index")
		.def("value_at_t", (double(oomph::Data::*)(unsigned const &, unsigned const &) const) & oomph::Data::value, py::arg("history_index"), py::arg("value_index"), "Returns the value at the given value index at a previous time level (history_index, with 0 being the current time)")
		.def("ntstorage", &oomph::Data::ntstorage, "Returns the number of history/time levels stored for this data object")
		.def("set_value", (void(oomph::Data::*)(unsigned const &, double const &)) & oomph::Data::set_value, py::arg("value_index"), py::arg("value"), "Sets the current value at the given value index")
		.def("set_value_at_t", (void(oomph::Data::*)(unsigned const &, unsigned const &, double const &)) & oomph::Data::set_value, py::arg("history_index"), py::arg("value_index"), py::arg("value"), "Sets the value at the given value index at a previous time level (history_index, with 0 being the current time)");

	py::class_<pyoomph::Node, oomph::Data>(m, "Node", "A mesh node, i.e. a point carrying an Eulerian (and possibly a Lagrangian) position plus the nodal values (degrees of freedom) living there")
		.def("x", (const double &(pyoomph::Node::*)(const unsigned int &) const) & pyoomph::Node::x, py::arg("coordinate_index"), "Returns the current Eulerian coordinate at the given coordinate index")
		.def("x_at_t", (const double &(pyoomph::Node::*)(const unsigned int &,const unsigned int &) const) & pyoomph::Node::x, py::arg("history_index"), py::arg("coordinate_index"), "Returns the Eulerian coordinate at the given coordinate index at a previous time level")
		.def("x_lagr", (const double &(pyoomph::Node::*)(const unsigned int &) const) & pyoomph::Node::xi, py::arg("coordinate_index"), "Returns the Lagrangian (undeformed/reference) coordinate at the given coordinate index")
		.def("ndim", (unsigned(pyoomph::Node::*)() const) & pyoomph::Node::ndim, "Returns the number of Eulerian coordinate dimensions of this node")
		.def("set_x", [](pyoomph::Node *n, unsigned const &ind, double const &x)
			 { n->x(ind) = x; }, py::arg("coordinate_index"), py::arg("value"), "Sets the current Eulerian coordinate at the given coordinate index")
		.def("set_x_at_t", [](pyoomph::Node *n, unsigned const &t,unsigned const &ind, double const &x)
			 { n->x(t,ind) = x; }, py::arg("history_index"), py::arg("coordinate_index"), py::arg("value"), "Sets the Eulerian coordinate at the given coordinate index at a previous time level")
		.def("set_x_lagr", [](pyoomph::Node *n, unsigned const &ind, double const &x)
			 { n->xi(ind) = x; }, py::arg("coordinate_index"), py::arg("value"), "Sets the Lagrangian (undeformed/reference) coordinate at the given coordinate index")
		.def("pin_position", (void(pyoomph::Node::*)(const unsigned &)) & pyoomph::Node::pin_position, py::arg("coordinate_index"), "Pins the position (i.e. removes it from the unknowns) at the given coordinate index, relevant for moving meshes")
		.def("unpin_position", (void(pyoomph::Node::*)(const unsigned &)) & pyoomph::Node::unpin_position, py::arg("coordinate_index"), "Unpins the position at the given coordinate index, i.e. makes it a free unknown again, relevant for moving meshes")
		.def("position_is_pinned", (bool(pyoomph::Node::*)(const unsigned &)) & pyoomph::Node::position_is_pinned, py::arg("coordinate_index"), "Returns whether the position at the given coordinate index is pinned")
		.def("is_hanging", (bool(pyoomph::Node::*)(const int &) const) & pyoomph::Node::is_hanging, "index"_a = -1, "Returns whether this node is a hanging node (i.e. constrained by other nodes due to non-conforming mesh refinement); index=-1 checks all values, otherwise only the given value index")
		.def("variable_position_pt", &pyoomph::Node::variable_position_pt, py::return_value_policy::reference, "Returns the underlying oomph-lib Data object storing the (potentially pinned) nodal position")
		.def("is_on_boundary", (bool(pyoomph::Node::*)() const) & pyoomph::Node::is_on_boundary, "Returns whether this node lies on any mesh boundary")
		.def("set_obsolete", &pyoomph::Node::set_obsolete, "Marks this node as obsolete, e.g. after adaptive refinement/unrefinement, so it can be pruned later")
		.def("is_obsolete", &pyoomph::Node::is_obsolete, "Returns whether this node has been marked as obsolete")
		.def("remove_from_boundary",&pyoomph::Node::remove_from_boundary, py::arg("boundary_index"), "Removes this node from the given mesh boundary")
		.def("add_to_boundary", &pyoomph::Node::add_to_boundary, py::arg("boundary_index"), "Adds this node to the given mesh boundary")
		//	.def("is_on_boundary_index",(bool (pyoomph::Node::*,const unsigned )() const) &pyoomph::Node::is_on_boundary)
		.def("get_boundary_indices", [](pyoomph::Node *n)
			 {std::set<unsigned> *pt; n->get_boundaries_pt(pt); std::set<unsigned> res; if (pt) {for (auto i : *pt) res.insert(i);}  return res; }, "Returns the set of indices of all mesh boundaries this node is part of")
		.def("additional_value_index", &pyoomph::Node::additional_value_index, py::arg("interface_id"), "Returns the value index of the additional (interface-only) values associated with the given interface id stored on this node, or -1 if not present")
		.def("set_coordinates_on_boundary",[](pyoomph::Node *self,unsigned boundary_index, std::vector<double> &zeta) {
			oomph::Vector<double> zeta_prime(zeta.size()); for(unsigned int i=0;i<zeta.size();i++){zeta_prime[i]=zeta[i];};
			self->set_coordinates_on_boundary(boundary_index, zeta_prime);
		}, py::arg("boundary_index"), py::arg("zeta"), "Sets the intrinsic boundary coordinate(s) zeta of this node on the given mesh boundary")
		.def("get_coordinates_on_boundary",[](pyoomph::Node *self,unsigned boundary_index) {
			oomph::Vector<double> zeta_prime(self->ncoordinates_on_boundary(boundary_index),0);
			self->get_coordinates_on_boundary(boundary_index, zeta_prime);
			std::vector<double> zeta(zeta_prime.size(),0);
			for(unsigned int i=0;i<zeta.size();i++){zeta[i]=zeta_prime[i];};
			return zeta;
		}, py::arg("boundary_index"), "Returns the intrinsic boundary coordinate(s) zeta of this node on the given mesh boundary")
		// Ties a slave node to a master node so they share the same degrees of freedom (used for periodic boundaries).
		// Resolves any pre-existing copy-master relations first and raises an error if that leads to an inconsistency.
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
	    mesh->store_copy_master(slv,mst); }, py::arg("master"), py::arg("mesh"))
		// Currently disabled: was meant to suppress a node's residual contribution for a specific dof index.
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
		.def("is_on_boundary", (bool(pyoomph::Node::*)(unsigned const &) const) & pyoomph::Node::is_on_boundary, py::arg("boundary_index"), "Returns whether this node lies on the given mesh boundary");

	// oomph::GeneralisedElement is oomph-lib's most generic element base class. Most of the methods
	// exposed here are only meaningful for pyoomph::BulkElementBase (the actual base of all
	// finite elements generated from pyoomph equations), so nearly every binding below does a
	// dynamic_cast to BulkElementBase first and returns a harmless fallback (0/-1/empty container)
	// if the element is of another kind (e.g. a plain oomph-lib element or an ODE element).
	auto &decl_GeneralisedElement = (*py_decl_GeneralisedElement);
	decl_GeneralisedElement
		.def("_debug_hessian", [](oomph::GeneralisedElement *self, std::vector<double> Y, std::vector<std::vector<double>> C, double epsilon)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) { throw_runtime_error("Not a BulkelementBase"); return;	   }
			be->debug_hessian(Y,C,epsilon); }, py::arg("Y"), py::arg("C"), py::arg("epsilon"), "Debugging helper comparing the analytically assembled Hessian against a finite-difference approximation")
		.def("get_meshio_type_index", [](oomph::GeneralisedElement *self) -> unsigned int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return -1;
			return be->get_meshio_type_index(); }, "Returns the meshio/VTK cell type index of this element, used when exporting the mesh")
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

		   },"Assembles the (dense) rank-3 Hessian tensor of the residuals with respect to the dofs, and the corresponding mass-matrix Hessian, both as numpy arrays"
		  )
		.def("refinement_level", [](oomph::GeneralisedElement *self) -> unsigned int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0;
			return be->refinement_level(); }, "Returns the refinement level of this element in the adaptive mesh refinement tree (0 for the coarsest/un-refined elements)")
		.def("ncont_interpolated_values", [](oomph::GeneralisedElement *self) -> unsigned int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0;
			return be->ncont_interpolated_values(); }, "Returns the number of continuous (nodally interpolated) fields on this element")
		.def("non_halo_proc_ID", [](oomph::GeneralisedElement *self) -> int
			 {
			#ifdef OOMPH_HAS_MPI
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return -1;
			return be->non_halo_proc_ID();
			#else
			return -1;
			#endif
			}, "In parallel (MPI) runs, returns the rank of the processor that actually owns this element if it is a halo copy, or -1 if it is not a halo element or MPI is disabled")

		.def("set_must_be_kept_as_halo", [](oomph::GeneralisedElement *self,bool halo)
			 {
			#ifdef OOMPH_HAS_MPI
			if (halo) self->set_must_be_kept_as_halo();
			else self->unset_must_be_kept_as_halo();
			#endif
			}, py::arg("keep_as_halo"), "In parallel (MPI) runs, marks (or unmarks) this element so it is kept as a halo element even if it would otherwise be pruned")
		.def("must_be_kept_as_halo", [](oomph::GeneralisedElement *self) -> bool
			 {
			#ifdef OOMPH_HAS_MPI
			return self->must_be_kept_as_halo();
			#else
			return false;
			#endif
			}, "Returns whether this element is marked to be kept as a halo element even if it would otherwise be pruned")
		.def_property_readonly("_cpp_ptr_address", [](const oomph::GeneralisedElement & self) {
            return reinterpret_cast<uintptr_t>(&self);
        }, "Returns the raw memory address of the underlying C++ object, used internally to identify/hash elements from Python")
		.def("describe_my_dofs", [](oomph::GeneralisedElement *self, std::string in)
			 {
	   std::ostringstream oss;
   	pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	   if (be) be->describe_my_dofs(std::cout,in);
	   return oss.str(); }, py::arg("indent")="", "Returns a human-readable description of all degrees of freedom of this element, useful for debugging")
		.def("get_nodal_index_by_name", [](oomph::GeneralisedElement *self, pyoomph::Node *n, std::string name) -> int
			 {
	   std::ostringstream oss;
   	pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	   if (!be) return -1;
	   return be->get_nodal_index_by_name(n,name); }, py::arg("node"), py::arg("field_name"), "Returns the nodal value index of the given field name at the given node of this element, or -1 if not present")
		.def(
			"get_code_instance", [](oomph::GeneralisedElement *self) -> pyoomph::DynamicBulkElementInstance *
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return NULL;
			return be->get_code_instance(); },
			py::return_value_policy::reference, "Returns the compiled equation/code instance (generated from the Python equation definitions) associated with this element")
		.def("get_macro_element_coordinate_at_s",[](oomph::GeneralisedElement *self, std::vector<double> s) -> std::vector<double>
		   {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return {};
			oomph::Vector<double> so(s.size()); for (unsigned int i=0;i<s.size();i++) so[i]=s[i];
			return be->get_macro_element_coordinate_at_s(so);
		   }, py::arg("s"), "Returns the position given by the element's macro element mapping at the local coordinate s (used e.g. for curved/undeformed geometries)")
		.def("evaluate_local_expression_at_s", [](oomph::GeneralisedElement *self, int index, std::vector<double> s) -> double
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0.0;
			oomph::Vector<double> so(s.size()); for (unsigned int i=0;i<s.size();i++) so[i]=s[i];
			return be->eval_local_expression_at_s(index,so); }, py::arg("expression_index"), py::arg("s"), "Evaluates a local (element-defined) expression, identified by its index, at the given local coordinate s")
		.def("evaluate_local_expression_at_midpoint", [](oomph::GeneralisedElement *self, int index) -> double
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0.0;
			return be->eval_local_expression_at_midpoint(index); }, py::arg("expression_index"), "Evaluates a local (element-defined) expression, identified by its index, at the element's midpoint")
		.def("evaluate_local_expression_at_node_index", [](oomph::GeneralisedElement *self, int index, unsigned node_index) -> double
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0.0;
			return be->eval_local_expression_at_node(index,node_index); }, py::arg("expression_index"), py::arg("node_index"), "Evaluates a local (element-defined) expression, identified by its index, at the given local node index")
		.def(
			"node_pt", [](oomph::GeneralisedElement *self, unsigned int i) -> pyoomph::Node *
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return NULL;
			return dynamic_cast<pyoomph::Node*>(be->node_pt(i)); },
			py::return_value_policy::reference, py::arg("local_node_index"), "Returns the node at the given local node index of this element")
		.def("nodes",[](oomph::GeneralisedElement *self)
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			std::vector<pyoomph::Node*> nodes;
			if (be)
			{
				for (unsigned int i=0;i<be->nnode();i++) nodes.push_back(dynamic_cast<pyoomph::Node*>(be->node_pt(i)));
			}
			return nodes;
			},py::return_value_policy::reference, "Returns the list of all nodes of this element")
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
			},py::return_value_policy::reference,py::arg("boundary_index")=-1, "Returns the list of nodes on the given mesh boundary (or all nodes if boundary_index is negative)")
		.def("boundary_vertex_nodes",[](oomph::GeneralisedElement *self,int boundary_index)
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			std::vector<pyoomph::Node*> nodes;
			if (be)
			{
				for (unsigned int i=0;i<be->nvertex_node();i++) if (be->vertex_node_pt(i)->is_on_boundary(boundary_index)) nodes.push_back(dynamic_cast<pyoomph::Node*>(be->vertex_node_pt(i)));
			}
			return nodes;
			},py::return_value_policy::reference, py::arg("boundary_index"), "Returns the list of vertex (corner) nodes of this element that lie on the given mesh boundary")
		.def("_connect_periodic_tree",[](oomph::GeneralisedElement *self,oomph::GeneralisedElement *other, int mydir, int otherdir)
			{
				dynamic_cast<pyoomph::BulkElementBase*>(self)->connect_periodic_tree(dynamic_cast<pyoomph::BulkElementBase*>(other),mydir,otherdir);
			}, py::arg("other_element"), py::arg("my_direction"), py::arg("other_direction"), "Connects the refinement trees of this element and another element along the given directions, used to set up periodic boundary conditions on refineable meshes")
		//Returns oomph::Data and value indices for a fields. If use_elemental_indices, it will return (NULL,-1) for elemental node indices that do not have data associated
		.def("get_field_data_list",[](oomph::GeneralisedElement *self, std::string name,bool use_elemental_indices) -> std::vector<std::pair<oomph::Data *,int> >
		   {
			 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			 if (!be) return {};
			 return be->get_field_data_list(name,use_elemental_indices);
		   },py::return_value_policy::reference, py::arg("field_name"), py::arg("use_elemental_indices"), "Returns the list of (Data object, value index) pairs storing the given field for every node/dof of this element")
		.def(
			"opposite_node_pt", [](oomph::GeneralisedElement *self, unsigned int i) -> pyoomph::Node *
			{
			pyoomph::InterfaceElementBase * ie=dynamic_cast<pyoomph::InterfaceElementBase*>(self);
			if (!ie) return NULL;
			return ie->opposite_node_pt(i); },
			py::return_value_policy::reference, py::arg("local_node_index"), "For an interface element, returns the corresponding node on the opposite side of the interface at the given local node index")
		.def("get_attached_element_equation_mapping", [](oomph::GeneralisedElement *self, std::string which)
		    {
			   pyoomph::InterfaceElementBase * ie=dynamic_cast<pyoomph::InterfaceElementBase*>(self);
			   if (!ie) { throw_runtime_error("Not an interface element"); }
			   return ie->get_attached_element_equation_mapping(which);
		    }, py::arg("which")
		    )
		.def("set_opposite_interface_element", [](oomph::GeneralisedElement *self, oomph::GeneralisedElement *opp,std::vector<double> offs)
			 {
			pyoomph::InterfaceElementBase * ie=dynamic_cast<pyoomph::InterfaceElementBase*>(self);
			pyoomph::InterfaceElementBase * io=dynamic_cast<pyoomph::InterfaceElementBase*>(opp);
			if (!ie || !io) { throw_runtime_error("Can only connect interface elements this way"); }
			ie->set_opposite_interface_element(io,offs); }, py::arg("opposite_element"), py::arg("offset"), "Connects this interface element to the corresponding element on the opposite side of the interface (e.g. for two-sided interfaces), with an optional coordinate offset")
		.def(
			"vertex_node_pt", [](oomph::GeneralisedElement *self, unsigned int i) -> pyoomph::Node *
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return NULL;
			return dynamic_cast<pyoomph::Node*>(be->vertex_node_pt(i)); },
			py::return_value_policy::reference, py::arg("local_vertex_index"), "Returns the vertex (corner) node at the given local vertex index of this element")
		.def("ndof", [](oomph::GeneralisedElement *self) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) {
			  pyoomph::BulkElementODE0d * ode=dynamic_cast<pyoomph::BulkElementODE0d*>(self);
			  if (ode) return ode->ndof();
			  else return self->ndof();
			}
			return be->ndof(); }, "Returns the number of degrees of freedom (unknowns) of this element")
		.def("eqn_number", [](oomph::GeneralisedElement *self, unsigned int i) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return -1;
			return be->eqn_number(i); }, py::arg("local_dof_index"), "Returns the global equation number of the given local degree of freedom of this element, or -1 if it is pinned")
		.def("nvertex_node", [](oomph::GeneralisedElement *self) -> unsigned
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0;
			return be->nvertex_node(); }, "Returns the number of vertex (corner) nodes of this element")
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
			}, "Per-element override of the maximum permitted spatial error used to drive adaptive mesh refinement; a value of 0 means no override")
		.def("num_Z2_flux_terms", [](oomph::GeneralisedElement *self) -> unsigned int
			 {
	  	pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		if (!be) return 0;
		else return be->num_Z2_flux_terms(); }, "Returns the number of flux terms used by the Zienkiewicz-Zhu (Z2) spatial error estimator on this element")
		.def(
			"get_bulk_element", [](oomph::GeneralisedElement *self)
			{
	    pyoomph::InterfaceElementBase * ie=dynamic_cast<pyoomph::InterfaceElementBase*>(self);
	    if (!ie) return (oomph::GeneralisedElement*)NULL;
	    else return dynamic_cast<oomph::GeneralisedElement*>(ie->bulk_element_pt()); },
			py::return_value_policy::reference, "For an interface element, returns the adjacent bulk element it is attached to")
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
			py::return_value_policy::reference, "For an interface element connected to an opposite interface, returns the bulk element attached to that opposite interface element")
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
			py::return_value_policy::reference, "For an interface element connected to an opposite interface, returns the opposite interface element itself")
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
			return data; },py::arg("lagrangian")=false, "Returns the coordinates of the outline (boundary polygon) of this element as a numpy array, either in Eulerian or (if lagrangian=True) Lagrangian coordinates")
		.def("get_debug_jacobian_info", [](oomph::GeneralisedElement *self)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			oomph::Vector<double> R;
			oomph::DenseMatrix<double> J;
			std::vector<std::string> dofnames;
			be->get_debug_jacobian_info(R,J,dofnames);
			std::vector<double> Rv(R.size()); for (unsigned int i=0;i<R.size();i++) Rv[i]=R[i];
			std::vector<double> Jv(J.ncol()*J.nrow()); for (unsigned int i=0;i<J.nrow();i++) for (unsigned int j=0;j<J.ncol();j++) Jv[J.ncol()*i+j]=J(i,j);
			return std::make_tuple(Rv,Jv,dofnames); }, "Returns the residual vector, (dense, row-major flattened) Jacobian matrix and the names of the degrees of freedom of this element, useful for debugging")
		.def("nnode", [](oomph::GeneralisedElement *self) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0;
			return be->nnode(); }, "Returns the number of nodes of this element")
		.def("get_quality_factor", [](oomph::GeneralisedElement *self) -> double
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 1.0;
			return be->get_quality_factor(); }, "Returns a measure of the geometric quality (shape regularity) of this element, used e.g. to trigger remeshing")
		.def("get_initial_cartesian_nondim_size", [](oomph::GeneralisedElement *self) -> double
			 {
	  	 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		 if (!be) return 0.0;
		 return be->initial_cartesian_nondim_size; }, "Returns the non-dimensional element size recorded at the time the element was created")
		.def("set_initial_cartesian_nondim_size", [](oomph::GeneralisedElement *self, double s)
			 {
	  	 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		 return be->initial_cartesian_nondim_size=s; }, py::arg("size"), "Sets the non-dimensional element size recorded at the time the element was created")
		.def("get_initial_quality_factor", [](oomph::GeneralisedElement *self) -> double
			 {
	  	 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		 if (!be) return 1.0;
		 return be->initial_quality_factor; }, "Returns the geometric quality factor recorded at the time the element was created")
		.def("set_initial_quality_factor", [](oomph::GeneralisedElement *self, double s)
			 {
	  	 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		 return be->initial_quality_factor=s; }, py::arg("quality_factor"), "Sets the geometric quality factor recorded at the time the element was created")
		.def("get_Eulerian_midpoint", [](oomph::GeneralisedElement *self)
			 {
	  pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	  std::vector<double> res;
	  if (!be) return res;
	  oomph::Vector<double> ores=be->get_Eulerian_midpoint_from_local_coordinate();
	  res.resize(ores.size()); for (unsigned int i=0;i<ores.size();i++) res[i]=ores[i];
	  return res; }, "Returns the Eulerian (current) position of the element's midpoint (local coordinate s=0)")
		.def("get_Lagrangian_midpoint", [](oomph::GeneralisedElement *self)
			 {
	  pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	  std::vector<double> res;
	  if (!be) return res;
	  oomph::Vector<double> ores=be->get_Lagrangian_midpoint_from_local_coordinate();
	  res.resize(ores.size()); for (unsigned int i=0;i<ores.size();i++) res[i]=ores[i];
	  return res; }, "Returns the Lagrangian (undeformed/reference) position of the element's midpoint (local coordinate s=0)")
		.def("get_current_cartesian_nondim_size", [](oomph::GeneralisedElement *self) -> double
			 {
	  	 pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
		 if (!be) return 0.0;
		 return be->size(); }, "Returns the current non-dimensional size (length/area/volume) of this element")
		.def("nnode_1d", [](oomph::GeneralisedElement *self) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0;
			return be->nnode_1d(); }, "Returns the number of nodes along one edge of this element")
		.def(
			"boundary_node_pt", [](oomph::GeneralisedElement *self, int dir, int index) -> pyoomph::Node *
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return 0;
			return dynamic_cast<pyoomph::Node*>(be->boundary_node_pt(dir,index)); },
			py::return_value_policy::reference, py::arg("direction"), py::arg("index"), "Returns the node at the given index along the given local edge/face direction of this element")
		.def("_ode_elem_to_numpy", [](oomph::GeneralisedElement *self)
			 {

			 pyoomph::BulkElementODE0d * ode=dynamic_cast<pyoomph::BulkElementODE0d *>(self);
			 if (!ode) { throw_runtime_error("Not an ODE element"); }
			 unsigned ndata=ode->get_code_instance()->get_func_table()->info_D0.numfields;
			 auto data=py::array_t<double>({ndata});
			 ode->to_numpy((double*)data.request().ptr);
			 std::map<std::string,unsigned> field_desc;
			 auto  nfd=ode->get_code_instance()->get_elemental_field_indices();
			 for (auto & nf : nfd)
			 {
				field_desc[nf.first]=nf.second;
			 }
			 return std::make_tuple(data,field_desc); }, "For an ODE (0-dimensional) element, returns its current field values as a numpy array plus a dict mapping field names to indices in that array")

		.def("ninternal_data", [](oomph::GeneralisedElement *self) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) {
				pyoomph::BulkElementODE0d * ode=dynamic_cast<pyoomph::BulkElementODE0d *>(self);
				if (ode) return ode->ninternal_data(); else return self->ninternal_data();
			}
			return be->ninternal_data(); }, "Returns the number of internal Data objects (dofs not tied to a node) of this element")
		.def("nexternal_data", [](oomph::GeneralisedElement *self) -> int
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) {
				pyoomph::BulkElementODE0d * ode=dynamic_cast<pyoomph::BulkElementODE0d *>(self);
				if (ode) return ode->ninternal_data(); else return self->nexternal_data();
			}
			return be->nexternal_data(); }, "Returns the number of external Data objects (dofs owned by other elements that this element also depends on) of this element")
		.def("get_dof_names", [](oomph::GeneralisedElement *self)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return std::vector<std::string>();
			else return be->get_dof_names(); }, "Returns the names of all degrees of freedom (fields) of this element")
		.def(
			"get_father_element", [](oomph::GeneralisedElement *self) -> oomph::GeneralisedElement *
			{
				pyoomph::BulkElementBase *be = dynamic_cast<pyoomph::BulkElementBase *>(self);
				if (!be)
					return NULL;
				return dynamic_cast<pyoomph::BulkElementBase *>(be->father_element_pt()); },
			py::return_value_policy::reference, "Returns the father element in the adaptive mesh refinement tree, i.e. the (coarser) element this one was refined from, or None for a root element")
		.def(
			"get_macro_element", [](oomph::GeneralisedElement *self) -> oomph::MacroElement *
			{
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return NULL;
			return be->macro_elem_pt(); },
			py::return_value_policy::reference, "Returns the macro element used to define this element's (potentially curved) undeformed geometry, if any")
		.def("set_macro_element", [](oomph::GeneralisedElement *self, oomph::MacroElement *m, bool map_nodes)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return ;
			be->set_macro_elem_pt(m);
			if (map_nodes) be->map_nodes_on_macro_element(); }, py::arg("macro_element"), py::arg("map_nodes"), "Assigns a macro element to define this element's undeformed geometry; if map_nodes is True, the nodal positions are immediately updated from the macro element mapping")
		.def("create_interpolated_node",[](oomph::GeneralisedElement *self, const std::vector<double> & s,bool as_boundary_node) -> pyoomph::Node *
		   {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return NULL;
			oomph::Vector<double> soomph(s.size());
			for (unsigned int i=0;i<s.size();i++) soomph[i]=s[i];
			return be->create_interpolated_node(soomph,as_boundary_node);
		   },py::return_value_policy::reference, py::arg("s"), py::arg("as_boundary_node"), "Creates a new node at the given local coordinate s of this element by interpolation, optionally marking it as a boundary node")
		.def("local_coordinate_of_node", [](oomph::GeneralisedElement *self, unsigned int l) -> std::vector<double>
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return std::vector<double>();
			oomph::Vector<double> s;
			be->local_coordinate_of_node(l, s);
			std::vector<double> res(s.size());
			for (unsigned int i=0;i<s.size();i++) res[i]=s[i];
			return res; }, py::arg("local_node_index"), "Returns the local coordinate s of the node at the given local node index of this element")
		.def("set_undeformed_macro_element", [](oomph::GeneralisedElement *self, oomph::MacroElement *m)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return ;
			be->set_undeformed_macro_elem_pt(m); }, py::arg("macro_element"), "Assigns a macro element defining this element's undeformed (reference) geometry")
		.def("map_nodes_on_macro_element", [](oomph::GeneralisedElement *self)
			 {
			pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
			if (!be) return ;
			be->map_nodes_on_macro_element(); }, "Updates the positions of all nodes of this element according to the assigned macro element mapping")
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
  	  return res; }, py::arg("zeta"), py::arg("s"), py::arg("use_coordinate_as_initial_guess"), "Locally inverts the position mapping: given a target Eulerian coordinate zeta, finds the local coordinate s within this element that maps to it (empty result if zeta is not inside this element)")
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
	  return res; }, py::arg("history_index"), py::arg("s"), "Returns the interpolated nodal field values at the given local coordinate s, at a given history/time level (0 being the current time)")
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
	  return res; }, py::arg("history_index"), py::arg("s"), py::arg("lagrangian"), "Returns the interpolated position at the given local coordinate s, either in Eulerian (at the given history/time level) or, if lagrangian=True, in Lagrangian coordinates")
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
	  return res; }, py::arg("history_index"), py::arg("s"), "Returns the interpolated discontinuous (element-local, non-nodally-shared) field values at the given local coordinate s, at a given history/time level")
		.def("dim", [](oomph::GeneralisedElement *self) -> int
			 {
		  pyoomph::BulkElementBase * be=dynamic_cast<pyoomph::BulkElementBase*>(self);
	     if (!be) return -1;
	     return be->dim(); }, "Returns the spatial dimension of this element (e.g. 1 for a line, 2 for a surface)")
		.def("external_data_pt", (oomph::Data * &(oomph::GeneralisedElement::*)(const unsigned &)) & oomph::GeneralisedElement::external_data_pt, py::return_value_policy::reference, py::arg("external_data_index"), "Returns the external Data object (a dof owned by another element that this element also depends on) at the given index")
		.def("internal_data_pt", (oomph::Data * &(oomph::GeneralisedElement::*)(const unsigned &)) & oomph::GeneralisedElement::internal_data_pt, py::return_value_policy::reference, py::arg("internal_data_index"), "Returns the internal Data object (a dof not tied to a node) at the given index");

	// oomph::Mesh is oomph-lib's base mesh class, exposed to Python mostly so that methods which are
	// only defined there (and not overridden/extended by pyoomph::Mesh) remain accessible. Most
	// pyoomph meshes are actually pyoomph::Mesh instances (see decl_PyoomphMesh below), which derives
	// from this class.
	auto &decl_OomphMesh = (*py_decl_OomphMesh);
	decl_OomphMesh
		.def(
			"as_pyoomph_mesh", [](oomph::Mesh *self)
			{ return dynamic_cast<pyoomph::Mesh *>(self); },
			py::return_value_policy::reference, "Downcasts this mesh to a pyoomph Mesh, if it is one (returns None otherwise)")
		.def("add_node_to_mesh",[](oomph::Mesh *self,pyoomph::Node *n)
			 {
				self->add_node_pt(n);
			 }, py::arg("node"), "Adds an already-constructed node to this mesh's list of nodes")
		.def("prune_dead_nodes",[](oomph::Mesh *self,bool with_bounds)
			{
				if (with_bounds) self->prune_dead_nodes();
				else dynamic_cast<pyoomph::TemplatedMeshBase*>(self)->prune_dead_nodes_without_respecting_boundaries();
			}, py::arg("with_bounds"), "Removes nodes marked as obsolete (e.g. after adaptive unrefinement) from the mesh; if with_bounds is False, boundary membership bookkeeping is skipped")
		.def("output_paraview", [](oomph::Mesh *self, const std::string &fname, const unsigned &order)
			 { std::ofstream f(fname); self->output_paraview(f,order); }, py::arg("filename"), py::arg("order"), "Writes this mesh to a ParaView-compatible file at the given polynomial output order")
		.def("nelement", &oomph::Mesh::nelement, "Returns the number of elements in this mesh")
		.def(
			"element_pt", [](oomph::Mesh &self, const unsigned &ei) -> oomph::GeneralisedElement *
			{ if (ei>=self.nelement()) return (pyoomph::BulkElementBase *)(NULL); else return  dynamic_cast<pyoomph::BulkElementBase *>(self.element_pt(ei)); },
			py::return_value_policy::reference, py::arg("element_index"), "Returns the element at the given index in this mesh")
		.def(
			"boundary_element_pt", [](oomph::Mesh &self, const unsigned &bi, const unsigned &ei) -> oomph::GeneralisedElement *
			{ return dynamic_cast<pyoomph::BulkElementBase *>(self.boundary_element_pt(bi, ei)); },
			py::return_value_policy::reference, py::arg("boundary_index"), py::arg("element_index"), "Returns the element at the given index among the elements adjacent to the given mesh boundary")
		.def("face_index_at_boundary", [](oomph::Mesh &self, const unsigned &bi, const unsigned &ei)
			 { return self.face_index_at_boundary(bi, ei); }, py::arg("boundary_index"), py::arg("element_index"), "Returns the local face/edge index at which the given boundary element touches the given mesh boundary")
		.def("nboundary", &oomph::Mesh::nboundary, "Returns the number of boundaries of this mesh")
		.def("nboundary_node", &oomph::Mesh::nboundary_node, py::arg("boundary_index"), "Returns the number of nodes on the given mesh boundary")
		.def("nboundary_element", &oomph::Mesh::nboundary_element, py::arg("boundary_index"), "Returns the number of elements adjacent to the given mesh boundary")
		.def("resolve_copy_master_node", [](oomph::Mesh *self, pyoomph::Node *n)->pyoomph::Node *
			 {
				if (!dynamic_cast<pyoomph::Mesh*>(self)) return NULL;
				return dynamic_cast<pyoomph::Node *>(dynamic_cast<pyoomph::Mesh*>(self)->resolve_copy_master(n));
			},py::return_value_policy::reference, py::arg("node")
			, "If the given node is a copy (e.g. due to periodic boundaries), returns its master node; otherwise returns None"
			)
		.def("_disable_adaptation", [](oomph::Mesh *self)
			 {
    oomph::RefineableMeshBase* refmesh =dynamic_cast<oomph::RefineableMeshBase*>(self);
    if (refmesh) refmesh->disable_adaptation(); }, "Disables adaptive mesh refinement/unrefinement for this mesh, if it is refineable")
		.def("_enable_adaptation", [](oomph::Mesh *self)
			 {
    oomph::RefineableMeshBase* refmesh =dynamic_cast<oomph::RefineableMeshBase*>(self);
    if (refmesh) refmesh->enable_adaptation(); }, "Enables adaptive mesh refinement/unrefinement for this mesh, if it is refineable")
		.def("nnode", &oomph::Mesh::nnode, "Returns the number of nodes in this mesh")
		.def("_set_interpolate_lagrangian_on_remeshing", [](oomph::Mesh *self, bool lagr)
			 {
			pyoomph::Mesh* mesh =dynamic_cast<pyoomph::Mesh*>(self);
			if (mesh) mesh->interpolated_lagrangian_coordinates_at_remeshing=lagr;
			 }, py::arg("interpolate"), "Controls whether the Lagrangian (undeformed) coordinates of new nodes are interpolated from the old mesh when remeshing")
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
			return res; }, "Returns the per-element spatial error estimate (as used by adaptive mesh refinement) for every element in this mesh; zero everywhere if adaptation is disabled")
		.def(
			"node_pt", [](oomph::Mesh *self, unsigned int i) -> pyoomph::Node *
			{ return dynamic_cast<pyoomph::Node *>(self->node_pt(i)); },
			py::return_value_policy::reference, py::arg("node_index"), "Returns the node at the given index in this mesh")
		.def(
			"boundary_node_pt", [](oomph::Mesh &self, const unsigned &b, const unsigned &n)
			{ return dynamic_cast<pyoomph::Node *>(self.boundary_node_pt(b, n)); },
			py::return_value_policy::reference, py::arg("boundary_index"), py::arg("node_index"), "Returns the node at the given index among the nodes on the given mesh boundary");

	// pyoomph's own mesh class, adding boundary/interface handling, adaptive refinement control,
	// (de)serialization of the mesh state and various evaluation helpers on top of oomph::Mesh.
	// This is the class that Python-level mesh objects (bulk domains, interfaces, ODE storage, ...) ultimately derive from.
	auto &decl_PyoomphMesh = (*py_decl_PyoomphMesh);
	decl_PyoomphMesh
		.def("check_integrity",&pyoomph::Mesh::check_integrity, "Runs internal consistency checks on this mesh, raising an error if a problem is found")
		.def("prepare_zeta_interpolation", [](pyoomph::Mesh *self, pyoomph::Mesh *old_mesh){self->prepare_zeta_interpolation(old_mesh);}, py::arg("old_mesh"), "Prepares the internal data structures required to interpolate field values from an old mesh onto this (typically newly adapted) mesh")
		.def("remove_boundary_nodes",[](pyoomph::Mesh *self) {self->remove_boundary_nodes();}, "Removes all nodes from all mesh boundaries (i.e. clears the boundary node lookup, without deleting the nodes themselves)")
		.def("remove_boundary_nodes_of_bound",[](pyoomph::Mesh *self,unsigned b) {self->remove_boundary_nodes(b);}, py::arg("boundary_index"), "Removes all nodes from the given mesh boundary (without deleting the nodes themselves)")
		.def("add_interpolated_nodes_at",&pyoomph::Mesh::add_interpolated_nodes_at,py::return_value_policy::reference, "Creates and adds new nodes interpolated at the given positions, used e.g. when refining or remeshing")
		.def("add_boundary_node",[](pyoomph::Mesh *self,unsigned bind,pyoomph::Node *n) {self->add_boundary_node(bind,n);}, py::arg("boundary_index"), py::arg("node"), "Adds the given node to the given mesh boundary")
		.def("flush_element_storage", [](pyoomph::Mesh *self){self->flush_element_storage();}, "Clears this mesh's list of elements without deleting them (ownership is assumed to be transferred elsewhere)")
		.def("_set_time_level_for_projection", [](pyoomph::Mesh *self, unsigned time_level){self->set_time_level_for_projection(time_level);}, py::arg("time_level"), "Sets the history/time level that is used as the source when projecting field values onto a new mesh")
		.def("get_field_information", [](pyoomph::Mesh *self)
			 { return self->get_field_information(); }, "Returns a description of all fields (nodal, discontinuous, elemental) available on this mesh")
		.def("describe_my_dofs", [](pyoomph::Mesh *self, std::string in)
			 {std::ostringstream oss;self->describe_my_dofs(oss,in);return oss.str(); }, py::arg("indent")="", "Returns a human-readable description of all degrees of freedom of this mesh, useful for debugging")
		.def("_pin_all_my_dofs", [](pyoomph::Mesh *self, std::set<std::string> only_dofs, std::set<std::string> ignore_dofs, std::set<unsigned> ignore_continuous_at_interfaces)
			 { self->pin_all_my_dofs(only_dofs, ignore_dofs, ignore_continuous_at_interfaces); }, py::arg("only_dofs"), py::arg("ignore_dofs"), py::arg("ignore_continuous_at_interfaces"), "Pins (Dirichlet-constrains) all degrees of freedom of this mesh, optionally restricted to only_dofs or excluding ignore_dofs/ignore_continuous_at_interfaces")
		.def("generate_interface_elements", [](pyoomph::Mesh *m, const std::string &bn, pyoomph::Mesh *im, pyoomph::DynamicBulkElementInstance *jitcode)
			 { m->generate_interface_elements(bn, im, jitcode); }, py::arg("boundary_name"), py::arg("interface_mesh"), py::arg("code_instance"), "Creates interface elements attached to the given boundary of this (bulk) mesh, using the given compiled equation code, and adds them to the given interface mesh")
		.def("is_mesh_distributed", [](pyoomph::Mesh *m)
			 { return m->is_mesh_distributed(); }, "Returns whether this mesh is distributed across multiple MPI processes")
		.def("_save_state", [](pyoomph::Mesh *m)
			 {std::vector<double> data; m->_save_state(data); return data; }, "Serializes the current state of this mesh (nodal positions/values, refinement pattern, ...) into a flat list of doubles, for checkpointing")
		.def("_setup_information_from_old_mesh", &pyoomph::Mesh::_setup_information_from_old_mesh, py::arg("old_mesh"), "Prepares this (new) mesh to inherit information (e.g. for state restoration) from an old mesh")
		.def("_load_state", &pyoomph::Mesh::_load_state, py::arg("data"), "Restores the mesh state previously serialized by _save_state")
		.def("_pin_noncontributing_dofs", &pyoomph::Mesh::pin_noncontributing_dofs, "Pins all degrees of freedom on this mesh that do not actually contribute to any residual (e.g. unused higher-order nodal values), to avoid singular Jacobians")
		.def("has_interface_dof_id", &pyoomph::Mesh::has_interface_dof_id, py::arg("name"), "Returns the index of the interface degree of freedom with the given name on this mesh, or -1 if not present")
		.def("list_integral_functions", &pyoomph::Mesh::list_integral_functions, "Returns the names of all integral (domain-averaged/summed) observable functions defined on this mesh")
		.def("list_local_expressions", &pyoomph::Mesh::list_local_expressions, "Returns the names of all local (per-element/per-node) observable expressions defined on this mesh")
		.def("prepare_interpolation", &pyoomph::Mesh::prepare_interpolation, "Builds the internal spatial search structures (e.g. a kd-tree) required for interpolating field values on this mesh")
		.def("nodal_interpolate_from", &pyoomph::Mesh::nodal_interpolate_from, py::arg("old_mesh"), py::arg("boundary_index")=-1, "Interpolates all nodal field values of this mesh from the given old mesh (e.g. after adaptive remeshing), optionally restricted to a single boundary")
		.def("nodal_interpolate_along_boundary", &pyoomph::Mesh::nodal_interpolate_along_boundary, py::arg("old_mesh"), py::arg("boundary_index"), py::arg("old_boundary_index"), py::arg("interface_mesh"), py::arg("old_interface_mesh"), py::arg("boundary_max_dist"), "Interpolates the nodal field values of the nodes on a boundary of this mesh from the corresponding boundary of an old mesh, using the interface meshes to also interpolate interface-only fields; boundary_max_dist limits how far a node may be from the boundary line/surface to still be considered")
		.def("_evaluate_integral_function", [](pyoomph::Mesh *m, const std::string &n)
			 { return m->evaluate_integral_function(n); }, py::arg("name"), "Evaluates the integral observable function with the given name over this mesh")
		.def("_evaluate_extremum", [](pyoomph::Mesh *m, const std::string &n,int sign,unsigned flags)
			 {
				pyoomph::BulkElementBase * extreme_element;
			    oomph::Vector<double> extreme_local_coords;
				GiNaC::ex resval=m->evaluate_extremum(n,sign,extreme_element,extreme_local_coords,flags);
				std::vector<double> s(extreme_local_coords.size());
				for (unsigned i=0;i<extreme_local_coords.size();i++) s[i]=extreme_local_coords[i];
				//return std::make_tuple(resval,s,std::unique_ptr<oomph::GeneralisedElement,py::nodelete>(dynamic_cast<oomph::GeneralisedElement*>(extreme_element)));
				return std::make_tuple(resval,s,dynamic_cast<oomph::GeneralisedElement*>(extreme_element));

			 },py::return_value_policy::reference, py::arg("name"), py::arg("sign"), py::arg("flags"), "Finds the extremum (minimum if sign<0, maximum if sign>0) of a local expression with the given name over this mesh; returns its value, the local coordinate and the element where it was found")
		.def("ensure_external_data", [](pyoomph::Mesh *m)
			 { m->ensure_external_data(); }, "Ensures that all external Data dependencies (dofs owned by other elements/meshes) required by this mesh's elements are properly registered")
		.def("ensure_halos_for_periodic_boundaries", [](pyoomph::Mesh *m)
			 { m->ensure_halos_for_periodic_boundaries(); }, "In parallel (MPI) runs, ensures that the required halo elements/nodes are present so periodic boundary conditions also work across processor boundaries")
		.def("nroot_halo_element", [](pyoomph::Mesh *m)
			 {
				#ifdef OOMPH_HAS_MPI
					return m->nroot_halo_element();
				#else
					return 0;
				#endif
			 }, "In parallel (MPI) runs, returns the number of root (non-refined) halo elements of this mesh"
			)
		.def("nrefined", [](pyoomph::Mesh *m)
			 { return m->nrefined(); }, "Returns the number of elements that were refined during the last adaptation step")
		.def("nunrefined", [](pyoomph::Mesh *m)
			 { return m->nunrefined(); }, "Returns the number of elements that were unrefined during the last adaptation step")
		.def("invalidate_lagrangian_kdtree", [](pyoomph::Mesh *m)
			 { m->invalidate_lagrangian_kdtree(); }, "Invalidates the cached kd-tree used for Lagrangian-coordinate spatial searches, forcing it to be rebuilt on next use")
		.def_property(
			"min_permitted_error", [](pyoomph::Mesh *m)
			{ return m->min_permitted_error(); },
			[](pyoomph::Mesh *m, double e)
			{ m->min_permitted_error() = e; }, "Lower threshold of the per-element spatial error estimate below which elements are unrefined during adaptation")
		.def_property(
			"max_permitted_error", [](pyoomph::Mesh *m)
			{ return m->max_permitted_error(); },
			[](pyoomph::Mesh *m, double e)
			{ m->max_permitted_error() = e; }, "Upper threshold of the per-element spatial error estimate above which elements are refined during adaptation")
		.def_property(
			"max_refinement_level", [](pyoomph::Mesh *m)
			{ return dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->max_refinement_level(); },
			[](pyoomph::Mesh *m, unsigned l)
			{ dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->max_refinement_level() = l; }, "Maximum refinement level (tree depth) that adaptive mesh refinement is allowed to reach on this mesh")
		.def_property(
			"min_refinement_level", [](pyoomph::Mesh *m)
			{ return dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->min_refinement_level(); },
			[](pyoomph::Mesh *m, unsigned l)
			{ dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->min_refinement_level() = l; }, "Minimum refinement level (tree depth) that adaptive mesh unrefinement is allowed to go below on this mesh")
		.def_property(
			"max_keep_unrefined", [](pyoomph::Mesh *m)
			{ return dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->max_keep_unrefined(); },
			[](pyoomph::Mesh *m, unsigned l)
			{ dynamic_cast<oomph::TreeBasedRefineableMeshBase *>(m)->max_keep_unrefined() = l; }, "Number of adaptation cycles for which an element flagged for unrefinement is kept before it is actually unrefined")
		.def("boundary_coordinate_bool", &pyoomph::Mesh::boundary_coordinates_bool, py::arg("boundary_index"), "Returns whether an intrinsic boundary coordinate has been set up for the given mesh boundary")
		.def("is_boundary_coordinate_defined", &pyoomph::Mesh::is_boundary_coordinate_defined, py::arg("boundary_index"), "Returns whether an intrinsic boundary coordinate is defined and up to date for the given mesh boundary")
		.def("fill_node_index_to_node_map",[](pyoomph::Mesh *self)
			{std::map<oomph::Node *, unsigned> node2index;
			self->fill_node_map(node2index);
			std::vector<pyoomph::Node *> index2node(node2index.size(),NULL);;
			for (auto & n2i : node2index){index2node[n2i.second]=dynamic_cast<pyoomph::Node*>(n2i.first);}
			return index2node;
			}, py::return_value_policy::reference, "Returns the list of all nodes of this mesh, ordered consistently with the internal node-to-index map")
		.def("setup_interior_boundary_elements", [](pyoomph::Mesh *self, unsigned bindex)
			 {
	 pyoomph::TemplatedMeshBase * templmesh=dynamic_cast<pyoomph::TemplatedMeshBase*>(self);
	 if (!templmesh) return;
	 templmesh->setup_interior_boundary_elements(bindex); }, py::arg("boundary_index"), "Sets up the boundary-element lookup for an interior (non-outer) boundary of a templated mesh")
		.def("fill_dof_types", [](pyoomph::Mesh *self, py::array_t<int> &desc)
			 { self->fill_dof_types((int *)desc.request().ptr); }, py::arg("dof_type_array"), "Fills the given numpy int array with a type tag for every degree of freedom of this mesh (used e.g. by block preconditioners)")
		.def("set_lagrangian_nodal_coordinates", &pyoomph::Mesh::set_lagrangian_nodal_coordinates, "Sets the Lagrangian (undeformed/reference) coordinates of all nodes to their current Eulerian coordinates")
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
	  return result; }, "Returns the refinement pattern of this (tree-based, refineable) mesh, i.e. for every refinement level the list of which base-mesh elements were split, allowing the pattern to be replayed via refine_base_mesh")
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
	  tbself->refine_base_mesh(ref); }, py::arg("refinement_pattern"), "Replays a refinement pattern (as returned by get_refinement_pattern) on the base mesh, e.g. to restore a checkpointed mesh")
		.def("reorder_nodes", [](pyoomph::Mesh *self, bool old_ordering)
			 { self->reorder_nodes(); }, py::arg("old_ordering"), "Reorders the internal node storage of this mesh (e.g. to improve cache locality/bandwidth of the resulting linear system)")
		.def(
			"get_node_reordering", [](pyoomph::Mesh *self, bool old_ordering)
			{
	    oomph::Vector<oomph::Node*> nodes;
	    self->get_node_reordering(nodes,old_ordering);
	    std::vector<pyoomph::Node*> result(nodes.size());
	    for (unsigned int i=0;i<result.size();i++) result[i]=dynamic_cast<pyoomph::Node*>(nodes[i]);
	    return result; },
			py::return_value_policy::reference, py::arg("old_ordering"), "Returns the nodes of this mesh in the (potential) reordering used by reorder_nodes, without actually reordering the internal storage")
		.def("evaluate_local_expression_at_nodes", [](pyoomph::Mesh *self, unsigned index, bool nondimensional,bool discontinuous)
			 { return self->evaluate_local_expression_at_nodes(index, nondimensional,discontinuous); }, py::arg("expression_index"), py::arg("nondimensional"), py::arg("discontinuous"), "Evaluates a local (element-defined) expression, identified by its index, at every node of this mesh")
		// Exports the entire mesh (nodal positions/fields, elemental connectivity, discontinuous and elemental
		// fields) into a set of numpy arrays in one go, for efficient plotting/post-processing from Python.
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
			 unsigned nadd_interf=0;
			 auto *ft=be->get_code_instance()->get_func_table();
			 for (unsigned int si=0;si<ft->num_present_continuous_spaces;si++)
			 {
				nadd_interf+=ft->present_continuous_spaces[si]->numfields-ft->present_continuous_spaces[si]->numfields_basebulk;
			 }			 
			 unsigned nnormal=0;
			 if (be->nodal_dimension()==be->dim()+1 || dynamic_cast<pyoomph::InterfaceMesh *>(self)) {nnormal=be->nodal_dimension();} //TODO: >= ? But what is a normal of a 1d line in 3d. XXX MAKE SURE TO ADJUST IT ALSO IN Mesh::to_numpy
			 auto nodal_data=py::array_t<double>({nnode,nodal_dim+nlagrange+ncontfields+nDGfields+nadd_interf+nnormal});
			 unsigned nelem;
			 unsigned numelem_indices=self->get_num_numpy_elemental_indices(tesselate_tri,nelem,discontinuous);
			 auto elemtypes=py::array_t<int>({nelem});
			 auto elem_node_inds=py::array_t<int>({nelem,numelem_indices});
			 unsigned numD0=be->get_code_instance()->get_func_table()->info_D0.numfields;
			 unsigned numDL=be->get_code_instance()->get_func_table()->info_DL.numfields;
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
			 	py::arg("tesselate_tri"),py::arg("nondimensional"),py::arg("history_index")=0,py::arg("discontinuous")=false,
				"Exports this mesh's nodal positions/fields, element connectivity, discontinuous and elemental fields as numpy arrays, "
				"together with dicts describing which array column corresponds to which field. tesselate_tri splits e.g. quads into "
				"triangles for plotting, nondimensional selects dimensional or non-dimensional output, history_index the time level, "
				"and discontinuous whether discontinuous fields are stored per-node (interpolated) or per-element")
		// Interpolates field values at a given batch of intrinsic ("zeta") coordinates, e.g. for sampling along a line.
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

			return std::make_tuple(values,masked_lines,descs); }, py::arg("coords"), py::arg("with_scales"), "Interpolates all field values of this mesh at the given (N, dim) numpy array of intrinsic coordinates; returns the values, a mask of coordinates that could not be located, and a dict describing which field corresponds to which value column")
		.def("describe_global_dofs", [](pyoomph::Mesh *self)
			 {
	 std::vector<int> types;
	 std::vector<std::string> names;
	 self->describe_global_dofs(types,names);
	 return std::make_tuple(types,names); }, "Returns, for every global degree of freedom associated with this mesh, its dof type and a human-readable name")
		.def("set_output_scale", &pyoomph::Mesh::set_output_scale, py::arg("name"), py::arg("scale"), py::arg("code_instance"), "Sets an output (dimensional rescaling) factor for the field of the given name, used e.g. when exporting to numpy/VTK")
		.def("get_output_scale", &pyoomph::Mesh::get_output_scale, py::arg("name"), "Returns the output (dimensional rescaling) factor previously set for the field of the given name")
		.def("get_element_dimension", &pyoomph::Mesh::get_element_dimension, "Returns the spatial dimension of the elements of this mesh")
		.def("set_initial_condition", &pyoomph::Mesh::set_initial_condition, py::keep_alive<1, 3>(), py::arg("fieldname"), py::arg("expression"), "Sets the initial condition expression for the given field on this mesh")
		.def("setup_Dirichlet_conditions", &pyoomph::Mesh::setup_Dirichlet_conditions, py::arg("only_update_values"), "(Re-)applies all Dirichlet boundary conditions defined on this mesh; if only_update_values is True, the set of pinned dofs is left unchanged and only their prescribed values are updated")
		.def("_set_dirichlet_active", &pyoomph::Mesh::set_dirichlet_active, py::arg("name"), py::arg("active"), "Enables or disables the named Dirichlet boundary condition on this mesh")
		.def("_get_dirichlet_active", &pyoomph::Mesh::get_dirichlet_active, py::arg("name"), "Returns whether the named Dirichlet boundary condition is currently active on this mesh")
		.def("setup_initial_conditions", &pyoomph::Mesh::setup_initial_conditions, py::arg("resetting_first_step"), py::arg("ic_name"), "Applies the initial condition(s) with the given name to this mesh; resetting_first_step indicates whether this is done for the very first time step")
		.def("get_boundary_index", &pyoomph::Mesh::get_boundary_index, py::arg("boundary_name"), "Returns the numeric index of the mesh boundary with the given name")
		.def("get_boundary_names", &pyoomph::Mesh::get_boundary_names, "Returns the names of all boundaries of this mesh, ordered by their numeric index")
		.def("set_spatial_error_estimator_pt", &pyoomph::Mesh::set_spatial_error_estimator_pt, py::keep_alive<1, 2>(), py::arg("error_estimator"), "Assigns the Z2ErrorEstimator used to drive adaptive mesh refinement on this mesh")
		.def("_enlarge_elemental_error_max_override_to_only_nodal_connected_elems", &pyoomph::Mesh::enlarge_elemental_error_max_override_to_only_nodal_connected_elems, py::arg("boundary_index"), "Propagates the per-element maximum-error override of elements on the given boundary to all elements sharing a node with them")
		.def("adapt_by_elemental_errors", [](pyoomph::Mesh *self, const std::vector<double> &errs)
			 {
 	   oomph::Vector<double> oerrs(errs.size());
 	   for (unsigned int i=0;i<errs.size();i++) oerrs[i]=errs[i];
 	   self->adapt(oerrs); }, py::arg("elemental_errors"), "Adapts (refines/unrefines) this mesh according to the given per-element spatial error estimate");

	   /*
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
*/
	// A special mesh consisting of a single ODE (0-dimensional) element, used to store the degrees
	// of freedom of ODE-based equations (e.g. global ODEs not tied to any spatial domain).
	py::class_<pyoomph::ODEStorageMesh, pyoomph::Mesh, oomph::Mesh>(m, "ODEStorageMesh", "A mesh holding a single ODE element, used to store degrees of freedom that are not associated with a spatial mesh")
		.def(py::init<>())
		.def("_set_problem", [](pyoomph::ODEStorageMesh *self, pyoomph::Problem *p, pyoomph::DynamicBulkElementInstance *inst)
			 { self->_set_problem(p, inst); }, py::arg("problem"), py::arg("code_instance"))
		.def("_create_ode_element", [](pyoomph::ODEStorageMesh *self, oomph::TimeStepper *ts)	->  oomph::GeneralisedElement *
			 {   oomph::GeneralisedElement * res=self->_create_ode_element(ts);
				return dynamic_cast<pyoomph::BulkElementBase *>(res);
			}, py::return_value_policy::reference, py::arg("time_stepper"), "Creates (and stores) the single ODE element of this mesh, using the given time stepper"
			)	;
		/*.def(
			"_add_ODE", [](pyoomph::ODEStorageMesh *self, std::string name, pyoomph::BulkElementODE0d *ode)

			{ return self->add_ODE(name, ode); },
			py::keep_alive<1, 3>())*/
		/*.def(
			"_get_ODE", [](pyoomph::ODEStorageMesh *self, std::string name)
			{ return dynamic_cast<pyoomph::BulkElementODE0d *>(self->get_ODE(name)); },
			py::return_value_policy::reference);*/


	py::class_<pyoomph::MeshTemplateElementCollection>(m, "MeshTemplateElementCollection")
		.def("_get_reference_position_for_IC_and_DBC", &pyoomph::MeshTemplateElementCollection::get_reference_position_for_IC_and_DBC, py::arg("boundary_indices"), "Returns a representative position on the given boundaries, used to evaluate initial conditions/Dirichlet boundary conditions that are only prescribed pointwise")
		.def("add_point_element", &pyoomph::MeshTemplateElementCollection::add_point_element,"Adds a single point element to the domain")
		.def("add_line_1d_C1", &pyoomph::MeshTemplateElementCollection::add_line_1d_C1,"Adds a line element by two node indices")
		.def("add_line_1d_C2", &pyoomph::MeshTemplateElementCollection::add_line_1d_C2,"Adds a second order line element by three node indices")
		.def("add_quad_2d_C1", &pyoomph::MeshTemplateElementCollection::add_quad_2d_C1,"Adds a quadrilateral element by four node indices")
		.def("add_quad_2d_C2", &pyoomph::MeshTemplateElementCollection::add_quad_2d_C2,"Adds a second-order quadrilateral element by nine node indices")
		.def("add_tri_2d_C1", &pyoomph::MeshTemplateElementCollection::add_tri_2d_C1, "Adds a triangular element by three node indices")
		.def("add_SV_tri_2d_C1", &pyoomph::MeshTemplateElementCollection::add_SV_tri_2d_C1, "Adds a Scott-Vogelius triangular element (discontinuous pressure) by three node indices")
		.def("add_tri_2d_C2", &pyoomph::MeshTemplateElementCollection::add_tri_2d_C2,"Adds a second-order triangular element by six node indices")
		.def("add_brick_3d_C1", &pyoomph::MeshTemplateElementCollection::add_brick_3d_C1,"Adds a hexahedral element by eight node indices")
		.def("add_brick_3d_C2", &pyoomph::MeshTemplateElementCollection::add_brick_3d_C2,"Adds a second-order hexahedral element by 27 node indices")
		.def("add_tetra_3d_C1", &pyoomph::MeshTemplateElementCollection::add_tetra_3d_C1,"Adds a tetrahedral element by four node indices")
		.def("add_tetra_3d_C2", &pyoomph::MeshTemplateElementCollection::add_tetra_3d_C2,"Adds a second-order tetrahedral element by ten node indices")
		.def("add_wedge_3d_C1", &pyoomph::MeshTemplateElementCollection::add_wedge_3d_C1,"Adds a wedge element by six node indices")
		.def("add_wedge_3d_C2", &pyoomph::MeshTemplateElementCollection::add_wedge_3d_C2,"Adds a second-order wedge element by eighteen node indices")
		.def("add_pyramid_3d_C1", &pyoomph::MeshTemplateElementCollection::add_pyramid_3d_C1,"Adds a pyramid element by five node indices")
		.def("add_pyramid_3d_C2", &pyoomph::MeshTemplateElementCollection::add_pyramid_3d_C2,"Adds a second-order pyramid element by fourteen node indices")
		.def("nodal_dimension", &pyoomph::MeshTemplateElementCollection::nodal_dimension,"Returns the dimension of the Eulerian coordinates")
		.def("lagrangian_dimension", &pyoomph::MeshTemplateElementCollection::lagrangian_dimension,"Returns the dimension of the Lagrangian coordinates")
		.def("set_nodal_dimension", &pyoomph::MeshTemplateElementCollection::set_nodal_dimension,"Sets the dimension of the Eulerian coordinates")
		.def("set_lagrangian_dimension", &pyoomph::MeshTemplateElementCollection::set_lagrangian_dimension,"Sets the dimension of the Lagrangian coordinates")
		.def("get_element_dimension", &pyoomph::MeshTemplateElementCollection::get_element_dimension, "Returns the spatial dimension of the elements in this domain")
		.def("get_adjacent_boundary_names", &pyoomph::MeshTemplateElementCollection::get_adjacent_boundary_names, "Returns the names of all mesh boundaries adjacent to this domain")
		.def("set_all_nodes_as_boundary_nodes",&pyoomph::MeshTemplateElementCollection::set_all_nodes_as_boundary_nodes, "Marks every node of this domain as a boundary node, e.g. for domains that consist purely of interface elements")
		.def("set_element_code", &pyoomph::MeshTemplateElementCollection::set_element_code, py::arg("code_instance"), "Assigns the compiled equation code instance used to create the elements of this domain").
		doc()="A collection of bulk elements, i.e. a bulk domain of a mesh. Must be created as part of a :py:class:`~pyoomph.meshes.mesh.MeshTemplate` by :py:meth:`~pyoomph.meshes.mesh.MeshTemplate.new_domain`";

	py::class_<pyoomph::MeshTemplate, pyoomph::PyMeshTemplateTrampoline>(m, "MeshTemplate")
		.def(py::init<>())
		.def("_set_problem", &pyoomph::MeshTemplate::_set_problem, py::arg("problem"), "Associates this mesh template with the given Problem")
		.def("get_node_position", &pyoomph::MeshTemplate::get_node_position, py::arg("node_index"), "Returns the position [x, y, z] of the node at the given index of this template")
		.def("new_bulk_element_collection", &pyoomph::MeshTemplate::new_bulk_element_collection, py::return_value_policy::reference, py::arg("name"), "Creates a new named bulk element domain (MeshTemplateElementCollection) within this template")
		.def("add_facet_to_boundary", &pyoomph::MeshTemplate::add_facet_to_boundary,"boundary_name"_a, "all_node_indices"_a,py::arg("vertex_node_indices")=std::vector<pyoomph::nodeindex_t>(),py::arg("curved_entity")=nullptr,"Adds a list of nodes, i.e. a facet, to a boundary. Must pass all nodes of the facet, the vertex nodes and a potential curved entity for MacroElements")
		//.def("add_nodes_to_boundary", &pyoomph::MeshTemplate::add_nodes_to_boundary,"Adds a list of nodes, i.e. a facet, to a boundary")
		//.def("add_facet_to_curve_entity", &pyoomph::MeshTemplate::add_facet_to_curve_entity,"Adds a facet to a curved boundary so that e.g. additional nodes of refined meshes will be exactly on this curve")
		.def("add_nodes_to_boundary", [](pyoomph::MeshTemplate & self, const std::string &boundname, const std::vector<pyoomph::nodeindex_t> &ni){ throw_runtime_error("add_nodes_to_boundary is deprecated. Use add_facet_to_boundary instead"); }, py::arg("boundary_name"), py::arg("node_indices"))
		.def("add_facet_to_curve_entity",[](pyoomph::MeshTemplate & self, const std::vector<pyoomph::nodeindex_t> &ni, pyoomph::MeshTemplateCurvedEntity *curved=nullptr){ throw_runtime_error("add_facet_to_curve_entity is deprecated. Use add_facet_to_boundary instead"); }, py::arg("node_indices"), py::arg("curved_entity")=nullptr)
		.def("_find_opposite_interface_connections", &pyoomph::MeshTemplate::_find_opposite_interface_connections, "Finds, for every pair of boundaries registered as mutually opposite interfaces, the corresponding node/facet connections between them (used for e.g. two-sided interfaces)")
		.def("_find_interface_intersections", &pyoomph::MeshTemplate::_find_interface_intersections, "Finds the set of boundary names where two or more interfaces intersect (e.g. contact lines/triple points)")
		.def("add_periodic_node_pair", &pyoomph::MeshTemplate::add_periodic_node_pair, "n_mst"_a, "n_slv"_a, "Registers the node at index n_slv as periodic copy of the node at index n_mst")
		.def("add_node_unique", &pyoomph::MeshTemplate::add_node_unique, "x"_a, "y"_a = 0.0, "z"_a = 0.0,"Adds a node at the given position. If there is already a node at this position,no new node is created")
		.def("add_node", &pyoomph::MeshTemplate::add_node, "x"_a, "y"_a = 0.0, "z"_a = 0.0,"Adds a node at the given position. Creates overlapping nodes, if there is already a node at this position.")
		.def("_reset", &pyoomph::MeshTemplate::reset, "Clears all nodes, domains and boundary information of this mesh template, so it can be rebuilt from scratch")
		.doc()="A base class for the MeshTemplate class in pyoomph";

	// A mesh built from a 1d part of a MeshTemplate (i.e. a chain of line elements), with the
	// adaptive-refinement and boundary-setup machinery specific to one spatial dimension.
	py::class_<pyoomph::TemplatedMeshBase1d, pyoomph::Mesh, oomph::Mesh>(m, "TemplatedMeshBase1d")
		.def(py::init<>())
		.def("_set_problem", [](pyoomph::TemplatedMeshBase1d *self, pyoomph::Problem *p, pyoomph::DynamicBulkElementInstance *inst)
			 { self->_set_problem(p, inst); }, py::arg("problem"), py::arg("code_instance"))
		.def(
			"_get_problem", [](pyoomph::TemplatedMeshBase1d *self)
			{ return self->get_problem(); },
			py::return_value_policy::reference)
		.def("refinement_possible", [](pyoomph::TemplatedMeshBase1d *self)
			 { return true; }, "Returns whether adaptive refinement is possible for this mesh (always True for 1d meshes)")
		.def(
			"refine_uniformly", [](pyoomph::TemplatedMeshBase1d *self, unsigned int num)
			{for (unsigned int i=0;i<num;i++) self->refine_uniformly(); },
			"num"_a = 1, "Uniformly refines this mesh the given number of times")
		.def(
			"unrefine_uniformly", [](pyoomph::TemplatedMeshBase1d *self, unsigned int num)
			{for (unsigned int i=0;i<num;i++) if (self->unrefine_uniformly()) return true; return false; },
			"num"_a = 1, "Uniformly unrefines this mesh up to the given number of times; returns True if unrefinement was not fully possible (e.g. the coarsest level was reached)")
		.def("generate_from_template", &pyoomph::TemplatedMeshBase1d::generate_from_template, py::arg("template_collection"), "Builds this mesh's elements and nodes from the given 1d domain of a MeshTemplate")
		.def("refine_selected_elements", [](pyoomph::TemplatedMeshBase1d *m, std::vector<unsigned int> inds)
			 { oomph::Vector<unsigned> oomphvec(inds.size()); for (unsigned int i=0;i<inds.size();i++) oomphvec[i]=inds[i]; m->refine_selected_elements(oomphvec); }, py::arg("element_indices"), "Refines only the elements at the given indices in this mesh")
		.def("setup_boundary_element_info", [](pyoomph::TemplatedMeshBase1d *self)
			 { std::ostringstream oss; self->setup_boundary_element_info(oss); }, "(Re-)builds the lookup of which elements are adjacent to which mesh boundary")
		.def_property("identication_of_boundary_elements_by_facets", [](pyoomph::TemplatedMeshBase1d *self)
			 { return self->identication_of_boundary_elements_by_facets; },
			 [](pyoomph::TemplatedMeshBase1d *self, bool val) { self->identication_of_boundary_elements_by_facets=val; }, "Controls whether boundary elements are identified by matching facets (True) or by node membership (False)")
		.def("setup_tree_forest", &pyoomph::TemplatedMeshBase1d::setup_tree_forest, "Builds the binary tree forest used for adaptive mesh refinement of this 1d mesh");

	// A mesh built from a 2d part of a MeshTemplate (triangles/quads), with the corresponding
	// quadtree-based adaptive-refinement and boundary-setup machinery.
	py::class_<pyoomph::TemplatedMeshBase2d, pyoomph::Mesh, oomph::Mesh>(m, "TemplatedMeshBase2d")
		.def(py::init<>())
		.def("_set_problem", [](pyoomph::TemplatedMeshBase2d *self, pyoomph::Problem *p, pyoomph::DynamicBulkElementInstance *inst)
			 { self->_set_problem(p, inst); }, py::arg("problem"), py::arg("code_instance"))
		.def(
			"_get_problem", [](pyoomph::TemplatedMeshBase2d *self)
			{ return self->get_problem(); },
			py::return_value_policy::reference)
		.def("set_max_neighbour_finding_tolerance", [](pyoomph::TemplatedMeshBase2d *self, double tol)
			 {  oomph::Tree::max_neighbour_finding_tolerance() = tol; }, py::arg("tolerance"), "Sets the geometric tolerance used by oomph-lib's quadtree neighbour-finding algorithm")
		.def("generate_from_template", &pyoomph::TemplatedMeshBase2d::generate_from_template, py::arg("template_collection"), "Builds this mesh's elements and nodes from the given 2d domain of a MeshTemplate")
		.def("add_tri_C1",[](pyoomph::TemplatedMeshBase2d *self,pyoomph::Node *n1,pyoomph::Node *n2,pyoomph::Node *n3){self->add_tri_C1(n1,n2,n3);}, py::arg("n1"), py::arg("n2"), py::arg("n3"), "Adds a linear (C1) triangular element with the given three nodes directly to this mesh")
		.def("add_tri_C1TB",[](pyoomph::TemplatedMeshBase2d *self,pyoomph::Node *n1,pyoomph::Node *n2,pyoomph::Node *n3,pyoomph::Node *n4=NULL){self->add_tri_C1TB(n1,n2,n3,n4);}, py::arg("n1"), py::arg("n2"), py::arg("n3"), py::arg("n4")=nullptr, "Adds a linear triangular element with an additional (optional) bubble node with the given nodes directly to this mesh")
		.def("refinement_possible", &pyoomph::TemplatedMeshBase2d::refinement_possible, "Returns whether adaptive refinement is possible for this mesh (e.g. False if it contains non-refineable elements)")
		.def(
			"refine_uniformly", [](pyoomph::TemplatedMeshBase2d *self, unsigned int num)
			{for (unsigned int i=0;i<num;i++) self->refine_uniformly(); },
			"num"_a = 1, "Uniformly refines this mesh the given number of times")
		.def(
			"unrefine_uniformly", [](pyoomph::TemplatedMeshBase2d *self, unsigned int num)
			{for (unsigned int i=0;i<num;i++) if (self->unrefine_uniformly()) return true; return false; },
			"num"_a = 1, "Uniformly unrefines this mesh up to the given number of times; returns True if unrefinement was not fully possible (e.g. the coarsest level was reached)")
		.def("setup_tree_forest", &pyoomph::TemplatedMeshBase2d::setup_tree_forest, "Builds the quadtree forest used for adaptive mesh refinement of this 2d mesh")
		.def("refine_selected_elements", [](pyoomph::TemplatedMeshBase2d *m, std::vector<unsigned int> inds)
			 { oomph::Vector<unsigned> oomphvec(inds.size()); for (unsigned int i=0;i<inds.size();i++) oomphvec[i]=inds[i]; m->refine_selected_elements(oomphvec); }, py::arg("element_indices"), "Refines only the elements at the given indices in this mesh")
		 .def_property("identication_of_boundary_elements_by_facets", [](pyoomph::TemplatedMeshBase2d *self)
			 { return self->identication_of_boundary_elements_by_facets; },
			 [](pyoomph::TemplatedMeshBase2d *self, bool val) { self->identication_of_boundary_elements_by_facets=val; }, "Controls whether boundary elements are identified by matching facets (True) or by node membership (False)")
		.def("setup_boundary_element_info", [](pyoomph::TemplatedMeshBase2d *self)
			 { std::ostringstream oss; self->setup_boundary_element_info(oss); }, "(Re-)builds the lookup of which elements are adjacent to which mesh boundary");

	// A mesh built from a 3d part of a MeshTemplate (tets/bricks/...), with the corresponding
	// octree-based adaptive-refinement and boundary-setup machinery.
	py::class_<pyoomph::TemplatedMeshBase3d, pyoomph::Mesh, oomph::Mesh>(m, "TemplatedMeshBase3d")
		.def(py::init<>())
		.def("_set_problem", [](pyoomph::TemplatedMeshBase3d *self, pyoomph::Problem *p, pyoomph::DynamicBulkElementInstance *inst)
			 { self->_set_problem(p, inst); }, py::arg("problem"), py::arg("code_instance"))
		.def(
			"_get_problem", [](pyoomph::TemplatedMeshBase3d *self)
			{ return self->get_problem(); },
			py::return_value_policy::reference)
		.def("generate_from_template", &pyoomph::TemplatedMeshBase3d::generate_from_template, py::arg("template_collection"), "Builds this mesh's elements and nodes from the given 3d domain of a MeshTemplate")
		.def("refinement_possible", &pyoomph::TemplatedMeshBase3d::refinement_possible, "Returns whether adaptive refinement is possible for this mesh (e.g. False if it contains non-refineable elements)")
		.def(
			"refine_uniformly", [](pyoomph::TemplatedMeshBase3d *self, unsigned int num)
			{for (unsigned int i=0;i<num;i++) self->refine_uniformly(); },
			"num"_a = 1, "Uniformly refines this mesh the given number of times")
		.def(
			"unrefine_uniformly", [](pyoomph::TemplatedMeshBase3d *self, unsigned int num)
			{ for (unsigned int i=0;i<num;i++) if (self->unrefine_uniformly()) return true; return false; },
			"num"_a = 1, "Uniformly unrefines this mesh up to the given number of times; returns True if unrefinement was not fully possible (e.g. the coarsest level was reached)")
		.def("setup_tree_forest", &pyoomph::TemplatedMeshBase3d::setup_tree_forest, "Builds the octree forest used for adaptive mesh refinement of this 3d mesh")
		.def("refine_selected_elements", [](pyoomph::TemplatedMeshBase3d *m, std::vector<unsigned int> inds)
			 { oomph::Vector<unsigned> oomphvec(inds.size()); for (unsigned int i=0;i<inds.size();i++) oomphvec[i]=inds[i]; m->refine_selected_elements(oomphvec); }, py::arg("element_indices"), "Refines only the elements at the given indices in this mesh")
		.def_property("identication_of_boundary_elements_by_facets", [](pyoomph::TemplatedMeshBase3d *self)
			 { return self->identication_of_boundary_elements_by_facets; },
			 [](pyoomph::TemplatedMeshBase3d *self, bool val) { self->identication_of_boundary_elements_by_facets=val; }, "Controls whether boundary elements are identified by matching facets (True) or by node membership (False)")
		.def("setup_boundary_element_info", [](pyoomph::TemplatedMeshBase3d *self)
			 { std::ostringstream oss; self->setup_boundary_element_info(oss); }, "(Re-)builds the lookup of which elements are adjacent to which mesh boundary");

	// A mesh of interface elements attached to a boundary of a bulk mesh (e.g. for surface tension,
	// contact lines, or coupling two bulk domains); see pyoomph::InterfaceMesh in mesh.hpp.
	py::class_<pyoomph::InterfaceMesh, pyoomph::Mesh, oomph::Mesh>(m, "InterfaceMesh")
		.def("clear_before_adapt", &pyoomph::InterfaceMesh::clear_before_adapt, "Removes this interface mesh's elements before the adjacent bulk mesh is adapted; they are regenerated afterwards by rebuild_after_adapt")
		.def("nullify_selected_bulk_dofs", &pyoomph::InterfaceMesh::nullify_selected_bulk_dofs, "Nullifies the contribution of selected bulk degrees of freedom that are being superseded by this interface's own degrees of freedom")
		.def("_connect_interface_elements_by_kdtree", &pyoomph::InterfaceMesh::connect_interface_elements_by_kdtree, py::arg("other"), "Connects this interface mesh's elements to the spatially closest elements of another interface mesh using a kd-tree search, e.g. to set up two-sided interfaces")
		.def("rebuild_after_adapt", &pyoomph::InterfaceMesh::rebuild_after_adapt, "Regenerates this interface mesh's elements after the adjacent bulk mesh has been adapted")
		.def("set_opposite_interface_offset_vector",&pyoomph::InterfaceMesh::set_opposite_interface_offset_vector, py::arg("offset"), "Sets a constant coordinate offset applied when matching this interface to its opposite side (e.g. for periodic geometries)")
		.def("get_opposite_interface_offset_vector",&pyoomph::InterfaceMesh::get_opposite_interface_offset_vector, "Returns the coordinate offset applied when matching this interface to its opposite side")
		.def("update_zeta_in_buffer",&pyoomph::InterfaceMesh::update_zeta_in_buffer, "Updates the buffered intrinsic (zeta) coordinates of the interface nodes, used for locating the opposite side of a two-sided interface")
		.def("update_equation_remapping",&pyoomph::InterfaceMesh::update_equation_remapping, "Updates the mapping between this interface's local equation numbers and the global equation numbers after the dof numbering has changed")
		.def("get_bulk_mesh", &pyoomph::InterfaceMesh::get_bulk_mesh, "Returns the bulk mesh this interface mesh is attached to")
		.def(
			"_get_problem", [](pyoomph::InterfaceMesh *self)
			{ return self->get_problem(); },
			py::return_value_policy::reference)
		.def("_set_problem", [](pyoomph::InterfaceMesh *self, pyoomph::Problem *p, pyoomph::DynamicBulkElementInstance *inst)
			 { self->_set_problem(p, inst); }, py::arg("problem"), py::arg("code_instance"))
		//  .def(py::init<pyoomph::Problem*>());
		.def(py::init<>());

	m.def("set_tolerance_for_singular_jacobian", [](double tol)
		  { oomph::FiniteElement::Tolerance_for_singular_jacobian = tol; }, py::arg("tolerance"), "Sets the tolerance below which the local coordinate-to-Eulerian Jacobian of an element is considered singular (raising an error)");
	m.def("set_interpolate_new_interface_dofs", [](bool on)
		  { pyoomph::InterfaceElementBase::interpolate_new_interface_dofs = on; }, py::arg("on"), "Globally controls whether newly created interface degrees of freedom (e.g. after refinement) are interpolated from the neighbouring nodes or initialized to zero");
	m.def("set_use_eigen_Z2_error_estimators", [](bool on)
		  { pyoomph::BulkElementBase::use_eigen_error_estimators = on; }, py::arg("on"), "Globally controls whether the Eigen-based implementation is used for the Zienkiewicz-Zhu (Z2) spatial error estimator");

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
