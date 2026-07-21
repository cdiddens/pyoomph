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

// pyoomph::Mesh/TemplatedMeshBase use virtual (diamond) inheritance internally (see mesh.hpp:
// "class Mesh : public virtual oomph::RefineableMeshBase, public virtual oomph::Mesh", and
// TemplatedMeshBase/1d/2d/3d likewise), which nanobind cannot safely cast across: unlike
// pybind11's RTTI-based casting, nanobind adjusts base-class pointers via a cached, supposedly
// fixed compile-time offset, which is unsound for virtual bases (their offset depends on the
// most-derived type and is only known at runtime via the vtable). Asking nanobind to expose an
// inherited method across such a boundary silently computes a garbage pointer and crashes.
//
// To avoid this entirely, every mesh kind exposed to Python (TemplatedMeshBase1d/2d/3d,
// InterfaceMesh, ODEStorageMesh; see mesh.cpp for the concrete MeshHandle<T> wrappers and their
// bindings) is backed not by the polymorphic pyoomph object directly, but by a small wrapper that
// owns it and exposes it via a real (correct, RTTI-based) dynamic_cast, done in plain C++ code
// rather than by nanobind. This header declares just the abstract interface, so that other
// translation units (e.g. problem.cpp) that need to accept "any mesh" argument from Python can do
// so without depending on mesh.cpp's concrete MeshHandle<T>/handle_method machinery.
#pragma once

#include "../mesh.hpp"
#include <unordered_map>
#include <nanobind/nanobind.h>

class MeshHandleBase
{
public:
	virtual ~MeshHandleBase() = default;
	virtual pyoomph::Mesh *mesh() const = 0;
	virtual oomph::Mesh *oomph_mesh() const = 0;
	// Forces immediate (synchronous) destruction of the owned oomph-lib mesh - and, via its
	// normal destructor, all of its elements and nodes - right now, rather than whenever this
	// handle's Python wrapper object eventually gets garbage collected. Used by
	// Problem.release() (see problem.py) to guarantee every element referencing a
	// DynamicBulkElementCode's compiled residual/Jacobian function table has actually been
	// destructed *before* that code's shared library is dlclose()'d - otherwise, if the mesh
	// happened to outlive release() (e.g. because a user script still holds a reference to it,
	// which is common and not itself a bug), its elements' destructors would run later against
	// an already-unloaded function table and crash. After this call, mesh()/oomph_mesh() return
	// nullptr; the handle must not be used for anything else afterwards.
	virtual void _destroy_now() = 0;
};

// A handful of oomph-lib/pyoomph APIs (e.g. Problem::mesh_pt()) hand back a raw, previously
// constructed mesh pointer rather than accepting one, and Python code expects to get back the
// very same Handle object it originally passed in (e.g. to add_sub_mesh()), not a fresh one --
// important since that object may already carry Python-level state (__dict__ attributes) other
// code depends on. nanobind's own instance registry only knows about MeshHandleBase/MeshHandle<T>
// objects it manages directly, not the raw pyoomph::Mesh pointer living inside them, so every
// MeshHandle<T> registers/unregisters itself here (see mesh.cpp) keyed by that raw pointer.
inline std::unordered_map<pyoomph::Mesh *, MeshHandleBase *> &pyoomph_mesh_handle_registry()
{
	static std::unordered_map<pyoomph::Mesh *, MeshHandleBase *> registry;
	return registry;
}

// Looks up the Python object of the MeshHandle<T> that owns "raw" (found via a real dynamic_cast,
// then the registry above), or returns None if raw is null or was never wrapped by a MeshHandle
// (e.g. a mesh constructed purely on the C++ side without ever crossing into Python).
inline nanobind::object pyoomph_find_mesh_handle(oomph::Mesh *raw)
{
	if (!raw)
		return nanobind::none();
	pyoomph::Mesh *key = dynamic_cast<pyoomph::Mesh *>(raw);
	if (!key)
		return nanobind::none();
	auto &registry = pyoomph_mesh_handle_registry();
	auto it = registry.find(key);
	if (it == registry.end())
		return nanobind::none();
	return nanobind::find(*it->second);
}
