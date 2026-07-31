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

// Python bindings for TQMesh, the two-dimensional triangle/quad mesh generator that
// cmake/ThirdPartyTQMesh.cmake downloads and prepares (see citools/patches/patch_tqmesh.cmake for
// the two changes made to its sources). Only compiled into _pyoomph_core when PYOOMPH_HAS_TQMESH is
// on (the default, see CMakeLists.txt), which is also the only place in pyoomph that may include
// TQMesh's headers - the rest of the C++ core must keep building without them.
//
// The bindings follow TQMesh's own three-step workflow - describe a Domain, hand it to a
// MeshGenerator, run meshing algorithms on the resulting Mesh - but hand out handles instead of
// the C++ references TQMesh itself passes around. TQMesh's MeshGenerator owns its meshes in a
// vector of unique_ptrs and destroys one whenever meshes get merged, so a Python object holding a
// raw Mesh& would be left dangling by an ordinary merge; every access through TQMeshMesh instead
// goes through the shared generator state, which knows which meshes are still alive and raises a
// RuntimeError rather than reading freed memory.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/optional.h>

#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "TQMesh.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace pyoomph
{
	namespace tqmesh_bindings
	{

		using Vec2d = CppUtils::Vec2d;
		using Coordinates = std::vector<std::pair<double, double>>;

		// Wraps a freshly filled, row-major buffer of "rows x cols" values as a new 2D numpy array
		// owning its data, in the same manner as vector_to_ndarray() in ../nb_array_utils.hpp (not
		// included here: that header pulls in the rest of the binding layer's conventions, while this
		// translation unit deliberately sees nothing but nanobind and TQMesh).
		template <typename T>
		static nb::ndarray<nb::numpy, T> to_ndarray_2d(const std::vector<T> &flat, size_t cols)
		{
			const size_t rows = cols ? flat.size() / cols : 0;
			T *data = new T[flat.size()];
			std::copy(flat.begin(), flat.end(), data);
			nb::capsule owner(data, [](void *p) noexcept
							  { delete[] (T *)p; });
			return nb::ndarray<nb::numpy, T>(data, {rows, cols}, owner);
		}

		// Turns whatever Python passed as a mesh size specification into TQMesh's UserSizeFunction:
		// either a constant (a plain number) or a callable f(x,y) that is invoked for every position
		// the meshing algorithms are interested in. The callable is captured by value, i.e. the
		// std::function stored inside the TQMesh::Domain keeps it alive for as long as the domain
		// exists. We never release the GIL while meshing, so calling back into Python is safe; the
		// explicit acquire only documents that and keeps this correct should that ever change.
		static TQMesh::UserSizeFunction make_size_function(nb::object size_function)
		{
			if (size_function.is_none())
				return [](const Vec2d &) { return 1.0; };

			if (nb::isinstance<nb::float_>(size_function) || nb::isinstance<nb::int_>(size_function))
			{
				const double size = nb::cast<double>(size_function);
				if (!(size > 0.0))
					throw std::runtime_error("A constant mesh size must be positive, got " + std::to_string(size));
				return [size](const Vec2d &) { return size; };
			}

			if (!nb::hasattr(size_function, "__call__"))
				throw std::runtime_error("The mesh size must either be a positive number or a callable f(x,y) returning the desired element size at the position (x,y)");

			return [size_function](const Vec2d &p) -> double
			{
				nb::gil_scoped_acquire gil;
				return nb::cast<double>(size_function(p.x, p.y));
			};
		}

		// Everything a generated mesh keeps referring to: TQMesh's Mesh objects hold a reference to
		// the Domain they were built from (their boundary edges point into its vertices), and the
		// domain in turn holds the Python size callback. Handing this around as a shared_ptr means a
		// script can drop its TQMeshDomain object right after generating and still work with the
		// mesh.
		struct DomainData
		{
			std::unique_ptr<TQMesh::Domain> domain;
			// Fixed vertices are addressed by index from Python (add_fixed_edge), since TQMesh's own
			// Vertex objects are not exposed.
			std::vector<TQMesh::Vertex *> fixed_vertices;
			// The quadtree scale in effect when this domain was created, i.e. the one its own vertex
			// lookup structure was built with (see check_within_quadtree).
			double quadtree_scale;
		};

		// TQMesh keeps its entities in quadtrees spanning [-scale/2, scale/2] in both directions,
		// always centered at the origin. Anything outside is silently not inserted (QuadTree::add()
		// just returns false), after which the meshing algorithms look up entities that are not
		// there and dereference null - i.e. this has to be caught here, before it reaches TQMesh.
		static void check_within_quadtree(double x, double y, double scale, const std::string &what)
		{
			const double half = 0.5 * scale;
			if (x < -half || x > half || y < -half || y > half)
				throw std::runtime_error(what + " at (" + std::to_string(x) + ", " + std::to_string(y) +
										 ") lies outside of TQMesh's quadtree, which spans [" + std::to_string(-half) +
										 ", " + std::to_string(half) + "] in each direction. Create the domain with a "
										 "quadtree_scale of at least twice the largest coordinate of the mesh.");
		}

		// The generator state shared between a TQMeshGenerator and every TQMeshMesh handle it handed
		// out. "alive" is what makes the handles safe: TQMeshGenerator::merge() lets TQMesh destroy
		// the donor mesh, and any handle still pointing at it must fail loudly afterwards.
		struct GeneratorData
		{
			TQMesh::MeshGenerator generator;
			std::set<TQMesh::Mesh *> alive;
			std::vector<std::shared_ptr<DomainData>> domains;
		};

		//===============================================================================
		// Domain
		//===============================================================================

		class Domain
		{
		public:
			Domain(nb::object size_function, double quadtree_scale)
			{
				// TQMesh reads the quadtree scale in the Domain constructor via its global setup
				// singleton, so it has to be set first and cannot be changed for this domain later on.
				if (quadtree_scale > 0.0)
					TQMesh::TQMeshSetup::get_instance().set_quadtree_scale(quadtree_scale);
				data = std::make_shared<DomainData>();
				data->quadtree_scale = TQMesh::TQMeshSetup::get_instance().get_quadtree_scale();
				data->domain = std::make_unique<TQMesh::Domain>(make_size_function(size_function));
			}

			TQMesh::Domain &domain() const { return *(data->domain); }

			size_t n_boundaries() const { return domain().size(); }
			double area() const { return domain().area(); }
			double size_function(double x, double y) const { return domain().size_function(Vec2d{x, y}); }

			// (x_min, x_max, y_min, y_max) over all boundary vertices
			std::tuple<double, double, double, double> extent() const
			{
				auto ext = domain().extent();
				return {ext.first[0], ext.first[1], ext.second[0], ext.second[1]};
			}

			bool is_inside(double x, double y) const { return domain().is_inside(Vec2d{x, y}); }

			size_t add_exterior_boundary(const Coordinates &coords, const std::vector<int> &colors,
										 const std::optional<Coordinates> &vertex_properties)
			{
				return add_boundary(domain().add_exterior_boundary(), coords, colors, vertex_properties);
			}

			size_t add_interior_boundary(const Coordinates &coords, const std::vector<int> &colors,
										 const std::optional<Coordinates> &vertex_properties)
			{
				return add_boundary(domain().add_interior_boundary(), coords, colors, vertex_properties);
			}

			size_t add_exterior_rectangle(int color, double x, double y, double width, double height,
										  double mesh_size, double mesh_range)
			{
				check_rectangle(x, y, width, height);
				domain().add_exterior_boundary().set_shape_rectangle(color, Vec2d{x, y}, width, height, mesh_size, mesh_range);
				return domain().size() - 1;
			}

			size_t add_interior_rectangle(int color, double x, double y, double width, double height,
										  double mesh_size, double mesh_range)
			{
				check_rectangle(x, y, width, height);
				domain().add_interior_boundary().set_shape_rectangle(color, Vec2d{x, y}, width, height, mesh_size, mesh_range);
				return domain().size() - 1;
			}

			size_t add_exterior_circle(int color, double x, double y, double radius, size_t n_segments,
									   double mesh_size, double mesh_range)
			{
				check_rectangle(x, y, 2.0 * radius, 2.0 * radius);
				domain().add_exterior_boundary().set_shape_circle(color, Vec2d{x, y}, radius, n_segments, mesh_size, mesh_range);
				return domain().size() - 1;
			}

			size_t add_interior_circle(int color, double x, double y, double radius, size_t n_segments,
									   double mesh_size, double mesh_range)
			{
				check_rectangle(x, y, 2.0 * radius, 2.0 * radius);
				domain().add_interior_boundary().set_shape_circle(color, Vec2d{x, y}, radius, n_segments, mesh_size, mesh_range);
				return domain().size() - 1;
			}

			// A vertex that is not part of any boundary, but which the mesh must contain - either to
			// refine the mesh locally around it, or as an end point of a fixed edge. Returns its index
			// for use in add_fixed_edge().
			size_t add_fixed_vertex(double x, double y, double mesh_size, double mesh_range)
			{
				check_within_quadtree(x, y, data->quadtree_scale, "The fixed vertex");
				TQMesh::Vertex &v = domain().add_fixed_vertex(Vec2d{x, y}, mesh_size, mesh_range);
				data->fixed_vertices.push_back(&v);
				return data->fixed_vertices.size() - 1;
			}

			// An edge that the mesh generator must not cut through, given by two fixed vertices
			// (as returned by add_fixed_vertex).
			void add_fixed_edge(size_t i1, size_t i2)
			{
				if (i1 >= data->fixed_vertices.size() || i2 >= data->fixed_vertices.size())
					throw std::runtime_error("Fixed vertex index out of range - only the indices returned by add_fixed_vertex() may be used here");
				domain().add_fixed_edge(*(data->fixed_vertices[i1]), *(data->fixed_vertices[i2]));
			}

			std::shared_ptr<DomainData> data;

		private:
			// The corners of a shape given by its center and its extents
			void check_rectangle(double x, double y, double width, double height) const
			{
				check_within_quadtree(x - 0.5 * width, y - 0.5 * height, data->quadtree_scale, "The boundary");
				check_within_quadtree(x + 0.5 * width, y + 0.5 * height, data->quadtree_scale, "The boundary");
			}

			size_t add_boundary(TQMesh::Boundary &bnd, const Coordinates &coords, const std::vector<int> &colors,
								const std::optional<Coordinates> &vertex_properties)
			{
				if (coords.size() < 3)
					throw std::runtime_error("A boundary requires at least three vertices, got " + std::to_string(coords.size()));
				if (colors.size() != coords.size())
					throw std::runtime_error("Expected one edge color per boundary vertex (" + std::to_string(coords.size()) + "), got " + std::to_string(colors.size()));
				if (vertex_properties && vertex_properties->size() != coords.size())
					throw std::runtime_error("Expected one (mesh size, mesh range) pair per boundary vertex (" + std::to_string(coords.size()) + "), got " + std::to_string(vertex_properties->size()));

				std::vector<Vec2d> v_coords;
				v_coords.reserve(coords.size());
				for (const auto &c : coords)
				{
					check_within_quadtree(c.first, c.second, data->quadtree_scale, "The boundary vertex");
					v_coords.push_back(Vec2d{c.first, c.second});
				}

				if (vertex_properties)
				{
					std::vector<Vec2d> props;
					props.reserve(vertex_properties->size());
					for (const auto &p : *vertex_properties)
						props.push_back(Vec2d{p.first, p.second});
					bnd.set_shape_from_coordinates(v_coords, colors, props);
				}
				else
				{
					bnd.set_shape_from_coordinates(v_coords, colors);
				}
				return domain().size() - 1;
			}
		};

		//===============================================================================
		// Mesh handle
		//===============================================================================

		class Mesh
		{
		public:
			Mesh(std::shared_ptr<GeneratorData> gen, std::shared_ptr<DomainData> dom, TQMesh::Mesh *mesh)
				: gen(gen), dom(dom), mesh(mesh) {}

			TQMesh::Mesh &get() const
			{
				if (!gen || !gen->alive.count(mesh))
					throw std::runtime_error("This mesh no longer exists - it was merged into another one and destroyed in the process");
				return *mesh;
			}

			size_t n_vertices() const { return get().n_vertices(); }
			size_t n_triangles() const { return get().n_triangles(); }
			size_t n_quads() const { return get().n_quads(); }
			size_t n_elements() const { return get().n_elements(); }
			size_t n_boundary_edges() const { return get().n_boundary_edges(); }
			size_t n_interior_edges() const { return get().n_interior_edges(); }
			// Summed up over the elements rather than taken from TQMesh's own Mesh::area(), which is a
			// counter incremented while elements are created and hence stale after a merge: the
			// receiver of a merge keeps reporting the area it had before.
			double area() const
			{
				TQMesh::Mesh &m = get();
				double area = 0.0;
				for (const auto &t : m.triangles())
					area += t->area();
				for (const auto &q : m.quads())
					area += q->area();
				return area;
			}
			int id() const { return get().id(); }
			int element_color() const { return get().element_color(); }

			// The (n_vertices, 2) coordinates of all mesh vertices. Every other array returned here
			// indexes into this one, i.e. into the row order it defines.
			nb::ndarray<nb::numpy, double> vertices() const
			{
				TQMesh::Mesh &m = get();
				std::vector<double> coords;
				coords.reserve(2 * m.n_vertices());
				for (const auto &v : m.vertices())
				{
					coords.push_back(v->xy().x);
					coords.push_back(v->xy().y);
				}
				return to_ndarray_2d(coords, 2);
			}

			nb::ndarray<nb::numpy, int64_t> triangles() const
			{
				TQMesh::Mesh &m = get();
				const auto index = vertex_indices(m);
				std::vector<int64_t> conn;
				conn.reserve(3 * m.n_triangles());
				for (const auto &t : m.triangles())
				{
					conn.push_back(index.at(&t->v1()));
					conn.push_back(index.at(&t->v2()));
					conn.push_back(index.at(&t->v3()));
				}
				return to_ndarray_2d(conn, 3);
			}

			nb::ndarray<nb::numpy, int64_t> quads() const
			{
				TQMesh::Mesh &m = get();
				const auto index = vertex_indices(m);
				std::vector<int64_t> conn;
				conn.reserve(4 * m.n_quads());
				for (const auto &q : m.quads())
				{
					conn.push_back(index.at(&q->v1()));
					conn.push_back(index.at(&q->v2()));
					conn.push_back(index.at(&q->v3()));
					conn.push_back(index.at(&q->v4()));
				}
				return to_ndarray_2d(conn, 4);
			}

			std::vector<int> triangle_colors() const
			{
				std::vector<int> colors;
				for (const auto &t : get().triangles())
					colors.push_back(t->color());
				return colors;
			}

			std::vector<int> quad_colors() const
			{
				std::vector<int> colors;
				for (const auto &q : get().quads())
					colors.push_back(q->color());
				return colors;
			}

			// The (n_boundary_edges, 3) array of [vertex 1, vertex 2, color] for each boundary edge.
			// The color is the one given to the boundary this edge stems from, i.e. it is what
			// identifies the boundary a mesh edge belongs to.
			nb::ndarray<nb::numpy, int64_t> boundary_edges() const
			{
				TQMesh::Mesh &m = get();
				const auto index = vertex_indices(m);
				std::vector<int64_t> edges;
				for (const auto &e : m.get_valid_boundary_edges())
				{
					edges.push_back(index.at(&e->v1()));
					edges.push_back(index.at(&e->v2()));
					edges.push_back(e->color());
				}
				return to_ndarray_2d(edges, 3);
			}

			// Whether the mesh covers its domain completely, i.e. whether meshing actually succeeded.
			// This must be checked after generating: TQMesh's algorithms can leave holes behind
			// without failing outright.
			bool check_completeness(bool mesh_cleanup) const
			{
				TQMesh::MeshChecker checker{get(), *(dom->domain)};
				return checker.check_completeness(mesh_cleanup);
			}

			std::shared_ptr<GeneratorData> gen;
			std::shared_ptr<DomainData> dom;
			TQMesh::Mesh *mesh;

		private:
			// TQMesh assigns each vertex an index() as well, but it is only guaranteed to agree with
			// the position in the vertex container while nothing has been removed from it - which
			// mesh cleanup and merging do. Mapping the pointers of the very container we export as
			// vertices() is correct regardless.
			static std::unordered_map<const TQMesh::Vertex *, int64_t> vertex_indices(TQMesh::Mesh &m)
			{
				std::unordered_map<const TQMesh::Vertex *, int64_t> index;
				index.reserve(m.n_vertices());
				int64_t i = 0;
				for (const auto &v : m.vertices())
					index[v.get()] = i++;
				return index;
			}
		};

		//===============================================================================
		// Generator
		//===============================================================================

		class Generator
		{
		public:
			Generator() : data(std::make_shared<GeneratorData>()) {}

			size_t size() const { return data->alive.size(); }

			Mesh new_mesh(Domain &domain, int mesh_id, int element_color)
			{
				// The mesh gets its own entity quadtrees, built with whatever the global scale is right
				// now - which is not necessarily the one the domain was created with, if another domain
				// has changed it meanwhile.
				const double scale = TQMesh::TQMeshSetup::get_instance().get_quadtree_scale();
				auto ext = domain.domain().extent();
				check_within_quadtree(ext.first[0], ext.second[0], scale, "The domain's lower left corner");
				check_within_quadtree(ext.first[1], ext.second[1], scale, "The domain's upper right corner");
				TQMesh::Mesh &m = data->generator.new_mesh(domain.domain(), mesh_id, element_color);
				data->alive.insert(&m);
				data->domains.push_back(domain.data);
				return Mesh{data, domain.data, &m};
			}

			// Fills the domain with triangles by the advancing front algorithm. All arguments left
			// unset keep TQMesh's own defaults.
			bool triangulate(Mesh &mesh, std::optional<size_t> n_elements, std::optional<double> mesh_range_factor,
							 std::optional<double> wide_search_factor, std::optional<double> min_cell_quality,
							 std::optional<double> max_cell_angle, std::optional<double> base_vertex_factor,
							 bool show_progress)
			{
				auto &tri = data->generator.triangulation(check(mesh));
				tri.show_progress(show_progress);
				if (n_elements)
					tri.n_elements(*n_elements);
				if (mesh_range_factor)
					tri.mesh_range_factor(*mesh_range_factor);
				if (wide_search_factor)
					tri.wide_search_factor(*wide_search_factor);
				if (min_cell_quality)
					tri.min_cell_quality(*min_cell_quality);
				if (max_cell_angle)
					tri.max_cell_angle(*max_cell_angle);
				if (base_vertex_factor)
					tri.base_vertex_factor(*base_vertex_factor);
				return tri.generate_elements();
			}

			// Puts a layer of quads along the boundary between the given starting and ending
			// position. Both must coincide with boundary vertices; passing the same point twice
			// covers the entire closed boundary it lies on. Must be run before triangulate().
			bool quad_layer(Mesh &mesh, size_t n_layers, double first_height, double growth_rate,
							std::pair<double, double> start, std::pair<double, double> end,
							std::optional<double> angle_factor, bool show_progress)
			{
				check_within_quadtree(start.first, start.second, mesh.dom->quadtree_scale, "The quad layer's starting position");
				check_within_quadtree(end.first, end.second, mesh.dom->quadtree_scale, "The quad layer's ending position");
				auto &layer = data->generator.quad_layer_generation(check(mesh));
				layer.show_progress(show_progress);
				layer.n_layers(n_layers);
				layer.first_height(first_height);
				layer.growth_rate(growth_rate);
				layer.starting_position(start.first, start.second);
				layer.ending_position(end.first, end.second);
				if (angle_factor)
					layer.angle_factor(*angle_factor);
				return layer.generate_elements();
			}

			// Merges pairs of triangles into quads wherever that improves the mesh quality.
			bool tri2quad(Mesh &mesh)
			{
				return data->generator.tri2quad_modification(check(mesh)).modify();
			}

			// Splits every element into quads, turning a triangular or mixed mesh into an all-quad
			// one (each triangle becomes three quads, each quad four).
			bool quad_refine(Mesh &mesh)
			{
				return data->generator.quad_refinement(check(mesh)).refine();
			}

			// Improves the element quality by moving the interior vertices. "kind" selects the
			// strategy: "mixed" (the default, torsion followed by laplacian), "laplace" or "torsion".
			bool smooth(Mesh &mesh, int iterations, const std::string &kind, std::optional<double> epsilon,
						std::optional<double> decay, std::optional<double> angle_factor, bool quad_layer_smoothing)
			{
				TQMesh::Mesh &m = check(mesh);
				if (kind == "mixed")
				{
					auto &s = data->generator.mixed_smoothing(m);
					if (epsilon)
						s.epsilon(*epsilon);
					if (decay)
						s.decay(*decay);
					if (angle_factor)
						s.angle_factor(*angle_factor);
					s.quad_layer_smoothing(quad_layer_smoothing);
					return s.smooth(iterations);
				}
				if (kind == "laplace")
				{
					auto &s = data->generator.laplace_smoothing(m);
					if (epsilon)
						s.epsilon(*epsilon);
					if (decay)
						s.decay(*decay);
					if (angle_factor)
						throw std::runtime_error("The argument angle_factor is only supported by the smoothing kinds 'mixed' and 'torsion'");
					s.quad_layer_smoothing(quad_layer_smoothing);
					return s.smooth(iterations);
				}
				if (kind == "torsion")
				{
					auto &s = data->generator.torsion_smoothing(m);
					if (epsilon)
						s.epsilon(*epsilon);
					if (decay)
						s.decay(*decay);
					if (angle_factor)
						s.angle_factor(*angle_factor);
					s.quad_layer_smoothing(quad_layer_smoothing);
					return s.smooth(iterations);
				}
				throw std::runtime_error("Unknown smoothing kind '" + kind + "' - use 'mixed', 'laplace' or 'torsion'");
			}

			// Merges the donor mesh into the receiver, which requires both to share a common
			// interface. On success the donor is destroyed, i.e. its handle cannot be used anymore.
			bool merge(Mesh &receiver, Mesh &donor)
			{
				TQMesh::Mesh &r = check(receiver);
				TQMesh::Mesh *d = &check(donor);
				if (&r == d)
					throw std::runtime_error("A mesh cannot be merged into itself");
				data->generator.merge_meshes(r, *d);
				// Neither the return value of merge_meshes() nor the donor itself can be consulted
				// afterwards: it returns true even when the merge failed and left both meshes as they
				// were, and destroys the donor when it worked. What the generator still owns is the
				// one thing that says which of the two happened.
				refresh_alive();
				return data->alive.count(d) == 0;
			}

			// Writes the mesh to <filename>.vtu or <filename>.txt (the extension is added by TQMesh).
			bool write(Mesh &mesh, const std::string &filename, const std::string &format)
			{
				TQMesh::MeshExportType type;
				if (format == "vtu")
					type = TQMesh::MeshExportType::VTU;
				else if (format == "txt")
					type = TQMesh::MeshExportType::TXT;
				else if (format == "cout")
					type = TQMesh::MeshExportType::COUT;
				else
					throw std::runtime_error("Unknown mesh export format '" + format + "' - use 'vtu', 'txt' or 'cout'");
				return data->generator.write_mesh(check(mesh), filename, type);
			}

			std::shared_ptr<GeneratorData> data;

		private:
			// Re-reads which meshes the generator currently owns. Called after a merge, which is the
			// only operation that destroys one.
			void refresh_alive()
			{
				data->alive.clear();
				for (size_t i = 0; i < data->generator.size(); i++)
					data->alive.insert(&(data->generator.mesh(i)));
			}

			// A handle may well come from a different generator, in which case none of the algorithms
			// below would find the mesh's domain and TQMesh would terminate the process.
			TQMesh::Mesh &check(Mesh &mesh) const
			{
				if (mesh.gen != data)
					throw std::runtime_error("This mesh belongs to a different TQMeshGenerator");
				return mesh.get();
			}
		};

	} // namespace tqmesh_bindings
} // namespace pyoomph

void PyReg_TQMesh(nb::module_ &m)
{
	using namespace pyoomph::tqmesh_bindings;

	m.def(
		"tqmesh_version", []()
		{ return std::to_string(TQMESH_VERSION_MAJOR) + "." + std::to_string(TQMESH_VERSION_MINOR); },
		"Version of the TQMesh library vendored in this build of pyoomph.");

	m.def(
		"tqmesh_set_quadtree_parameters", [](std::optional<double> scale, std::optional<size_t> max_items, std::optional<size_t> max_depth)
		{
			auto &setup = TQMesh::TQMeshSetup::get_instance();
			if (scale) setup.set_quadtree_scale(*scale);
			if (max_items) setup.set_quadtree_max_items(*max_items);
			if (max_depth) setup.set_quadtree_max_depth(*max_depth); },
		"scale"_a = nb::none(), "max_items"_a = nb::none(), "max_depth"_a = nb::none(),
		"Sets the parameters of the quadtree that TQMesh uses to look up mesh entities by position. These are global settings of the TQMesh library, and each of them is read when a TQMeshDomain is constructed, i.e. changing them afterwards has no effect on domains that already exist. The scale should be somewhat larger than the extent of the mesh to be generated and can also be passed directly to TQMeshDomain.");

	nb::class_<Domain>(m, "TQMeshDomain", "The domain to be meshed by TQMesh: the mesh size distribution together with the boundaries enclosing it. Boundaries are closed polygonal chains of edges, where each edge carries an integer 'color' that identifies which boundary the resulting mesh edges belong to. Exactly one exterior boundary (counter-clockwise) must be present, plus any number of interior boundaries (clockwise) describing holes.")
		.def(nb::init<nb::object, double>(), "mesh_size"_a = 1.0, "quadtree_scale"_a = 0.0,
			 "The mesh_size is either a positive number for a uniform mesh or a callable f(x,y) returning the desired element size at the position (x,y). A quadtree_scale > 0 sets TQMesh's global quadtree scale (see tqmesh_set_quadtree_parameters) before this domain is created; it should be somewhat larger than the extent of the mesh.")
		.def("add_exterior_boundary", &Domain::add_exterior_boundary, "coordinates"_a, "colors"_a, "vertex_properties"_a = nb::none(),
			 "Adds the exterior boundary from a closed polygonal chain, given as a sequence of (x,y) coordinates in counter-clockwise order, along with one edge color per vertex (the color of the edge starting at that vertex). Optionally, one (mesh size, mesh range) pair per vertex refines the mesh locally around it, where a mesh size of 0 means no local refinement. Returns the index of the new boundary.")
		.def("add_interior_boundary", &Domain::add_interior_boundary, "coordinates"_a, "colors"_a, "vertex_properties"_a = nb::none(),
			 "Adds an interior boundary, i.e. a hole, from a closed polygonal chain given in clockwise order. Arguments as in add_exterior_boundary. Returns the index of the new boundary.")
		.def("add_exterior_rectangle", &Domain::add_exterior_rectangle, "color"_a, "x"_a, "y"_a, "width"_a, "height"_a, "mesh_size"_a = 0.0, "mesh_range"_a = 0.0,
			 "Adds a rectangular exterior boundary around the center (x,y), with all edges having the given color. Returns the index of the new boundary.")
		.def("add_interior_rectangle", &Domain::add_interior_rectangle, "color"_a, "x"_a, "y"_a, "width"_a, "height"_a, "mesh_size"_a = 0.0, "mesh_range"_a = 0.0,
			 "Adds a rectangular hole around the center (x,y), with all edges having the given color. Returns the index of the new boundary.")
		.def("add_exterior_circle", &Domain::add_exterior_circle, "color"_a, "x"_a, "y"_a, "radius"_a, "n_segments"_a = 30, "mesh_size"_a = 0.0, "mesh_range"_a = 0.0,
			 "Adds a circular exterior boundary around the center (x,y), discretized by n_segments edges of the given color. Returns the index of the new boundary.")
		.def("add_interior_circle", &Domain::add_interior_circle, "color"_a, "x"_a, "y"_a, "radius"_a, "n_segments"_a = 30, "mesh_size"_a = 0.0, "mesh_range"_a = 0.0,
			 "Adds a circular hole around the center (x,y), discretized by n_segments edges of the given color. Returns the index of the new boundary.")
		.def("add_fixed_vertex", &Domain::add_fixed_vertex, "x"_a, "y"_a, "mesh_size"_a = 0.0, "mesh_range"_a = 0.0,
			 "Adds a vertex inside the domain which the generated mesh must contain, optionally refining the mesh around it to the given mesh size within the given range. Returns its index, to be used in add_fixed_edge.")
		.def("add_fixed_edge", &Domain::add_fixed_edge, "vertex1"_a, "vertex2"_a,
			 "Adds an edge between two fixed vertices (given by the indices returned from add_fixed_vertex) which the generated mesh must respect, i.e. no element will cross it. Unlike an interior boundary, it does not enclose a hole.")
		.def("size_function", &Domain::size_function, "x"_a, "y"_a,
			 "Evaluates the mesh size at the position (x,y), including the local refinements set at boundary and fixed vertices.")
		.def("is_inside", &Domain::is_inside, "x"_a, "y"_a, "Whether the position (x,y) lies inside the domain, i.e. within the exterior and outside all interior boundaries.")
		.def("extent", &Domain::extent, "The bounding box (x_min, x_max, y_min, y_max) of all boundary vertices.")
		.def("area", &Domain::area, "The area enclosed by the boundaries.")
		.def_prop_ro("n_boundaries", &Domain::n_boundaries, "Number of boundaries added so far.");

	nb::class_<Mesh>(m, "TQMeshMesh", "A mesh generated by a TQMeshGenerator. All connectivity arrays index into the rows of vertices().")
		.def("vertices", &Mesh::vertices, "The (n_vertices, 2) array of vertex coordinates.")
		.def("triangles", &Mesh::triangles, "The (n_triangles, 3) array of vertex indices of all triangular elements.")
		.def("quads", &Mesh::quads, "The (n_quads, 4) array of vertex indices of all quadrilateral elements.")
		.def("triangle_colors", &Mesh::triangle_colors, "The element color of each triangle, in the order of triangles().")
		.def("quad_colors", &Mesh::quad_colors, "The element color of each quadrilateral, in the order of quads().")
		.def("boundary_edges", &Mesh::boundary_edges, "The (n_boundary_edges, 3) array of [vertex 1, vertex 2, color] of all boundary edges. The color is the one assigned to the domain boundary the edge stems from.")
		.def("check_completeness", &Mesh::check_completeness, "mesh_cleanup"_a = true,
			 "Whether the mesh covers the entire domain. This should always be checked after generating elements, since the algorithms can fail to fill the domain without reporting an error.")
		.def("area", &Mesh::area, "The total area covered by the mesh elements.")
		.def_prop_ro("n_vertices", &Mesh::n_vertices)
		.def_prop_ro("n_triangles", &Mesh::n_triangles)
		.def_prop_ro("n_quads", &Mesh::n_quads)
		.def_prop_ro("n_elements", &Mesh::n_elements)
		.def_prop_ro("n_boundary_edges", &Mesh::n_boundary_edges)
		.def_prop_ro("n_interior_edges", &Mesh::n_interior_edges)
		.def_prop_ro("id", &Mesh::id, "The mesh id given when the mesh was created.")
		.def_prop_ro("element_color", &Mesh::element_color, "The element color given when the mesh was created.");

	nb::class_<Generator>(m, "TQMeshGenerator", "Generates meshes for TQMeshDomains and owns them. The meshing algorithms are applied in the order quad_layer -> triangulate -> tri2quad/quad_refine -> smooth.")
		.def(nb::init<>())
		.def("new_mesh", &Generator::new_mesh, "domain"_a, "mesh_id"_a = 0, "element_color"_a = 0,
			 "Creates a new, empty mesh for the given domain. The domain is kept alive by the generated mesh, so it may be dropped by the caller afterwards.")
		.def("triangulate", &Generator::triangulate, "mesh"_a, "n_elements"_a = nb::none(), "mesh_range_factor"_a = nb::none(),
			 "wide_search_factor"_a = nb::none(), "min_cell_quality"_a = nb::none(), "max_cell_angle"_a = nb::none(),
			 "base_vertex_factor"_a = nb::none(), "show_progress"_a = false,
			 "Fills the domain with triangles using the advancing front algorithm. Arguments left at None keep TQMesh's defaults; n_elements limits the number of elements to generate (None means as many as required to fill the domain).")
		.def("quad_layer", &Generator::quad_layer, "mesh"_a, "n_layers"_a, "first_height"_a, "growth_rate"_a = 1.0,
			 "start"_a = std::pair<double, double>{0.0, 0.0}, "end"_a = std::pair<double, double>{0.0, 0.0},
			 "angle_factor"_a = nb::none(), "show_progress"_a = false,
			 "Generates n_layers of quadrilateral elements along the boundary from the position start to the position end, both of which must coincide with boundary vertices. Passing the same position twice covers the entire closed boundary it belongs to. The first layer has the height first_height, each following one is larger by the growth_rate. Must be called before triangulate.")
		.def("tri2quad", &Generator::tri2quad, "mesh"_a, "Merges suitable pairs of triangles into quadrilaterals, giving a mixed mesh.")
		.def("quad_refine", &Generator::quad_refine, "mesh"_a, "Refines the mesh into an all-quadrilateral one, splitting each triangle into three and each quadrilateral into four quads.")
		.def("smooth", &Generator::smooth, "mesh"_a, "iterations"_a = 2, "kind"_a = "mixed", "epsilon"_a = nb::none(),
			 "decay"_a = nb::none(), "angle_factor"_a = nb::none(), "quad_layer_smoothing"_a = false,
			 "Improves the element quality by moving interior vertices for the given number of iterations. The kind is 'mixed', 'laplace' or 'torsion'; arguments left at None keep TQMesh's defaults.")
		.def("merge", &Generator::merge, "receiver"_a, "donor"_a,
			 "Merges the donor mesh into the receiver mesh, which requires the two to share a common interface. If successful, the donor mesh is destroyed and its handle must not be used anymore.")
		.def("write", &Generator::write, "mesh"_a, "filename"_a, "format"_a = "vtu",
			 "Writes the mesh to a file, where format is 'vtu' (ParaView), 'txt' or 'cout' (standard output). TQMesh appends the corresponding file extension to the filename itself.")
		.def_prop_ro("n_meshes", &Generator::size, "Number of meshes currently owned by this generator.");
}
