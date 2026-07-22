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

// Small helpers bridging std::vector<T> and nb::ndarray<nb::numpy, T>, used throughout
// src/nanobind/*.cpp. nanobind's ndarray is a non-owning view type (unlike pybind11's array_t,
// which could itself allocate/own a numpy-managed buffer), so returning a freshly computed
// C++ std::vector<T> as a new (Python-owned) numpy array requires explicitly moving its data
// onto the heap and attaching a capsule that frees it when the array is garbage-collected.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>
#include <algorithm>

namespace nb = nanobind;

// Wraps the contents of a std::vector<T> as a brand-new numpy array, moving its data onto the
// heap and transferring ownership to Python (freed via the capsule once the array is garbage
// collected).
template <typename T>
static inline nb::ndarray<nb::numpy, T> vector_to_ndarray(const std::vector<T> &v)
{
	T *data = new T[v.size()];
	std::copy(v.begin(), v.end(), data);
	nb::capsule owner(data, [](void *p) noexcept
					   { delete[](T *) p; });
	return nb::ndarray<nb::numpy, T>(data, {v.size()}, owner);
}

// Copies the (flattened) contents of an incoming numpy array into a std::vector<T>.
template <typename T>
static inline std::vector<T> ndarray_to_vector(const nb::ndarray<nb::numpy, T> &arr)
{
	size_t n = 1;
	for (size_t i = 0; i < arr.ndim(); i++)
		n *= arr.shape(i);
	return std::vector<T>(arr.data(), arr.data() + n);
}

// Wraps a rectangular std::vector<std::vector<T>> (all inner vectors of equal size) as a new
// 2D numpy array, flattening it onto the heap analogously to vector_to_ndarray. Callers must
// only use this for genuinely rectangular data (e.g. per-point field samples); ragged data
// (e.g. per-row CSR arrays of differing length) must instead be returned as a list of
// individually-converted 1D arrays.
template <typename T>
static inline nb::ndarray<nb::numpy, T> nested_vector_to_ndarray(const std::vector<std::vector<T>> &v)
{
	size_t rows = v.size();
	size_t cols = rows ? v[0].size() : 0;
	T *data = new T[rows * cols];
	for (size_t i = 0; i < rows; i++)
		std::copy(v[i].begin(), v[i].end(), data + i * cols);
	nb::capsule owner(data, [](void *p) noexcept
					   { delete[](T *) p; });
	return nb::ndarray<nb::numpy, T>(data, {rows, cols}, owner);
}
