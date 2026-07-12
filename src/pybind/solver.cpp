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


#ifdef OOMPH_HAS_MPI
#include "mpi.h"
#endif

#include <iostream>

#include <functional>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "../oomph_lib.hpp"

#include "../exception.hpp"


// TODO: This must agree with the setting in partitioning.h from modified oomph-lib
#define idx_t int64_t
#define real_t double

namespace py = pybind11;

// This file implements a "solver shim": oomph-lib's linear algebra / mesh-partitioning code
// calls out to external C libraries (SuperLU, SuperLU_DIST, METIS) through a handful of
// extern "C" functions (superlu(), superlu_dist_*(), METIS_PartGraphKway(), ...). Instead of
// linking the real libraries, pyoomph provides its own implementations of exactly those C
// symbols (further below, in the extern "C" block) with matching signatures, so that oomph-lib's
// calls get intercepted here and forwarded to a Python-side GeneralSolverCallback object (set via
// set_Solver_callback()). This lets pyoomph's Python solver backends (see pyoomph/solvers/*.py,
// e.g. scipy/MUMPS/Pardiso wrappers) act as the actual linear solver / graph partitioner beneath
// oomph-lib's usual SuperLU/METIS call sites, without those libraries actually having to be
// present.
namespace pyoomph
{

    typedef void *fptr;

    // Abstract callback interface (overridden in Python) that actually performs the linear
    // solves / graph partitioning requested by oomph-lib. The array arguments mirror the raw
    // pointer/size arguments of the C library functions being shimmed (see the extern "C" block
    // below), wrapped as (typically zero-copy) numpy arrays over the same underlying buffers.
    class GeneralSolverCallback
    {
    public:
        unsigned last_nrow_local;
        // Solve a serial (non-distributed) sparse linear system, mirroring SuperLU's serial
        // "superlu" driver interface. op_flag selects the requested operation (factorize/solve/
        // free, as used by oomph-lib); values/rowind/colptr describe the matrix in compressed
        // column format; rhs is overwritten with the solution.
        virtual int solve_la_system_serial(int op_flag, int n, int nnz, int nrhs,
                                           py::array_t<double> &values, py::array_t<int> &rowind, py::array_t<int> &colptr,
                                           py::array_t<double> &rhs, int ldb, int transpose) { return -1; };

        // Solve a distributed (MPI) sparse linear system, mirroring SuperLU_DIST's interface.
        // values/col_index/row_start describe the process-local rows of the matrix in compressed
        // row format; b is overwritten with the (process-local) solution.
        virtual void solve_la_system_distributed(int op_flag, int allow_permutations, int n, int nnz_local, int nrow_local, int first_row, py::array_t<double> &values, py::array_t<int> &col_index, py::array_t<int> &row_start, py::array_t<double> &b, int nprow, int npcol, int doc, py::array_t<size_t> &data, py::array_t<int> &info){}; //,MPI_Comm comm

        // Partition a graph into nparts parts, mirroring METIS's METIS_PartGraphKway(). The
        // graph is given in CSR-like form via xadj_Py/adjacency_vector_Py; part_Py is overwritten
        // with the resulting partition index of each vertex. Not implemented by default - a
        // Python override is required if graph partitioning is actually used.
        virtual int metis_partgraph_kway(idx_t nvertex,idx_t nconnection, py::array_t<idx_t> &xadj_Py, py::array_t<idx_t> &adjacency_vector_Py, py::array_t<idx_t> &vwgt_Py, idx_t nparts, py::array_t<idx_t> &options_Py, py::array_t<idx_t> &edgecut_Py, py::array_t<idx_t> &part_Py) {throw_runtime_error("METIS Link is not properly set up!"); return -1;}
    };

    // pybind11 trampoline: forwards the three virtual solve/partition calls above to Python
    // overrides.
    class PyGeneralSolverCallback : public GeneralSolverCallback
    {
    public:
        /* Inherit the constructors */
        using GeneralSolverCallback::GeneralSolverCallback;

        int solve_la_system_serial(int op_flag, int n, int nnz, int nrhs,
                                   py::array_t<double> &values, py::array_t<int> &rowind, py::array_t<int> &colptr,
                                   py::array_t<double> &rhs, int ldb, int transpose) override
        {
            PYBIND11_OVERLOAD(
                int,                                                               /* Return type */
                GeneralSolverCallback,                                             /* Parent class */
                solve_la_system_serial,                                            /* Name of function in C++ (must match Python name) */
                op_flag, n, nnz, nrhs, values, rowind, colptr, rhs, ldb, transpose /* Argument(s) */
            );
        }

        void solve_la_system_distributed(int op_flag, int allow_permutations, int n, int nnz_local, int nrow_local, int first_row, py::array_t<double> &values, py::array_t<int> &col_index, py::array_t<int> &row_start, py::array_t<double> &b, int nprow, int npcol, int doc, py::array_t<size_t> &data, py::array_t<int> &info) override
        { //,MPI_Comm comm
            PYBIND11_OVERLOAD(
                void,                                                                                                                              /* Return type */
                GeneralSolverCallback,                                                                                                             /* Parent class */
                solve_la_system_distributed,                                                                                                       /* Name of function in C++ (must match Python name) */
                op_flag, allow_permutations, n, nnz_local, nrow_local, first_row, values, col_index, row_start, b, nprow, npcol, doc, data, info); //,comm
        }

        int metis_partgraph_kway(idx_t nvertex,idx_t nconnection, py::array_t<idx_t> &xadj_Py, py::array_t<idx_t> &adjacency_vector_Py, py::array_t<idx_t> &vwgt_Py, idx_t nparts, py::array_t<idx_t> &options_Py, py::array_t<idx_t> &edgecut_Py, py::array_t<idx_t> &part_Py)
        {
            PYBIND11_OVERLOAD(
                int,                  /* Return type */
                GeneralSolverCallback, /* Parent class */
                metis_partgraph_kway,  /* Name of function in C++ (must match Python name) */
                nvertex, nconnection, xadj_Py, adjacency_vector_Py, vwgt_Py, nparts, options_Py, edgecut_Py, part_Py);
        }
    };

    // The single active callback that the extern "C" shim functions below forward to; NULL
    // until a Python GeneralSolverCallback instance is installed via set_Solver_callback().
    GeneralSolverCallback *g_solver_cb = NULL;
    void set_Solver_callback(GeneralSolverCallback *cb) { g_solver_cb = cb; }

}

// C-linkage functions with the exact names/signatures oomph-lib's linear solver / mesh
// partitioning code expects from the real SuperLU / SuperLU_DIST / METIS libraries. Since these
// are defined here instead, oomph-lib's calls resolve to pyoomph's own shims, which wrap the raw
// pointers into numpy arrays and forward everything to pyoomph::g_solver_cb (see above).
extern "C"
{
    typedef void *fptr;
    // Shim for SuperLU's serial factorize/solve driver (the actual op_flag semantics come from
    // oomph-lib's SuperLU wrapper); forwards to GeneralSolverCallback::solve_la_system_serial().
    int superlu(int *op_flag, int *n, int *nnz, int *nrhs,
                double *values, int *rowind, int *colptr,
                double *b, int *ldb, int *transpose, int *doc,
                fptr *f_factors, int *info)
    {
        py::array_t<double> values_arr;
        if (values)
            values_arr = py::array_t<double>({*nnz}, {sizeof(double)}, values, py::capsule(values, [](void *f) {}));

        py::array_t<int> rowind_arr;
        if (rowind)
            rowind_arr = py::array_t<int>({*nnz}, {sizeof(int)}, rowind, py::capsule(rowind, [](void *f) {}));

        py::array_t<int> colptr_arr;
        if (colptr)
            colptr_arr = py::array_t<int>({*n + 1}, {sizeof(int)}, colptr, py::capsule(colptr, [](void *f) {}));

        py::array_t<double> b_arr;

        if (b)
            b_arr = py::array_t<double>({*n}, {sizeof(double)}, b, py::capsule(b, [](void *f) {}));

        int nrhs_val = 0;
        if (nrhs)
            nrhs_val = *nrhs;
        int nnz_val = 0;
        if (nnz)
            nnz_val = *nnz;
        int ldb_val = 0;
        if (ldb)
            ldb_val = *ldb;

        int res = pyoomph::g_solver_cb->solve_la_system_serial(*op_flag, *n, nnz_val, nrhs_val, values_arr, rowind_arr, colptr_arr, b_arr, ldb_val, (transpose ? 1 : 0));

        *info = 0; // XXX Hack. Really check for errors here

        return res;
    }

#ifdef OOMPH_HAS_MPI
    // Shim for SuperLU_DIST's globally-replicated-matrix solve entry point; not implemented
    // (pyoomph only supports the row-distributed variant below).
    void superlu_dist_global_matrix(int opt_flag, int allow_permutations,
                                    int n, int nnz, double *values,
                                    int *row_index, int *col_start,
                                    double *b, int nprow, int npcol,
                                    int doc, void **data, int *info,
                                    MPI_Comm comm)
    {
        throw_runtime_error("SUPERLU DIST IMPLEM GLOBAL MATRIX");
    }

    // Shim for SuperLU_DIST's row-distributed matrix solve entry point; forwards to
    // GeneralSolverCallback::solve_la_system_distributed(). b is wrapped with
    // g_solver_cb->last_nrow_local entries: on a factorize call (opt_flag==1) that count is
    // (re-)captured from nrow_local first, so a later solve-only call (which may not know
    // nrow_local itself) still wraps b with the correct length.
    void superlu_dist_distributed_matrix(int opt_flag, int allow_permutations,
                                         int n, int nnz_local,
                                         int nrow_local, int first_row,
                                         double *values, int *col_index,
                                         int *row_start, double *b,
                                         int nprow, int npcol,
                                         int doc, void **data, int *info,
                                         MPI_Comm comm)
    {
        py::array_t<double> Py_values;
        if (values)
            Py_values = py::array_t<double>({nnz_local}, {sizeof(double)}, values, py::capsule(values, [](void *f) {}));

        py::array_t<int> Py_col_index;
        if (col_index)
            Py_col_index = py::array_t<int>({nnz_local}, {sizeof(int)}, col_index, py::capsule(col_index, [](void *f) {})); // TODO: NS

        py::array_t<int> Py_row_start;
        if (row_start)
            Py_row_start = py::array_t<int>({nrow_local + 1}, {sizeof(int)}, row_start, py::capsule(row_start, [](void *f) {})); // TODO: NS

        if (opt_flag == 1)
            pyoomph::g_solver_cb->last_nrow_local = nrow_local;
        py::array_t<double> Py_b;
        if (b)
            Py_b = py::array_t<double>({pyoomph::g_solver_cb->last_nrow_local}, {sizeof(double)}, b, py::capsule(b, [](void *f) {}));

        size_t *conv_data = ((size_t **)(data))[0];
        py::array_t<size_t> Py_Data;
        if (conv_data)
            Py_Data = py::array_t<size_t>({1}, {sizeof(size_t)}, conv_data, py::capsule(conv_data, [](void *f) {}));

        py::array_t<int> Py_Info;
        if (info)
            Py_Info = py::array_t<int>({1}, {sizeof(int)}, info, py::capsule(info, [](void *f) {}));

        pyoomph::g_solver_cb->solve_la_system_distributed(opt_flag, allow_permutations, n, nnz_local, nrow_local, first_row, Py_values, Py_col_index, Py_row_start, Py_b, nprow, npcol, doc, Py_Data, Py_Info); //,comm


    }

    // Shim for SuperLU_DIST's compressed-row-to-compressed-column conversion helper; not
    // implemented (pyoomph's Python solvers work directly with compressed-row data).
    void superlu_cr_to_cc(int nrow, int ncol, int nnz, double *cr_values,
                          int *cr_index, int *cr_start, double **cc_values,
                          int **cc_index, int **cc_start)
    {
        throw_runtime_error("SUPERLU DIST IMPLEM CR TO CC");
    }


    // Shim for METIS's k-way graph partitioning entry point (used by oomph-lib to partition the
    // mesh's element connectivity graph across MPI processes). Validates the handful of
    // parameters oomph-lib is known to pass (e.g. vsize/adjwgt/tpwgts/ubvec unused) and forwards
    // the rest to GeneralSolverCallback::metis_partgraph_kway().
    int METIS_PartGraphKway(idx_t * nvtxs,idx_t * ncon,idx_t * xadj,idx_t * adjncy,idx_t * vwgt,idx_t * vsize,idx_t * adjwgt,idx_t * nparts,real_t * tpwgts,real_t * ubvec,idx_t * options,idx_t * edgecut,idx_t * part)
    //void METIS_PartGraphKway(int *nvertex_pt, int *xadj, int *adjacency_vector, int *vwgt, int *adjwgt, int *wgtflag_pt, int *numflag_pt, int *nparts_pt, int *options, int *edgecut, int *part)
    {
        idx_t nvertex = *nvtxs; //=total_number_of_root_elements
        idx_t nconnection=*ncon; // number of coennections
        py::array_t<idx_t> xadj_Py; // xadj [total_number_of_root_elements+1]
        if (xadj)
            xadj_Py = py::array_t<idx_t>({nvertex + 1}, {sizeof(idx_t)}, xadj, py::capsule(xadj, [](void *f) {}));

        py::array_t<idx_t> adjacency_vector_Py; // adjacency_vector // [xadj[-1]]
        if (adjncy)
            adjacency_vector_Py = py::array_t<idx_t>({xadj[nvertex]}, {sizeof(idx_t)}, adjncy, py::capsule(adjncy, [](void *f) {}));

        py::array_t<idx_t> vwgt_Py; // vwgt //Assembly times [total_number_of_root_elements]
        if (vwgt)
            vwgt_Py = py::array_t<idx_t>({nvertex}, {sizeof(idx_t)}, vwgt, py::capsule(vwgt, [](void *f) {}));
        
        // vsize=NULL in oomph-lib, so we don't need to convert it to a numpy array
        if (vsize)
            throw_runtime_error("METIS IMPLEM: vsize should be NULL");        
        // *adjwgt=0 in oomph-lib, so we don't need to convert it to a numpy array
        if (adjwgt)
            throw_runtime_error("METIS IMPLEM: *adjwgt should be 0");
        if (*nparts<=0)
            throw_runtime_error("METIS IMPLEM: nparts should be > 0");
        if (tpwgts)
            throw_runtime_error("METIS IMPLEM: tpwgts should be NULL");
        if (ubvec)            
            throw_runtime_error("METIS IMPLEM: ubvec should be NULL");    

        
        py::array_t<idx_t> options_Py; // options 
        if (options)
            options_Py = py::array_t<idx_t>({40}, {sizeof(idx_t)}, options, py::capsule(options, [](void *f) {}));

        py::array_t<idx_t> edgecut_Py; // edgecut [ = total_number_of_root_elements]
        if (edgecut)
            edgecut_Py = py::array_t<idx_t>({nvertex}, {sizeof(idx_t)}, edgecut, py::capsule(edgecut, [](void *f) {}));

        py::array_t<idx_t> part_Py; // part : partition
        if (part)
            part_Py = py::array_t<idx_t>({nvertex}, {sizeof(idx_t)}, part, py::capsule(part, [](void *f) {}));
        return pyoomph::g_solver_cb->metis_partgraph_kway(nvertex,nconnection, xadj_Py, adjacency_vector_Py, vwgt_Py, *nparts, options_Py, edgecut_Py, part_Py);
    }


    // Shim for METIS's alternative "VKway" (minimal-communication-volume) partitioner variant;
    // not implemented (only the k-way edge-cut variant above is supported).
    void METIS_PartGraphVKway(int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *)
    {
        throw_runtime_error("METIS IMPLEM: METIS_PartGraphVKway");
    }

#endif
}

void PyReg_Solvers(py::module &m)
{
    py::class_<pyoomph::GeneralSolverCallback, pyoomph::PyGeneralSolverCallback /* <--- trampoline*/>(
        m, "GeneralSolverCallback",
        "Base class, to be subclassed in Python, that implements the actual sparse linear solves "
        "and/or graph partitioning oomph-lib requests through its (shimmed) SuperLU/SuperLU_DIST/"
        "METIS call sites. Install an instance via set_Solver_callback() to make it active.")
        .def(py::init<>())
        .def("metis_partgraph_kway", &pyoomph::GeneralSolverCallback::metis_partgraph_kway,
             py::arg("nvertex"), py::arg("nconnection"), py::arg("xadj"), py::arg("adjacency_vector"), py::arg("vwgt"),
             py::arg("nparts"), py::arg("options"), py::arg("edgecut"), py::arg("part"),
             "Partition a graph with ``nvertex`` vertices into ``nparts`` parts using a METIS-compatible k-way partitioning "
             "interface; must fill ``part`` with the resulting partition index of each vertex.")
        .def("solve_la_system_distributed", &pyoomph::GeneralSolverCallback::solve_la_system_distributed,
             py::arg("op_flag"), py::arg("allow_permutations"), py::arg("n"), py::arg("nnz_local"), py::arg("nrow_local"), py::arg("first_row"),
             py::arg("values"), py::arg("col_index"), py::arg("row_start"), py::arg("b"), py::arg("nprow"), py::arg("npcol"), py::arg("doc"),
             py::arg("data"), py::arg("info"),
             "Solve (or factorize, depending on ``op_flag``) a distributed (MPI, row-partitioned) sparse linear system given "
             "in compressed row format by ``values``/``col_index``/``row_start``, overwriting ``b`` with the solution.")
        .def("solve_la_system_serial", &pyoomph::GeneralSolverCallback::solve_la_system_serial,
             py::arg("op_flag"), py::arg("n"), py::arg("nnz"), py::arg("nrhs"), py::arg("values"), py::arg("rowind"), py::arg("colptr"),
             py::arg("b"), py::arg("ldb"), py::arg("transpose"),
             "Solve (or factorize, depending on ``op_flag``) a serial (non-distributed) sparse linear system given in "
             "compressed column format by ``values``/``rowind``/``colptr``, overwriting ``b`` with the solution.");

    m.def(
        "set_Solver_callback", [](pyoomph::GeneralSolverCallback *cb)
        { pyoomph::g_solver_cb = cb; },
        py::arg("callback"), "Install ``callback`` as the active GeneralSolverCallback used by oomph-lib's (shimmed) SuperLU/METIS call sites.");
    m.def(
        "get_Solver_callback", []()
        { return pyoomph::g_solver_cb; },
        py::return_value_policy::reference, "Return the currently installed GeneralSolverCallback, or None if none has been set.");

    m.def(
        "csr_rows_to_coo_rows", [](const py::array_t<int> &csr_rows, unsigned nzz, unsigned first_row)
        {
            auto coo_rows = py::array_t<int>({nzz});
            int *in_buf = (int *)csr_rows.request().ptr;
            int *res_buff = (int *)coo_rows.request().ptr;
            unsigned i_row = 0;
            for (unsigned count = 0; count < nzz; count++)
            {
                if (count < (unsigned)in_buf[i_row + 1])
                {
                    res_buff[count] = first_row + i_row;
                }
                else
                {
                    i_row++;
                    res_buff[count] = first_row + i_row;
                }
            }
            return coo_rows;
        },
        py::arg("csr_rows"), py::arg("nnz"), py::arg("first_row") = 0,
        "Expand a compressed-sparse-row ``row_start`` array (``csr_rows``, length nrow+1) into a plain row-index array of "
        "length ``nnz`` in coordinate (COO) format, i.e. one entry per non-zero giving its (``first_row``-offset) row index.");
}
