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


#ifndef PYOOMPH_JIT_BRIDGE_H_G_
#define PYOOMPH_JIT_BRIDGE_H_G_

#if defined _MSC_VER
#pragma warning(disable : 4018 4005 4996 4101)
#endif

#ifndef PYOOMPH_TCC_TO_MEMORY
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>

#else

double acos(double);
double asin(double);
double atan(double);
double atan2(double, double);
double acosh(double);
double asinh(double);
double atanh(double);
double cos(double);
double sin(double);
double tan(double);
double cosh(double);
double sinh(double);
double tanh(double);
double exp(double);
double log(double);
double pow(double, double);
double log10(double);
double sqrt(double);
double fabs(double);
double fmax(double, double);
double fmin(double, double);
// tcc is called with -nostdinc, so every libm function used by generated code must be declared here. A missing one is
// not a compile error but an implicit int-returning declaration, i.e. silently garbled results (erf/erfc used to be).
double erf(double);
double erfc(double);

long long unsigned int strlen(const char *);
char *strdup(const char *);
void *malloc(size_t);
void *free(void *);
void *calloc(size_t, size_t);
char *strncpy(char *, const char *, size_t);

#define bool _Bool
#define true 1
#define false 0
#define __bool_true_false_are_defined 1

// Just a hack since tccbox currently has issues with these. They are used internally by TCC for an unary minus operation.
double __mzerodf=-0.0;
float __mzerosf=-0.0;

#endif


#ifndef NULL
#define PYOOMPH_NULL (void *)0
#else
#define PYOOMPH_NULL NULL
#endif

//#define PYOOMPH_RESTRICT __restrict
#define PYOOMPH_RESTRICT  // Does not really help so far, but also not completed in e.g. the args for e.g. Hessian, etc

// This file defines the structures which are required to transfer the oomph-lib data (e.g. shape functions) to the C-compiled code

#define NUM_CONTINUOUS_SPACES 4 // C2TB,C2,C1TB,C1
#define SPACE_INDEX_C2TB 0
#define SPACE_INDEX_C2 1
#define SPACE_INDEX_C1TB 2
#define SPACE_INDEX_C1 3
// Same for the DG variants
#define SPACE_INDEX_D2TB 0
#define SPACE_INDEX_D2 1
#define SPACE_INDEX_D1TB 2
#define SPACE_INDEX_D1 3

// Second spatial derivatives of the shape functions.
//
// Storage is the full square [i][j], not the symmetry-packed form oomph-lib uses in N2deriv. That is
// deliberate: the second derivative is only symmetric when the element has no codimension. pyoomph
// builds spatial derivatives from the metric (dpsi/dx_i = g^{ab} t_{a,i} dpsi/ds_b), which on a
// surface gives the tangential derivative of the surface gradient,
//    D_ij psi = M_i^b M_j^c psi_,bc + M_j^c [ (d g^{ab}/d s_c) t_{a,i} + g^{ab} x_{i,ac} ] psi_,b ,
// and that is genuinely asymmetric - on a unit circle it comes out as t_i t_j psi'' - n_i t_j psi'.
// (Its trace is still exactly the Laplace-Beltrami operator, since the asymmetric part is
// trace-free, so div(grad(u)) on an interface is correct.) Packing would silently alias the two
// orders. The symmetry is exploited one level up instead: the code generator canonicalises
// d/dx_i d/dx_j to i<=j only on domains where element_dim == nodal_dimension, so codimension-free
// problems still emit only 6 of the 9 interpolation loops.
//
// The first index is the INNER derivative direction, the second the OUTER differentiation
// direction, i.e. slot(i,j) addresses d/dx_j ( d psi / d x_i ).
#define PYOOMPH_MAX_NODAL_DIM 3
#define MAX_N2DERIV (PYOOMPH_MAX_NODAL_DIM * PYOOMPH_MAX_NODAL_DIM)
#define PYOOMPH_D2_SLOT(i, j) (PYOOMPH_MAX_NODAL_DIM * (i) + (j))

typedef struct JITElementInfo
{

  unsigned int nnode; // Total number of nodes
  unsigned int nnode_of_space[NUM_CONTINUOUS_SPACES]; // Number of nodes per type (C2TB,C2,C1TB,C1) // Set during problem level
  
     
  unsigned int nnode_DL;   // This are actually not nodes, but internal data (since discontinous)
  // unsigned int nnode_D0;  //This are actually not nodes, but internal data (since discontinous), This is always 1
  unsigned int nodal_dim; // Nodal dimension

  //double  * PYOOMPH_RESTRICT  * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT nodal_coords; // Nodal coordinates (node index, xindex, time index)
  //double  * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT nodal_data;   // Nodal data (node index,data index, time index)
  double  ***  nodal_coords; // Nodal coordinates (node index, xindex, time index)
  double  *** nodal_data;   // Nodal data (node index,data index, time index)

  
  //int * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT nodal_local_eqn; // Nodal equations (node index, data index)
  //int * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT pos_local_eqn;   // Nodal equations (node index, data index)
  int ** nodal_local_eqn; // Nodal equations (node index, data index)
  int ** pos_local_eqn;   // Nodal equations (node index, data index)

  // bool * nullified_residual_dof;

  // double **extdata_value;		//External data values (ext data index,  time index)
  // int * ext_local_eqn; //External equations (ext data index, value index)

  // double *** direct_nodal_data; //Nodal data without consideration of hanging nodes (is used as backup direct_nodal_data[nodeindex]=nodal_data[nodindex] for non-hanging
  unsigned int ndof; // Number of local dofs

  bool alloced;
  void *elem_ptr; // Pointer to the element //TODO: This is problematic, as the this pointer cannot be restored for multiple inheritance

  struct JITElementInfo * PYOOMPH_RESTRICT bulk_eleminfo;
  // struct JITElementInfo * otherbulk_eleminfo;
  struct JITElementInfo * PYOOMPH_RESTRICT opposite_eleminfo;
} JITElementInfo_t;


//Is a bit faster with static arrays, but does not really pay off...
//#define FIXED_SIZE_SHAPE_BUFFER

#ifdef FIXED_SIZE_SHAPE_BUFFER

#define MAX_NODES  32
#define MAX_NODAL_DIM  3
#define MAX_TIME_WEIGHTS  7
#define MAX_HANG 16
#define MAX_FIELDS 32
#define MAX_RESIDUAL_DESTINATIONS 16
#define ARRAY_DECL_NDIM(what) what[MAX_NODAL_DIM]
#define ARRAY_DECL_UNITY(what) what[1]
#define ARRAY_DECL_NNODE(what) what[MAX_NODES]
#define ARRAY_DECL_NDT(what) what[MAX_TIME_WEIGHTS]
#define ARRAY_DECL_NHANG(what) what[MAX_HANG]
#define ARRAY_DECL_NFIELDS(what) what[MAX_FIELDS]
#define ARRAY_DECL_RESIDUAL_DESTINATION(what) what[MAX_RESIDUAL_DESTINATIONS]
#define ARRAY_DECL_N2DERIV(what) what[MAX_N2DERIV]
#define DX_SHAPE_FUNCTION_DECL(what) const double(*what)[MAX_NODAL_DIM]
#define D2X_SHAPE_FUNCTION_DECL(what) const double(*what)[MAX_N2DERIV]

#else

#define ARRAY_DECL_NDIM(what) * PYOOMPH_RESTRICT what
#define ARRAY_DECL_UNITY(what) * PYOOMPH_RESTRICT what
#define ARRAY_DECL_NNODE(what) * PYOOMPH_RESTRICT what
#define ARRAY_DECL_NDT(what) * PYOOMPH_RESTRICT what
#define ARRAY_DECL_NHANG(what) *PYOOMPH_RESTRICT what
#define ARRAY_DECL_NFIELDS(what) * PYOOMPH_RESTRICT what
#define ARRAY_DECL_RESIDUAL_DESTINATION(what) * PYOOMPH_RESTRICT what
#define ARRAY_DECL_N2DERIV(what) * PYOOMPH_RESTRICT what
#define DX_SHAPE_FUNCTION_DECL(what) double * const * const  PYOOMPH_RESTRICT what
#define D2X_SHAPE_FUNCTION_DECL(what) double * const * const  PYOOMPH_RESTRICT what
#endif


typedef struct JITHangInfoEntry
{
  double weight;
  int local_eqn; // Replaced equation
  // double **master_coordinate; //Coordinate (x/y/z, time index)
} JITHangInfoEntry_t;

typedef struct JITHangInfo
{
  int nummaster;
  JITHangInfoEntry_t ARRAY_DECL_NHANG(masters); // 0..nummasters-1
} JITHangInfo_t;



typedef struct JITShapeInfo
{
  unsigned int n_int_pt;             // Number of integration points
  double int_pt_weight[3];            // Eulerian weight at the current integration point (or at history steps 1 and 2)
  double int_pt_weight_Lagrangian; // Lagrangian weight at the current integration point
  double int_pt_weight_unity;            // Weight at the current integration point in s space, i.e. without any mapping [ sqrt(det(g_ab)) ]
  double ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(int_pt_weights_d_coords)); // Weights derived by coordinates, [i_dim,l_node], i.e. w*dJ_Eulerian/dX^l_i
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT int_pt_weights_d2_coords; // Weights derived by coordinates, [i_dim,j_dim,l_node_i,l_node_j], i.e. w*d2J_Eulerian/(dX^l_i*dX^l_j)
  
  double elemsize_Eulerian[3],elemsize_Eulerian_cartesian[3];      // Eulerian element size (history index), with e.g. 2*pi*r in integration or not
  double elemsize_Lagrangian,elemsize_Lagrangian_cartesian; // Lagrangian element size
  double ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(elemsize_d_coords)); // Eulerian element size derived by coordinates, [i_dim,l_node], i.e. sum(w*dJ_Eulerian)/dX^l_i
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT elemsize_d2_coords; // Weights derived by coordinates, [i_dim,j_dim,l_node_i,l_node_j], i.e. sum(w*d2J_Eulerian)/(dX^l_i*dX^l_j)      
  // Cartesian variants
  double ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(elemsize_Cart_d_coords)); // Eulerian element size derived by coordinates, [i_dim,l_node], i.e. sum(w*dJ_Eulerian)/dX^l_i
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT  elemsize_Cart_d2_coords; // Weights derived by coordinates, [i_dim,j_dim,l_node_i,l_node_j], i.e. sum(w*d2J_Eulerian)/(dX^l_i*dX^l_j)  


  double ARRAY_DECL_NNODE(shapes)[NUM_CONTINUOUS_SPACES];               // non-derived shapes
  // First index is the history level, exactly like int_pt_weight[] above: 0 is the current
  // configuration, 1 and 2 are the previous ones, used by evaluate_in_past(...,apply_on_others=True).
  // Only the EULERIAN derivative needs this - the undifferentiated shapes and the Lagrangian/local
  // coordinate derivatives are properties of the reference element and do not move with the mesh.
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dx_shapes))[3][NUM_CONTINUOUS_SPACES]; // Derived shapes (history, space, node index, coord index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dX_shapes))[NUM_CONTINUOUS_SPACES]; // Corresponding Lagrangian version
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dS_shapes))[NUM_CONTINUOUS_SPACES]; // Corresponding local coordinate version
  // Second spatial derivatives, slot index built with PYOOMPH_D2_SLOT(inner dir, outer dir).
  // d2x_shapes carries the history level like dx_shapes above, for evaluate_in_past(...,apply_on_others=True).
  double ARRAY_DECL_N2DERIV(ARRAY_DECL_NNODE(d2x_shapes))[3][NUM_CONTINUOUS_SPACES]; // (history, space, node index, slot)
  double ARRAY_DECL_N2DERIV(ARRAY_DECL_NNODE(d2S_shapes))[NUM_CONTINUOUS_SPACES];    // Corresponding local coordinate version (space, node index, slot)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(d_dx_shape_dcoord))))[NUM_CONTINUOUS_SPACES]; // derivative of dx_shape w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index)
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2_dx2_shape_dcoord[NUM_CONTINUOUS_SPACES]; // second derivative of dx_shape_C2 w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index,deriv. coord node index2, deriv coord dir index2)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(ARRAY_DECL_N2DERIV(ARRAY_DECL_NNODE(d_d2x_shape_dcoord))))[NUM_CONTINUOUS_SPACES]; // derivative of d2x_shape w/r to nodal coords (node index, slot, deriv. coord node index, deriv coord dir index)
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2_d2x2_shape_dcoord[NUM_CONTINUOUS_SPACES]; // second derivative of d2x_shape w/r to nodal coords (node index, slot, deriv. coord node index, deriv coord dir index, deriv. coord node index2, deriv coord dir index2)

  double ARRAY_DECL_NNODE(shape_DL);                // DL shapes (node index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dx_shape_DL))[3];         // DL shapes (history, node index, coord index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dX_shape_DL));            // Corresponding Lagrangian derivatives
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dS_shape_DL));            // Corresponding local coordinate version
  double ARRAY_DECL_N2DERIV(ARRAY_DECL_NNODE(d2x_shape_DL))[3];     // DL second spatial derivatives (history, node index, slot)
  double ARRAY_DECL_N2DERIV(ARRAY_DECL_NNODE(d2S_shape_DL));        // Corresponding local coordinate version
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(d_dx_shape_dcoord_DL)))); // derivative of dx_shape_DL w/r to nodal coords (intpt,node index, coord index, deriv. coord node index, deriv coord dir index)
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2_dx2_shape_dcoord_DL; // second derivative of dx_shape_DL w/r to nodal coords (intpt,node index, coord index, deriv. coord node index, deriv coord dir index,deriv. coord node index2, deriv coord dir index2)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(ARRAY_DECL_N2DERIV(ARRAY_DECL_NNODE(d_d2x_shape_dcoord_DL)))); // derivative of d2x_shape_DL w/r to nodal coords (node index, slot, deriv. coord node index, deriv coord dir index)
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2_d2x2_shape_dcoord_DL; // second derivative of d2x_shape_DL w/r to nodal coords (node index, slot, deriv. coord node index, deriv coord dir index, deriv. coord node index2, deriv coord dir index2)

  #ifdef FIXED_SIZE_SHAPE_BUFFER
  double *shape_Pos; // Pos space shapes. These will be mapped to the dominant element space
  double (*dx_shape_Pos[3])[MAX_NODAL_DIM];
  double (*dX_shape_Pos)[MAX_NODAL_DIM];
  double (*dS_shape_Pos)[MAX_NODAL_DIM];
  double (*d2x_shape_Pos[3])[MAX_N2DERIV];
  double (*d2S_shape_Pos)[MAX_N2DERIV];
  double (*d_dx_shape_dcoord_Pos)[MAX_NODAL_DIM][MAX_NODES][MAX_NODAL_DIM];
  double (*d_d2x_shape_dcoord_Pos)[MAX_N2DERIV][MAX_NODES][MAX_NODAL_DIM];
  #else
  double * PYOOMPH_RESTRICT shape_Pos; // Pos space shapes. These will be mapped to the dominant element space
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT dx_shape_Pos[3];
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT dX_shape_Pos;
  double * PYOOMPH_RESTRICT* PYOOMPH_RESTRICT dS_shape_Pos;
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2x_shape_Pos[3];
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2S_shape_Pos;
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d_dx_shape_dcoord_Pos;
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d_d2x_shape_dcoord_Pos;
  #endif
   double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT* PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2_dx2_shape_dcoord_Pos; // second derivative of dx_shape_DL w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index,deriv. coord node index2, deriv coord dir index2)
   double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT* PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2_d2x2_shape_dcoord_Pos; // second derivative of d2x_shape w/r to nodal coords (node index, slot, deriv. coord node index, deriv coord dir index, deriv. coord node index2, deriv coord dir index2)

  // double ** shape_D0; //DL shapes (intpt, "node" index) -> Actually always 1 //TODO: Simplify this
  // double *** dx_shape_D0; //DL shapes (intpt, "node" index,coord index) -> Actually always zero //TODO: Simplify this

  unsigned int jacobian_size;
  unsigned int mass_matrix_size;

  double ARRAY_DECL_NDIM(normal)[3];         // (history, direction)
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT* PYOOMPH_RESTRICT d_normal_dcoord; // Derivative of the normal wrt. nodal coordinates [dir][coord node][coord dir]
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2_normal_d2coord; // Second order derivative of the normal wrt. nodal coordinates [dir][coord node 1][coord dir 1][coord node 2][coord dir 2]

  // First SPATIAL derivative of the normal, dn_i/dx_j - i.e. minus the second fundamental form, whose
  // trace div(n) is the mean curvature. Built as -M_i^(c) M_j^(b) n_k X_{k,bc} (see
  // fill_shape_info_at_s), hence symmetric in i,j and independent of the normal's orientation
  // convention. The history index matches normal[] above.
  //
  // The "coord node" index of the sensitivities below is the node of the BULK element, exactly as for
  // d_normal_dcoord: on an interface the normal's coordinate dependence is expressed in terms of the
  // parent element's nodes, and the generated code loops over that same set.
  double ARRAY_DECL_NDIM(ARRAY_DECL_NDIM(dnormal_dx))[3];                        // [history][normal comp i][spatial dir j]
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d_dnormal_dx_dcoord;  // [i][j][coord node][coord dir]
  double * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT * PYOOMPH_RESTRICT d2_dnormal_dx_d2coord; // [i][j][coord node][coord dir][coord node 2][coord dir 2]

  // double * dx_shape_at_center_C1; //Gradients of C1 space at center //Required for SUPG

  double ARRAY_DECL_NDT(t);
  double ARRAY_DECL_NDT(dt);                           // Current time buffer and desired [and history] time steps
  unsigned int timestepper_ntstorage;       // Number of timestepper weights
  double ARRAY_DECL_NDT(timestepper_weights_dt_BDF1);      // Weights for calculating \partial_t
  double ARRAY_DECL_NDT(timestepper_weights_dt_BDF2);      // Weights for calculating \partial_t
  double ARRAY_DECL_NDT(timestepper_weights_dt_Newmark2);  // Weights for calculating \partial_t
  double ARRAY_DECL_NDT(timestepper_weights_d2t_Newmark2); // Weights for calculating \partial^2_t

  // Possibly degraded variants
  double * PYOOMPH_RESTRICT timestepper_weights_dt_BDF2_degr;
  double * PYOOMPH_RESTRICT timestepper_weights_dt_Newmark2_degr;

  JITHangInfo_t ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(hanginfo_Pos));  
  JITHangInfo_t ARRAY_DECL_NFIELDS(ARRAY_DECL_NNODE(hanginfo)); // Hang info for the nodal_data buffer 
   
  


  struct JITShapeInfo * PYOOMPH_RESTRICT bulk_shapeinfo;
  // struct JITShapeInfo * otherbulk_shapeinfo; //Bulk element on the other side
  struct JITShapeInfo * PYOOMPH_RESTRICT opposite_shapeinfo; // Shape info on the other side
} JITShapeInfo_t;

typedef void (*JITFuncSpec_ResidualAndJacobian_FiniteElement)(const JITElementInfo_t *, const JITShapeInfo_t *, double * PYOOMPH_RESTRICT, double * PYOOMPH_RESTRICT, double * PYOOMPH_RESTRICT, unsigned);
typedef void (*JITFuncSpec_HessianVectorProduct_FiniteElement)(const JITElementInfo_t *, const JITShapeInfo_t *, const double *, double *, double *, unsigned, unsigned);
typedef void (*JITFuncSpec_GetZ2Fluxes_FiniteElement)(const JITElementInfo_t *, const JITShapeInfo_t *, double *);

typedef double (*JITFuncSpec_InitialCondition_FiniteElement)(const JITElementInfo_t *, int, double *, double *, double *,double, int, double);
typedef double (*JITFuncSpec_DirichletCondition_FiniteElement)(const JITElementInfo_t *, int, double *, double *, double *,double, double);
typedef double (*JITFuncSpec_EvalIntegralExpr_FiniteElement)(const JITElementInfo_t *, const JITShapeInfo_t *, unsigned);
/* Tracer advection: (eleminfo, shapeinfo, index, result). One registered entry per tracer name AND
   per nodal time-history level - the caller blends the levels itself, since the blending weights
   follow from t(0),t(1),t(2), which are not known when this code is generated. */
typedef void (*JITFuncSpec_EvalTracerAdvection_FiniteElement)(const JITElementInfo_t *, const JITShapeInfo_t *, unsigned, double *);

typedef double (*JITFuncSpec_GeometricJacobian)(const JITElementInfo_t *, const double *);

typedef void (*JITFuncSpec_GeometricJacobianSpatialDerivative)(const JITElementInfo_t *, const double *, double *);

// d2X_psi (Lagrangian second derivatives) is reserved but not implemented - the code generator
// refuses to set it. It is declared now so that adding it later is not a second ABI break.
// There is deliberately no d2S_psi: local-coordinate derivatives have no flag of their own either.
// dS_shapes rides on dX_psi - D1XBasisFunctionLocalCoord derives from D1XBasisFunctionLagr, so
// mark_shapes_required classifies it as a Lagrangian derivative - and d2S_shapes rides on d2x_psi.
typedef struct JITFuncSpec_RequiredShapes_For_Space
{
  bool psi,dx_psi,dX_psi,d2x_psi,d2X_psi;
} JITFuncSpec_RequiredShapes_For_Space_t;

typedef struct JITFuncSpec_RequiredShapes_FiniteElement
{
  JITFuncSpec_RequiredShapes_For_Space_t Pos;              // Position space. This is always the dominant element space, i.e. C2TB>C2>C1TB>C1. If an element has a "C2" and a "C1TB" space, it will be C2TB.
    

  JITFuncSpec_RequiredShapes_For_Space_t continuous_spaces[NUM_CONTINUOUS_SPACES]; // C2TB,C2,C1TB,C1
          
  JITFuncSpec_RequiredShapes_For_Space_t DL;    
  JITFuncSpec_RequiredShapes_For_Space_t D0;      
    
  bool normal;
  // Separate from `normal`, because dn_i/dx_j needs the GEOMETRY's second local derivatives, which
  // are otherwise only computed when some field asks for d2x_psi. A plain var("normal") must not
  // start paying for d2shape_local.
  bool normal_deriv;
  bool elemsize_Eulerian,elemsize_Lagrangian;
  bool elemsize_Eulerian_cartesian,elemsize_Lagrangian_cartesian;  

  bool history_integral_dx1;
  bool history_integral_dx2;
  // Whether the geometry itself (Eulerian shape derivatives, normal, element size) is needed on the
  // mesh as it was 1 or 2 steps ago, i.e. whether the [1]/[2] slots have to be filled at all.
  bool history_geometry1;
  bool history_geometry2;
  struct JITFuncSpec_RequiredShapes_FiniteElement *bulk_shapes;
  struct JITFuncSpec_RequiredShapes_FiniteElement *opposite_shapes;
  // struct JITFuncSpec_RequiredShapes_FiniteElement * otherbulk_shapes;
} JITFuncSpec_RequiredShapes_FiniteElement_t;

typedef struct JITFuncSpec_Callback_Entry
{
  char *idname;
  unsigned unique_id;
  int is_deriv_of;
  int deriv_index;
  void *cb_obj;
} JITFuncSpec_Callback_Entry_t;

typedef struct JITFuncSpec_MultiRet_Entry
{
  char *idname;
  unsigned unique_id;
  void *cb_obj;
} JITFuncSpec_MultiRet_Entry_t;


typedef struct JITFuncSpec_Table_FiniteElement_SpaceInfo
{
  // numfields are the total number of fields, numfields_bulk are the ones which are defined on the 
  //   bulk mesh (including the additional field of all parent interface meshes)
  // numfields_basebulk are indeed the fields that are directly implemented only at the lowest level. 
  // numfields__new are the number of fields defined directly at this element level, i.e for deepest level
  //   ON BULK ELEMENTS (lowest level): numfields_new=numfields_bulk=numfields_basebulk=numfields 
  //   ON INTERFACE ELEMENTS:           numfields_new=numfields-numfields_bulk;
 unsigned int numfields,numfields_bulk,numfields_basebulk,numfields_new;
 char **fieldnames;
 int hangindex;

 unsigned int buffer_offset_basebulk; // Offsets in the nodal data buffer (basebulk fields only)
 unsigned int buffer_offset_interf; // Offsets in the nodal data buffer (interface fields only)
 // For continuous fields
 unsigned int nodal_offset_basebulk; //Offsets to the indices in the nodal values (basebulk fields only)
 // For discontinuous fields
 unsigned int internal_offset_new; // Offset to the internal_data entries. These are only there on the current element level  
 unsigned int external_offset_bulk; // Offset to the external_data entries. These refer to DG spaces on parent elements    
 char space_name[16]; // Name of the space, e.g. C1, C2, C1TB, C2TB, DL, D0, ED0

 unsigned int * interface_dof_indices; // For continuous fields (C2TB-C1), this is of length numfields-numfields_basebulk and gives the index for additional dofs on interface nodes. Created at problem level

 // Parallel to fieldnames: the index into contribution_names (and hence the row/column class of
 // contributes_to_jacobian / contributes_to_mass_matrix) that each field of this space belongs to.
 // -2 for a field that is present but takes part in NO contribution of this code, which is a
 // POSITIVE statement: that field's row and column of this element's block are empty, so nothing has
 // to be stored for them. Distinguished from -1, which means "could not be attributed" and must be
 // read as coupled to everything -- conflating the two is what forced a structural zero onto the
 // diagonal of every unclassifiable dof. Lets an element translate a local dof into the class the
 // symbolic coupling tables are indexed by; see dev_docs/structural_assembly.md.
 int * field_contribution_index;

 unsigned space_index; // Index to the arrays [4]
 bool is_dominant; // Is this the dominant space for the element, i.e. the geometric space where also the coordinates live? (e.g. C2TB>C2>C1TB>C1) // Set during problem level 

} JITFuncSpec_Table_FiniteElement_SpaceInfo_t;


/* Per-block property bits for jacobian_block_flags / mass_matrix_block_flags below.
   A SET bit means "proven at code generation time from the symbolic block expression"; an UNSET bit
   means "not proven", never "disproven" -- a consumer may exploit a set bit and must assume nothing
   from a clear one. Kept in sync with the module attributes exported in src/nanobind/mesh.cpp
   (_pyoomph_core.JACOBIAN_BLOCK_*). */
#define JACOBIAN_BLOCK_SYMMETRIC 1u         /* block (i,j) == +transpose of block (j,i); set on BOTH mirror entries (diagonal blocks: self-transpose) */
#define JACOBIAN_BLOCK_ANTISYMMETRIC 2u     /* block (i,j) == -transpose of block (j,i); set on BOTH mirror entries */
#define JACOBIAN_BLOCK_CONSTANT 4u          /* entries independent of unknowns, nodal positions, global parameters, time AND time-stepper weights */
#define JACOBIAN_BLOCK_CONSTANT_FIXED_DT 8u /* as CONSTANT, but may contain time-stepper (BDF/Newmark) weights, i.e. constant as long as dt is; implied by CONSTANT */

typedef struct JITFuncSpec_Table_FiniteElement
{
  unsigned int nodal_dim, lagr_dim;


  // Filled at problem level
  unsigned int total_num_fields; // Including all fields, DG, interfaces, DL, D0, but not ED0
  unsigned int total_num_fields_basebulk; // Only the continuous fields, i.e. C2TB,C2,C1TB,C1
  

  // New way of handling things 
  JITFuncSpec_Table_FiniteElement_SpaceInfo_t info_Pos; 
  JITFuncSpec_Table_FiniteElement_SpaceInfo_t continuous_spaces[NUM_CONTINUOUS_SPACES]; // C2TB,C2,C1TB,C1
  JITFuncSpec_Table_FiniteElement_SpaceInfo_t dg_spaces[NUM_CONTINUOUS_SPACES]; // D2TB,D2,D1TB,D1
  JITFuncSpec_Table_FiniteElement_SpaceInfo_t info_DL,info_D0,info_ED0;

  unsigned num_present_continuous_spaces; // Only the ones that are actually present, i.e. num_continuous_spaces<=4
  JITFuncSpec_Table_FiniteElement_SpaceInfo_t *present_continuous_spaces[NUM_CONTINUOUS_SPACES]; // points to the infos C2TB,C2,C1TB,C1 for looping. Note that not all are filled, only if they are present
  unsigned num_present_dg_spaces; // Only the ones that are actually present, i.e. num_dg_spaces<=4
  JITFuncSpec_Table_FiniteElement_SpaceInfo_t *present_dg_spaces[NUM_CONTINUOUS_SPACES]; // points to the infos D2TB,D2,D1TB,D1 for looping. Note that not all are filled, only if they are present
  

  //Exponents for the D0 fields upon refinement. 
  // If zero [default]: 
  // 		[Coarse D0 value]=(sum of [son D0 value])/nsons
  // 		[Refined son D0 value]=[father D0 value]
  // else:
  // 		[Coarse D0 value] = (sum of [son D0 value])*(1/nsons)**(1-discontinuous_refinement_exponent)  
  // 		[Refined son D0 value] = [father D0 value]*(1/nsons)**(discontinuous_refinement_exponent)  
  // For e.g. a D0 field storing the element size, it will have to be 1
  double *discontinuous_refinement_exponents;

  double *temporal_error_scales;
  bool has_temporal_estimators;

  unsigned num_res_jacs;
  int current_res_jac;
  char **res_jac_names;

  JITFuncSpec_RequiredShapes_FiniteElement_t *shapes_required_ResJac;
  JITFuncSpec_RequiredShapes_FiniteElement_t *shapes_required_Hessian;
  JITFuncSpec_RequiredShapes_FiniteElement_t merged_required_shapes;
  // The same OR restricted to the contributions that are actually ASSEMBLED (residual/Jacobian/mass
  // and Hessian). merged_required_shapes additionally covers integral, local, extremum, Z2-flux and
  // tracer-advection expressions, which are evaluated on their own and never build a Jacobian - using
  // it to decide what to attach as external data let one output observable widen the dense elemental
  // block of every element of the domain. Attachment and the equation remapping that addresses the
  // attached dofs must always read the SAME one of the two, or the remap hands out local equation
  // numbers for data the element does not carry. Buffer sizing and the evaluators keep the full merge.
  JITFuncSpec_RequiredShapes_FiniteElement_t assembly_required_shapes;
  unsigned numglobal_params;
  unsigned *global_paramindices;
  double **global_parameters;

  JITFuncSpec_ResidualAndJacobian_FiniteElement **ParameterDerivative;

  //  unsigned numextdata;
  //  char **extdata_names;

  unsigned numintegral_expressions;
  char **integral_expressions_names;

  unsigned numlocal_expressions;
  char **local_expressions_names;

  unsigned numextremum_expressions;
  char **extremum_expressions_names;

  unsigned numtracer_advections;
  char **tracer_advection_names;

  char *dominant_space; // Use this for e.g. second order position space, but with first order field dofs

  // unsigned num_nullified_bulk_residuals;
  // char **nullified_bulk_residuals;

  int max_dt_order;
  bool fd_jacobian;
  bool fd_position_jacobian;
  double debug_jacobian_epsilon;
  bool with_adaptivity;
  bool stop_on_jacobian_difference;

  int integration_order;
  bool moving_nodes;
  bool use_shared_shape_buffer_during_multi_assemble,during_shared_multi_assembling;

  void *handle; // Handle to the SO
  JITFuncSpec_ResidualAndJacobian_FiniteElement *ResidualAndJacobian;
  JITFuncSpec_ResidualAndJacobian_FiniteElement *ResidualAndJacobianSteady;
  /* Twins of the two above for elements in which nothing hangs, selected per element by
     BulkElementBase::fill_in_generic_residual_contribution_jit. Never NULL where the corresponding
     hanging slot is non-NULL: where no specialised body was emitted they simply point at it. */
  JITFuncSpec_ResidualAndJacobian_FiniteElement *ResidualAndJacobian_NoHang;
  JITFuncSpec_ResidualAndJacobian_FiniteElement *ResidualAndJacobianSteady_NoHang;
  bool * missing_residual_assembly; // Some residuals are not calculated (if not needed, e.g. for azimuthal eigenproblem). We cannot FD then!

  JITFuncSpec_HessianVectorProduct_FiniteElement *HessianVectorProduct;
  bool hessian_generated;

  unsigned num_Z2_flux_terms,num_Z2_flux_terms_for_eigen;
  JITFuncSpec_GetZ2Fluxes_FiniteElement GetZ2Fluxes,GetZ2FluxesForEigen;
  JITFuncSpec_RequiredShapes_FiniteElement_t shapes_required_Z2Fluxes;

  /* Compound-flux grouping for the Z2 error estimator. oomph-lib normalises each group by its own
     recovered-flux norm and combines the groups by taking the maximum, so two error criteria added
     independently to one domain coexist without either diluting the other.
     All four pointers stay NULL and num_Z2_compound_fluxes stays 0 for the historical case of a
     single, fully relative, unweighted group; the estimator checks that to keep its old code path
     (and hence bit-identical errors) for everything that does not ask for grouping.
     Z2_flux_group_index has num_Z2_flux_terms entries, the other two num_Z2_compound_fluxes. */
  unsigned num_Z2_compound_fluxes,num_Z2_compound_fluxes_for_eigen;
  unsigned *Z2_flux_group_index,*Z2_flux_group_index_for_eigen;
  double *Z2_group_normalize_relative,*Z2_group_normalize_relative_for_eigen;
  double *Z2_group_weight,*Z2_group_weight_for_eigen;

  JITFuncSpec_InitialCondition_FiniteElement *InitialConditionFunc;
  unsigned num_ICs;
  char **IC_names;
  JITFuncSpec_DirichletCondition_FiniteElement DirichletConditionFunc;
  bool *Dirichlet_set;
  unsigned Dirichlet_set_size;
  char **Dirichlet_names;
  // Python callbacks. First arg is the functable ptr, func id, then list of doubles, finally num of args
  double (*invoke_callback)(void *, int, double *, int);
  void (*invoke_multi_ret)(void *, int,int, double *,double *,double *, int, int);   //Index, flag,args,returns,derivative matrix, nargs,nret

  JITFuncSpec_EvalIntegralExpr_FiniteElement EvalIntegralExpression;
  JITFuncSpec_RequiredShapes_FiniteElement_t shapes_required_IntegralExprs; // TODO: Split this into the individual contribs?
  JITFuncSpec_EvalIntegralExpr_FiniteElement EvalLocalExpression;
  JITFuncSpec_EvalIntegralExpr_FiniteElement EvalExtremumExpression;
  JITFuncSpec_RequiredShapes_FiniteElement_t shapes_required_LocalExprs; // TODO: Split this into the individual contribs?
  JITFuncSpec_RequiredShapes_FiniteElement_t shapes_required_ExtremumExprs; // TODO: Split this into the individual contribs?
  JITFuncSpec_EvalTracerAdvection_FiniteElement EvalTracerAdvection;
  JITFuncSpec_RequiredShapes_FiniteElement_t shapes_required_TracerAdvection; // TODO: Split this into the individual contribs?

  // Which residuals are actually contributed to by this code: 
  // Bool from residual index, field
  unsigned contribution_entries_size;
  char **contribution_names;
  bool ARRAY_DECL_RESIDUAL_DESTINATION(ARRAY_DECL_NFIELDS(contributes_to_residual));
  bool ARRAY_DECL_RESIDUAL_DESTINATION(ARRAY_DECL_NFIELDS(ARRAY_DECL_NFIELDS(contributes_to_jacobian)));
  // Same shape as contributes_to_jacobian, but restricted to the mass-matrix half of each Jacobian
  // contribution (the part carrying a time derivative of the column field). Always a subset of
  // contributes_to_jacobian. Decided symbolically at code generation time, so it is a superset of
  // whatever entries turn out to be numerically nonzero -- i.e. a valid sparsity pattern. Used to give
  // the mass matrix its own tight, value-independent pattern; see dev_docs/structural_assembly.md.
  bool ARRAY_DECL_RESIDUAL_DESTINATION(ARRAY_DECL_NFIELDS(ARRAY_DECL_NFIELDS(contributes_to_mass_matrix)));
  // Same again for the SECOND derivative: whether d2(residual)/d(field_i)d(field_j) is not
  // identically zero. A Hessian contracted with a vector lives on THIS pattern, which is typically far
  // tighter than the Jacobian's -- every linear term of the residual drops out of it.
  bool ***contributes_to_hessian;
  // Proven properties of the individual elemental blocks, indexed exactly like contributes_to_jacobian
  // ([residual index][row class][column class]) with the JACOBIAN_BLOCK_* bits above. Consumers MUST
  // check contributes_to_jacobian/_mass_matrix first: all bits are 0 for a non-contributing block, not
  // because nothing was proven, but because the block is identically zero (and hence trivially
  // symmetric and constant) so there was nothing to record.
  // For the mass half CONSTANT and CONSTANT_FIXED_DT always coincide: the time-stepper weights sit in
  // the Jacobian half only, so a constant mass block is constant outright. Both bits are set there.
  unsigned char ***jacobian_block_flags;
  unsigned char ***mass_matrix_block_flags;
  // Fields defined on this domain (i.e. without taking over from parent)
  unsigned num_defined_fields_on_this_domain;
  char **defined_field_names_on_this_domain;
  int * dirichlet_field_index_to_global_field_index; // The Dirichlet index is usually different from the contribution index, this maps the Dirichlet to the global field index (problem level). It is filled by the problem, but allocated in the functable. Used for automatic pinning of non-contributing fields (fields without equations on the current residual)
  

  unsigned numcallbacks;
  JITFuncSpec_Callback_Entry_t *callback_infos;
  unsigned num_multi_rets;  
  JITFuncSpec_MultiRet_Entry_t *multi_ret_infos;  

  JITFuncSpec_GeometricJacobian GeometricJacobian;
  JITFuncSpec_GeometricJacobian JacobianForElementSize;
  JITFuncSpec_GeometricJacobianSpatialDerivative JacobianForElementSizeSpatialDerivative;
  JITFuncSpec_GeometricJacobianSpatialDerivative JacobianForElementSizeSecondSpatialDerivative;              

  char * domain_name;

  bool * has_constant_mass_matrix_for_sure;

  // Quick resolving interface dofs by index. Filled for interface codes in the core
  /*
  //TODO: This should be implemented at some point, however, these are created more or less dynamically, so it is not trivial. Potentially, it works when first time creating an interface element
  unsigned int * interface_dof_ids_C1;
  unsigned int * interface_dof_ids_CTB1;
  unsigned int * interface_dof_ids_C2;
  unsigned int * interface_dof_ids_C2TB;
  */

  // Exported functions
  void (*check_compiler_size)(unsigned long long,unsigned long long,char *);  
  double (*get_element_size)(void *);
  void (*fill_shape_buffer_for_point)(unsigned,JITFuncSpec_RequiredShapes_FiniteElement_t *,int);
  void (*clean_up)(struct JITFuncSpec_Table_FiniteElement *functable);
} JITFuncSpec_Table_FiniteElement_t;

typedef void (*JIT_ELEMENT_init_SPEC)(JITFuncSpec_Table_FiniteElement_t *functable);

#ifdef JIT_ELEMENT_SHARED_LIB

static double step(double x)
{
  if (x < 0)
    return 0;
  else if (x > 0)
    return 1.0;
  return 0.5;
}

static double signum(double x)
{
  if (x < 0)
    return -1.0;
  else if (x > 0)
    return 1.0;
  return x; // Nan, Inf progression
}

////////////

// #define Pi M_PI

#define pyoomph_tested_free(x) if (x) free(x);

#define PRINT_RESIDUAL_VECTOR()                          \
  {                                                      \
    printf("ResVec [%d]: ", eleminfo->ndof);             \
    for (unsigned int _i = 0; _i < eleminfo->ndof; _i++) \
      printf("%f\t", residuals[_i]);                     \
    printf("\n");                                        \
  }
#define PRINT_JACOBIAN()                                                       \
  {                                                                            \
    printf("JACOBIAN [%d   %d]:\n", eleminfo->ndof, shapeinfo->jacobian_size); \
    for (unsigned int _i = 0; _i < shapeinfo->jacobian_size; _i++)             \
    {                                                                          \
      for (unsigned int _j = 0; _j < shapeinfo->jacobian_size; _j++)           \
        printf("%f\t", jacobian[_i * shapeinfo->jacobian_size + _j]);          \
      printf("\n");                                                            \
    }                                                                          \
  }

#define SET_INTERNAL_FIELD_NAME(tab, index, name)                   \
  {                                                                 \
    tab[index] = (char *)malloc(sizeof(char) * (strlen(name) + 1)); \
    strncpy(tab[index], name, strlen(name));                        \
    tab[index][strlen(name)] = '\0';                                \
  }
#define SET_INTERNAL_NAME(var, name)                         \
  {                                                          \
    var = (char *)malloc(sizeof(char) * (strlen(name) + 1)); \
    strncpy(var, name, strlen(name));                        \
    var[strlen(name)] = '\0';                                \
  }

// Residual/Jacobian/Hessian assembly macros used by the generated element code (formerly
// in the separate jitbridge_hang.h, always included right after this file). BEGIN_RESIDUAL/
// BEGIN_JACOBIAN_NOHANG/BEGIN_HESSIAN_* are the plain (no hanging nodes involved) variants;
// the *_CONTINUOUS_SPACE/_HANG variants below additionally loop over the master nodes of a
// hanging node, distributing the contribution via the hanging weights.

#ifndef PYOOMPH_TCC_TO_MEMORY
#include <assert.h>
#else
#define assert(expr) ((void)0)
#endif

// No hanging spaces or without considering hang infos
#define BEGIN_RESIDUAL(EQN, CONTRIB)                                                                              \
  local_eqn = EQN;                                                                                                \
  if (local_eqn >= 0) /*&& (!eleminfo->nullified_residual_dof || !eleminfo->nullified_residual_dof[local_eqn] )*/ \
  {                                                                                                               \
    _res_contrib = CONTRIB;

#define ADD_TO_RESIDUAL()             \
  assert(local_eqn < eleminfo->ndof); \
  residuals[local_eqn] += _res_contrib;
#define END_RESIDUAL() }

// The Residual/Jacobian/Mass function is emitted ONCE, as an implementation taking a compile-time
// constant flag, behind a three-line entry point that calls it with 0, 1 and 2. Forcing the inline is
// the whole point: the compiler then folds `if (flag)` and `if (flag == 2)` away, so a residual-only
// assembly runs a body that contains no Jacobian code at all - it does not merely skip it. That matters
// because the register allocator works on the whole function, and a large Jacobian block was measured
// to slow the residual-only path it never enters (dev_docs/code_generation.md 9.4.6).
//
// A compiler without the attribute simply gets one body with a runtime flag, exactly as before:
// correct, just not specialised.
//
// Per compiler, from the vendor documentation rather than from testing (we build with GCC here):
//
//  * MSVC has no __attribute__ syntax; the equivalent is the __forceinline KEYWORD, which "overrides
//    the cost-benefit analysis". It is honoured under the /O2 that ccompiler.py passes. It cannot
//    inline under /Ob0 (the debug default) and issues no diagnostic in that case, so a debug build
//    degrades silently to the runtime-flag form - which is exactly the intended fallback.
//    clang-cl defines _MSC_VER and accepts __forceinline, so it takes this branch too; that is why
//    _MSC_VER is tested FIRST, since clang-cl may define both.
//  * Clang accepts the GNU spelling and defines __GNUC__, so it takes the GCC branch. Its
//    always_inline "disables inlining heuristics and inlining is always attempted regardless of
//    optimization level", but like every compiler here it "does not guarantee that inline
//    substitution actually occurs".
//  * GCC diagnoses a FAILURE to inline an always_inline function as a hard ERROR, not a warning. On a
//    JIT path that turns a missed optimisation into a failed compile, so it is worth knowing that the
//    documented causes - target-option mismatch across translation units, definition not available,
//    varargs, recursion - none of them apply here: one translation unit, one -march, and the
//    implementation is defined immediately above its only three callers.
//
// The __OPTIMIZE__ term is about the debug build: with PYOOMPH_DEBUG=1 ccompiler.py compiles at -O0,
// where specialising buys nothing (nobody measures a debug build) and a single out-of-line body is far
// easier to step through than three inlined copies of it. Measured on the 3D solid element:
//
//   -O0, guarded (this branch off): dispatcher 121 B + one _impl of 210 775 B, .so 403 672 B, 0.51 s
//   -O0, always_inline forced     : one symbol of 221 877 B,                    .so 420 008 B, 0.59 s
//   -O3, always_inline            : one symbol of  71 523 B,                    .so  94 128 B
//
// Note what those numbers do NOT say: forcing it at -O0 does not triple the code. GCC folds the
// constant `flag` branches while inlining, even with no optimiser running, so the three copies collapse
// to +5%. The guard is worth having for debuggability and the 16% of compile time, not because the
// alternative is catastrophic - an earlier version of this comment claimed it was.
#if defined(_MSC_VER)
#define PYOOMPH_RJM_IMPL static __forceinline void
#elif defined(__GNUC__) && !defined(__TINYC__) && defined(__OPTIMIZE__)
#define PYOOMPH_RJM_IMPL static inline __attribute__((always_inline)) void
#else
#define PYOOMPH_RJM_IMPL static void
#endif

#define BEGIN_JACOBIAN() \
  if (flag)              \
  {

#define ADD_TO_JACOBIAN_NOHANG_NOHANG() jacobian[local_eqn * shapeinfo->jacobian_size + local_unknown] += _J_contrib;

#define ADD_TO_MASS_MATRIX_NOHANG_NOHANG(MPART)                                    \
  if (flag == 2)                                                                   \
  {                                                                                \
    mass_matrix[local_eqn * shapeinfo->mass_matrix_size + local_unknown] += MPART; \
  }

#define END_JACOBIAN() }

#define BEGIN_JACOBIAN_NOHANG(EQN, CONTRIB) \
  local_unknown = EQN;                      \
  if (local_unknown >= 0)                   \
  {                                         \
    double _J_contrib = CONTRIB;

#define END_JACOBIAN_NOHANG() }

#define BEGIN_HESSIAN_TEST_LOOP(EQN)                                                                               \
  local_eqn = EQN;                                                                                                 \
  if (local_eqn >= 0) /*&& (!eleminfo->nullified_residual_dof || !eleminfo->nullified_residual_dof[local_eqn] ) */ \
  {

#define END_HESSIAN_TEST_LOOP() }

#define BEGIN_HESSIAN_SHAPE_LOOP1(EQN) \
  local_unknown = EQN;                 \
  if (local_unknown >= 0)              \
  {

#define BEGIN_HESSIAN_SHAPE_LOOP2(EQN, CONTRIB) \
  local_deriv = EQN;                            \
  if (local_deriv >= 0)                         \
  {                                             \
    _H_contrib = CONTRIB;

// Hanging macros.
//
// The leading HANGON argument is what lets ONE emitted body serve both the hanging and the
// non-hanging element populations (dev_docs/code_generation.md 9.4.14). It is either the literal 1
// or the constant `_impl` parameter pyoomph_hang_on, so after inlining the compiler sees a compile-
// time constant:
//
//  * HANGON == 0: `&&` short-circuits, so HANGINFO is never even loaded, `nummaster` folds to 0, the
//    guard collapses to `(EQN) >= 0`, the master loop to a single iteration and hang_weight to 1.0 -
//    i.e. exactly the shape of BEGIN_JACOBIAN_NOHANG, without emitting the body a second time.
//  * HANGON == 1: the arithmetic of the hanging path, unchanged.
//
// Testing `nummaster || (EQN) >= 0` BEFORE evaluating CONTRIB is the second half of the point: a
// pinned, non-hanging dof used to compute its contribution and then throw it away. Only a real hang
// needs the contribution before the row is known (the same value is scattered to several masters),
// and that case still gets it. CONTRIB is pure arithmetic plus pure callbacks, so not evaluating it
// is unobservable - pyoomph does not unmask FP exceptions.
#define BEGIN_RESIDUAL_CONTINUOUS_SPACE(HANGON, EQN, CONTRIB, HANGINFO, LINDEX)                                     \
  nummaster = ((HANGON) && HANGINFO[LINDEX].nummaster) ? HANGINFO[LINDEX].nummaster : 0u;                           \
  if (nummaster || (EQN) >= 0)                                                                                      \
  {                                                                                                                 \
    const unsigned _nmaster = (nummaster ? nummaster : 1u);                                                         \
    _res_contrib = CONTRIB;                                                                                         \
    for (unsigned m = 0; m < _nmaster; m++)                                                                         \
    {                                                                                                               \
      if (nummaster)                                                                                                \
      {                                                                                                             \
        local_eqn = HANGINFO[LINDEX].masters[m].local_eqn;                                                          \
        hang_weight = HANGINFO[LINDEX].masters[m].weight;                                                           \
      }                                                                                                             \
      else                                                                                                          \
      {                                                                                                             \
        local_eqn = EQN;                                                                                            \
        hang_weight = 1.0;                                                                                          \
      }                                                                                                             \
      if (local_eqn >= 0) /*&& (!eleminfo->nullified_residual_dof || !eleminfo->nullified_residual_dof[local_eqn] )*/ \
      {

#define ADD_TO_RESIDUAL_CONTINUOUS_SPACE() \
  assert(local_eqn < eleminfo->ndof);      \
  residuals[local_eqn] += hang_weight * _res_contrib;

#define END_RESIDUAL_CONTINUOUS_SPACE() \
  }                                     \
  }                                     \
  }

// Note the asymmetry to BEGIN_JACOBIAN_NOHANG, which is deliberate and predates this: that macro
// DECLARES its own `double _J_contrib`, shadowing the one declared per integration block, while this
// one ASSIGNS the outer variable.
#define BEGIN_JACOBIAN_HANG(HANGON, EQN, CONTRIB, HANGINFO, LINDEX)       \
  nummaster2 = ((HANGON) && HANGINFO[LINDEX].nummaster) ? HANGINFO[LINDEX].nummaster : 0u; \
  if (nummaster2 || (EQN) >= 0)                                           \
  {                                                                       \
    const unsigned _nmaster2 = (nummaster2 ? nummaster2 : 1u);            \
    _J_contrib = CONTRIB;                                                 \
    for (unsigned m2 = 0; m2 < _nmaster2; m2++)                           \
    {                                                                     \
      if (nummaster2)                                                     \
      {                                                                   \
        local_unknown = HANGINFO[LINDEX].masters[m2].local_eqn;           \
        hang_weight2 = HANGINFO[LINDEX].masters[m2].weight;               \
      }                                                                   \
      else                                                                \
      {                                                                   \
        local_unknown = EQN;                                              \
        hang_weight2 = 1.0;                                               \
      }                                                                   \
      if (local_unknown >= 0)                                             \
      {

#define ADD_TO_JACOBIAN_HANG_NOHANG() jacobian[local_eqn * shapeinfo->jacobian_size + local_unknown] += hang_weight * _J_contrib;
#define ADD_TO_JACOBIAN_NOHANG_HANG() jacobian[local_eqn * shapeinfo->jacobian_size + local_unknown] += hang_weight2 * _J_contrib;
#define ADD_TO_JACOBIAN_HANG_HANG() jacobian[local_eqn * shapeinfo->jacobian_size + local_unknown] += hang_weight * hang_weight2 * _J_contrib;

#define ADD_TO_MASS_MATRIX_HANG_NOHANG(MPART)                                                      \
  if (flag == 2)                                                                                   \
  {                                                                                                \
    mass_matrix[local_eqn * shapeinfo->mass_matrix_size + local_unknown] += hang_weight * (MPART); \
  }
#define ADD_TO_MASS_MATRIX_NOHANG_HANG(MPART)                                                       \
  if (flag == 2)                                                                                    \
  {                                                                                                 \
    mass_matrix[local_eqn * shapeinfo->mass_matrix_size + local_unknown] += hang_weight2 * (MPART); \
  }
#define ADD_TO_MASS_MATRIX_HANG_HANG(MPART)                                                                       \
  if (flag == 2)                                                                                                  \
  {                                                                                                               \
    mass_matrix[local_eqn * shapeinfo->mass_matrix_size + local_unknown] += hang_weight * hang_weight2 * (MPART); \
  }

#define END_JACOBIAN_HANG() \
  }                         \
  }                         \
  }

// Hanging macros (Hessian)
#define BEGIN_HESSIAN_TEST_LOOP_CONTINUOUS_SPACE(EQN, HANGINFO, LINDEX)                                                \
  if (HANGINFO[LINDEX].nummaster)                                                                                     \
  {                                                                                                                   \
    nummaster = HANGINFO[LINDEX].nummaster;                                                                           \
  }                                                                                                                   \
  else                                                                                                                \
  {                                                                                                                   \
    nummaster = 1;                                                                                                    \
  }                                                                                                                   \
  for (int m = 0; m < nummaster; m++)                                                                                 \
  {                                                                                                                   \
    if (HANGINFO[LINDEX].nummaster)                                                                                   \
    {                                                                                                                 \
      local_eqn = HANGINFO[LINDEX].masters[m].local_eqn;                                                              \
      hang_weight = HANGINFO[LINDEX].masters[m].weight;                                                               \
    }                                                                                                                 \
    else                                                                                                              \
    {                                                                                                                 \
      local_eqn = EQN;                                                                                                \
      hang_weight = 1.0;                                                                                              \
    }                                                                                                                 \
    if (local_eqn >= 0) /* && (!eleminfo->nullified_residual_dof || !eleminfo->nullified_residual_dof[local_eqn] ) */ \
    {

#define END_HESSIAN_TEST_LOOP_CONTINUOUS_SPACE() \
  }                                              \
  }

#define BEGIN_HESSIAN_SHAPE_LOOP1_CONTINUOUS_SPACE(EQN, HANGINFO, LINDEX)           \
  if (HANGINFO[LINDEX].nummaster)                                                   \
  {                                                                                 \
    nummaster2 = HANGINFO[LINDEX].nummaster;                                        \
  }                                                                                 \
  else                                                                              \
  {                                                                                 \
    nummaster2 = 1;                                                                 \
  }                                                                                 \
  for (int m2 = 0; m2 < nummaster2; m2++)                                           \
  {                                                                                 \
    if (HANGINFO[LINDEX].nummaster)                                                 \
    {                                                                               \
      local_unknown = HANGINFO[LINDEX].masters[m2].local_eqn;                       \
      hang_weight2 = HANGINFO[LINDEX].masters[m2].weight;                           \
    }                                                                               \
    else                                                                            \
    {                                                                               \
      local_unknown = EQN;                                                          \
      hang_weight2 = 1.0;                                                           \
    }                                                                               \
    if (local_unknown >= 0)                                                         \
    {

#define END_HESSIAN_SHAPE_LOOP1_CONTINUOUS_SPACE() \
  }                                                \
  }

#define BEGIN_HESSIAN_SHAPE_LOOP2_CONTINUOUS_SPACE(EQN, CONTRIB, HANGINFO, LINDEX)           \
  if (HANGINFO[LINDEX].nummaster)                                                            \
  {                                                                                          \
    nummaster3 = HANGINFO[LINDEX].nummaster;                                                 \
  }                                                                                          \
  else                                                                                       \
  {                                                                                          \
    nummaster3 = 1;                                                                          \
  }                                                                                          \
  _H_contrib = CONTRIB;                                                                      \
  for (int m3 = 0; m3 < nummaster3; m3++)                                                    \
  {                                                                                          \
    if (HANGINFO[LINDEX].nummaster)                                                          \
    {                                                                                        \
      local_deriv = HANGINFO[LINDEX].masters[m3].local_eqn;                                  \
      hang_weight3 = HANGINFO[LINDEX].masters[m3].weight;                                    \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
      local_deriv = EQN;                                                                     \
      hang_weight3 = 1.0;                                                                    \
    }                                                                                        \
    if (local_deriv >= 0)                                                                    \
    {

#define END_HESSIAN_SHAPE_LOOP2_CONTINUOUS_SPACE() \
  }                                                \
  }

#define END_HESSIAN_SHAPE_LOOP1() }
#define END_HESSIAN_SHAPE_LOOP2() }


// HESSIAN ASSEMBLY USING H_{ijk}=H_{ikj}
#ifdef ASSEMBLE_HESSIAN_VIA_SYMMETRY

#define ADD_TO_HESSIAN_FACTOR(FACTOR) \
   const double _H_symm_contrib=(FACTOR) * (_H_contrib); \
   hessian_buffer[local_eqn*n_dof*n_dof+local_unknown*n_dof+local_deriv] +=_H_symm_contrib;\
   if (!symmetry_assembly_same_field) hessian_buffer[local_eqn*n_dof*n_dof+local_deriv*n_dof+local_unknown] +=_H_symm_contrib;\

#define ADD_TO_HESSIAN_NOHANG_NOHANG_NOHANG()  \
   const double _H_symm_contrib=_H_contrib; \
   hessian_buffer[local_eqn*n_dof*n_dof+local_unknown*n_dof+local_deriv] += _H_symm_contrib;\
   if (!symmetry_assembly_same_field) hessian_buffer[local_eqn*n_dof*n_dof+local_deriv*n_dof+local_unknown] +=_H_symm_contrib;\

// Mass matrix not symmetric!
#define __ADD_TO_MASS_HESSIAN_FACTOR(FACTOR, MCONTRIB)                               \
  if (flag >= 2)                                                                   \
  { \
    const double _M_symm_contrib=(FACTOR)*(MCONTRIB);                                        \
    hessian_M_buffer[local_eqn*n_dof*n_dof+local_unknown*n_dof+local_deriv] +=_M_symm_contrib;\
    if (!symmetry_assembly_same_field) hessian_M_buffer[local_eqn*n_dof*n_dof+local_deriv*n_dof+local_unknown] +=_M_symm_contrib;\
  }
#define __ADD_TO_MASS_HESSIAN_NOHANG_NOHANG_NOHANG(MCONTRIB)                \
  if (flag >= 2)                                                          \
  {                                                                       \
    const double _M_symm_contrib=(MCONTRIB);\
    hessian_M_buffer[local_eqn*n_dof*n_dof+local_unknown*n_dof+local_deriv] += _M_symm_contrib;\
    if (!symmetry_assembly_same_field) hessian_M_buffer[local_eqn*n_dof*n_dof+local_deriv*n_dof+local_unknown] +=_M_symm_contrib;\
  }


// HESSIAN ASSEMBLY __NOT__ USING H_{ijk}=H_{ikj}
#else

#define ADD_TO_HESSIAN_FACTOR(FACTOR) \
   hessian_buffer[local_eqn*n_dof*n_dof+local_unknown*n_dof+local_deriv] +=(FACTOR) * _H_contrib;

#define ADD_TO_HESSIAN_NOHANG_NOHANG_NOHANG()  \
   hessian_buffer[local_eqn*n_dof*n_dof+local_unknown*n_dof+local_deriv] += _H_contrib;



// End of Hessian assembly information
#endif


#define ADD_TO_MASS_HESSIAN_FACTOR(FACTOR, MCONTRIB)                               \
  if (flag>=2) \
  {\
    hessian_M_buffer[local_eqn*n_dof*n_dof+local_unknown*n_dof+local_deriv] += (FACTOR) *  (MCONTRIB); \
  }

#define ADD_TO_MASS_HESSIAN_NOHANG_NOHANG_NOHANG(MCONTRIB)                \
  if (flag >= 2) \
  { \
   hessian_M_buffer[local_eqn*n_dof*n_dof+local_unknown*n_dof+local_deriv] += (MCONTRIB); \
  }


#define ADD_TO_HESSIAN_HANG_NOHANG_NOHANG() ADD_TO_HESSIAN_FACTOR(hang_weight)
#define ADD_TO_HESSIAN_NOHANG_HANG_NOHANG() ADD_TO_HESSIAN_FACTOR(hang_weight2)
#define ADD_TO_HESSIAN_NOHANG_NOHANG_HANG() ADD_TO_HESSIAN_FACTOR(hang_weight3)

#define ADD_TO_HESSIAN_HANG_HANG_NOHANG() ADD_TO_HESSIAN_FACTOR(hang_weight *hang_weight2)
#define ADD_TO_HESSIAN_NOHANG_HANG_HANG() ADD_TO_HESSIAN_FACTOR(hang_weight2 *hang_weight3)
#define ADD_TO_HESSIAN_HANG_NOHANG_HANG() ADD_TO_HESSIAN_FACTOR(hang_weight *hang_weight3)

#define ADD_TO_HESSIAN_HANG_HANG_HANG() ADD_TO_HESSIAN_FACTOR(hang_weight *hang_weight2 *hang_weight3)


#define ADD_TO_MASS_HESSIAN_HANG_NOHANG_NOHANG(MCONTRIB) ADD_TO_MASS_HESSIAN_FACTOR(hang_weight, MCONTRIB)
#define ADD_TO_MASS_HESSIAN_NOHANG_HANG_NOHANG(MCONTRIB) ADD_TO_MASS_HESSIAN_FACTOR(hang_weight2, MCONTRIB)
#define ADD_TO_MASS_HESSIAN_NOHANG_NOHANG_HANG(MCONTRIB) ADD_TO_MASS_HESSIAN_FACTOR(hang_weight3, MCONTRIB)

#define ADD_TO_MASS_HESSIAN_HANG_HANG_NOHANG(MCONTRIB) ADD_TO_MASS_HESSIAN_FACTOR(hang_weight *hang_weight2, MCONTRIB)
#define ADD_TO_MASS_HESSIAN_NOHANG_HANG_HANG(MCONTRIB) ADD_TO_MASS_HESSIAN_FACTOR(hang_weight2 *hang_weight3, MCONTRIB)
#define ADD_TO_MASS_HESSIAN_HANG_NOHANG_HANG(MCONTRIB) ADD_TO_MASS_HESSIAN_FACTOR(hang_weight *hang_weight3, MCONTRIB)

#define ADD_TO_MASS_HESSIAN_HANG_HANG_HANG(MCONTRIB) ADD_TO_MASS_HESSIAN_FACTOR(hang_weight *hang_weight2 *hang_weight3, MCONTRIB)

#define Pi 3.14159265359

#endif

#ifndef PYOOMPH_TCC_TO_MEMORY
#if defined __ELF__
#define JIT_API_EXPORT __attribute((visibility("default")))
#elif defined __APPLE__
#define JIT_API_EXPORT __attribute((visibility("default")))
#elif defined EXPORT_API_FOR_JIT
#define JIT_API_EXPORT __declspec(dllexport)
#else
#define JIT_API_EXPORT __declspec(dllimport)
#endif
#else
#define JIT_API_EXPORT
#endif

#ifdef __cplusplus
#include <string>
int LoadJITFiniteElementCode(std::string);
#endif

#ifndef PYOOMPH_TCC_TO_MEMORY

#if defined __TINYC__
#define JIT_API __attribute__((dllexport))
#define PYOOMPH_AQUIRE_ARRAY(typ, varname, size) typ varname[size];
#define PYOOMPH_AQUIRE_TWO_D_ARRAY(typ, varname, size1,size2) typ varname[size1][size2];
#elif defined __ELF__
// #define API __attribute((visibility("default")))
#define PYOOMPH_AQUIRE_ARRAY(typ, varname, size) typ varname[size];
#define PYOOMPH_AQUIRE_TWO_D_ARRAY(typ, varname, size1,size2) typ varname[size1][size2];
#define JIT_API
#elif defined __WIN32__
#define JIT_API __declspec(dllexport)
#define PYOOMPH_AQUIRE_ARRAY(typ, varname, size) typ *varname = (typ *)_alloca(size * sizeof(typ));
#define PYOOMPH_AQUIRE_TWO_D_ARRAY(typ, varname, size1,size2) typ **varname = (typ **)_alloca(size1 * sizeof(typ*)); { for (int _i=0;_i<size1;_i++) varname[_i]=(typ *)_alloca(size2 * sizeof(typ)); }
#else
#define JIT_API
#define PYOOMPH_AQUIRE_ARRAY(typ, varname, size) typ varname[size];
#define PYOOMPH_AQUIRE_TWO_D_ARRAY(typ, varname, size1,size2) typ varname[size1][size2];
#endif
#endif

#define JIT_GDB_BREAKPOINT __asm__("int $3");

#endif

#if defined _MSC_VER
#undef PYOOMPH_AQUIRE_ARRAY
#define PYOOMPH_AQUIRE_ARRAY(typ, varname, size) typ *varname = (typ *)_alloca(size * sizeof(typ));
#define PYOOMPH_AQUIRE_TWO_D_ARRAY(typ, varname, size1,size2) typ **varname = (typ **)_alloca(size1 * sizeof(typ*)); { for (int _i=0;_i<size1;_i++) varname[_i]=(typ *)_alloca(size2 * sizeof(typ)); }
#ifndef JIT_API
#undef JIT_API
#endif
#define JIT_API __declspec(dllexport)
#endif

#ifndef JIT_API
#define JIT_API
#endif

#ifndef PYOOMPH_AQUIRE_ARRAY
#define PYOOMPH_AQUIRE_ARRAY(typ, varname, size) typ varname[size];
#define PYOOMPH_AQUIRE_TWO_D_ARRAY(typ, varname, size1,size2) typ varname[size1][size2];
#endif

#define ASSEMBLE_HESSIAN_VECTOR_PRODUCTS_FROM(jac_y, Cs, n_dof, n_vec, product) \
  for (unsigned i = 0; i < n_dof; i++)                                          \
  {                                                                             \
    for (unsigned k = 0; k < n_dof; k++)                                        \
    {                                                                           \
      const double j_y = jac_y[i * n_dof + k];                                  \
      for (unsigned v = 0; v < n_vec; v++)                                      \
      {                                                                         \
        product[v * n_dof + i] += j_y * Cs[v * n_dof + k];                      \
      }                                                                         \
    }                                                                           \
  }

#define ASSEMBLE_SYMMETRIC_HESSIAN_VECTOR_PRODUCTS_FROM(Y,Cs,n_dof,n_vec,product) \
for (unsigned int i=0;i<n_dof;i++) \
{\
  for (unsigned int k=0;k<n_dof;k++)\
  {\
      double Yj_Hijk=0.0;\
      for (unsigned int j=0;j<n_dof;j++)\
      {\
        Yj_Hijk+=Y[j]*hessian_buffer[i*n_dof*n_dof+j*n_dof+k];\
      }\
      for (unsigned int v=0;v<n_vec;v++)\
      {\
        product[v*n_dof+i]+=Yj_Hijk*Cs[v*n_dof+k];\
      }\
  }\
}\

#define SET_DIRECTIONAL_HESSIAN_FROM(jac_y, n_dof, product) \
  for (unsigned i = 0; i < n_dof; i++)                             \
  {                                                                \
    for (unsigned k = 0; k < n_dof; k++)                           \
    {                                                              \
      product[i * n_dof + k] = jac_y[i * n_dof + k];               \
    }                                                              \
  }

#define SET_DIRECTIONAL_SYMMETRIC_HESSIAN_FROM(assm_buffer,Y,n_dof,product)\
  for (unsigned int ivec=0;ivec<numvectors;ivec++) \
  { \
   for (unsigned i = 0; i < n_dof; i++)                             \
   {                                                                \
    for (unsigned k = 0; k < n_dof; k++)                           \
    {                                                              \
      for (unsigned int j=0;j<n_dof;j++)\
      {\
        product[n_dof*n_dof*ivec+ i * n_dof + k] += assm_buffer[i*n_dof*n_dof+j*n_dof+k]*Y[n_dof*ivec+j];\
      }\
    }                                                              \
   } \
  }

#define SET_DIRECTIONAL_SYMMETRIC_HESSIAN_FROM_TRANSPOSED(assm_buffer,Y,n_dof,product)\
  for (unsigned int ivec=0;ivec<numvectors;ivec++) \
  { \
   for (unsigned i = 0; i < n_dof; i++)                             \
   {                                                                \
    for (unsigned k = 0; k < n_dof; k++)                           \
    {                                                              \
      for (unsigned int j=0;j<n_dof;j++)\
      {\
        product[n_dof*n_dof*ivec+ i * n_dof + k] += assm_buffer[j*n_dof*n_dof+i*n_dof+k]*Y[n_dof*ivec+j];\
      }\
    }                                                              \
   } \
  }

//Place at the end of a MultiReturnFunction C-code if you are to lazy to implement the derivative.
// epsilon_fd controlls the finite-difference step
#define FILL_MULTI_RET_JACOBIAN_BY_FD(epsilon_fd) \
if (flag)\
{\
  for (unsigned int i=0;i<nargs*nret;i++) derivative_matrix[i]=0.0;\
  PYOOMPH_AQUIRE_ARRAY(double, res_p, nret);\
  for (unsigned int i=0;i<nargs;i++)\
  {\
    const double oldarg=arg_list[i];\
    arg_list[i]+=epsilon_fd;\
    CURRENT_MULTIRET_FUNCTION(0, arg_list, res_p, PYOOMPH_NULL,nargs,nret);\
    for (unsigned int j=0;j<nret;j++)\
    {\
      derivative_matrix[j*nargs+i]=(res_p[j]-result_list[j])/epsilon_fd;\
    }\
    arg_list[i]=oldarg;\
  }\
}\

