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

// This file defines the structures which are required to transfer the oomph-lib data (e.g. shape functions) to the C-compiled code

typedef struct JITElementInfo
{
  unsigned int nnode; // Total number of nodes
  unsigned int nnode_C1;
  unsigned int nnode_C2;
  unsigned int nnode_C1TB; // C1 on quadrilaterals, C1+Bubble on TElements, on interfaces, these are C1
  unsigned int nnode_C2TB; // C2 on quadrilaterals, C2+Bubble on TElements, on interfaces, these are C2  
  unsigned int nnode_DL;   // This are actually not nodes, but internal data (since discontinous)
  // unsigned int nnode_D0;  //This are actually not nodes, but internal data (since discontinous), This is always 1
  unsigned int nodal_dim; // Nodal dimension

  double ***nodal_coords; // Nodal coordinates (node index, xindex, time index)
  double ***nodal_data;   // Nodal data (node index,data index, time index)

  // Two arrays, since you can nullify dofs, i.e. no contribution to residual by this element, but still for the jacobian
  int **nodal_local_eqn; // Nodal equations (node index, data index)
  int **pos_local_eqn;   // Nodal equations (node index, data index)

  // bool * nullified_residual_dof;

  // double **extdata_value;		//External data values (ext data index,  time index)
  // int * ext_local_eqn; //External equations (ext data index, value index)

  // double *** direct_nodal_data; //Nodal data without consideration of hanging nodes (is used as backup direct_nodal_data[nodeindex]=nodal_data[nodindex] for non-hanging
  unsigned int ndof; // Number of local dofs

  bool alloced;
  void *elem_ptr; // Pointer to the element //TODO: This is problematic, as the this pointer cannot be restored for multiple inheritance

  struct JITElementInfo *bulk_eleminfo;
  // struct JITElementInfo * otherbulk_eleminfo;
  struct JITElementInfo *opposite_eleminfo;
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
#define DX_SHAPE_FUNCTION_DECL(what) const double(*what)[MAX_NODAL_DIM]

#else

#define ARRAY_DECL_NDIM(what) *what
#define ARRAY_DECL_UNITY(what) *what
#define ARRAY_DECL_NNODE(what) *what
#define ARRAY_DECL_NDT(what) *what
#define ARRAY_DECL_NHANG(what) *what
#define ARRAY_DECL_NFIELDS(what) *what
#define ARRAY_DECL_RESIDUAL_DESTINATION(what) *what
#define DX_SHAPE_FUNCTION_DECL(what) double * const * const what
#endif


typedef struct JITHangInfoEntry
{
  double weight;
  int ARRAY_DECL_NFIELDS(local_eqn); // Field index
  // double **master_coordinate; //Coordinate (x/y/z, time index)
} JITHangInfoEntry_t;

typedef struct JITHangInfo
{
  int nummaster;
  bool ARRAY_DECL_NFIELDS(has_contributions); // Is any of the hanging equations non-pinned?
  JITHangInfoEntry_t ARRAY_DECL_NHANG(masters); // 0..nummasters-1
} JITHangInfo_t;



typedef struct JITShapeInfo
{
  unsigned int n_int_pt;             // Number of integration points
  double int_pt_weight;            // Eulerian weight at the current integration point
  double int_pt_weight_Lagrangian; // Lagrangian weight at the current integration point
  double int_pt_weight_unity;            // Weight at the current integration point in s space, i.e. without any mapping [ sqrt(det(g_ab)) ]
  double ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(int_pt_weights_d_coords)); // Weights derived by coordinates, [i_dim,l_node], i.e. w*dJ_Eulerian/dX^l_i
  double ****int_pt_weights_d2_coords; // Weights derived by coordinates, [i_dim,j_dim,l_node_i,l_node_j], i.e. w*d2J_Eulerian/(dX^l_i*dX^l_j)
  
  double elemsize_Eulerian,elemsize_Eulerian_cartesian;            // Eulerian element size, with e.g. 2*pi*r in integration or not
  double elemsize_Lagrangian,elemsize_Lagrangian_cartesian; // Lagrangian element size
  double ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(elemsize_d_coords)); // Eulerian element size derived by coordinates, [i_dim,l_node], i.e. sum(w*dJ_Eulerian)/dX^l_i
  double ****elemsize_d2_coords; // Weights derived by coordinates, [i_dim,j_dim,l_node_i,l_node_j], i.e. sum(w*d2J_Eulerian)/(dX^l_i*dX^l_j)      
  // Cartesian variants
  double ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(elemsize_Cart_d_coords)); // Eulerian element size derived by coordinates, [i_dim,l_node], i.e. sum(w*dJ_Eulerian)/dX^l_i
  double ****elemsize_Cart_d2_coords; // Weights derived by coordinates, [i_dim,j_dim,l_node_i,l_node_j], i.e. sum(w*d2J_Eulerian)/(dX^l_i*dX^l_j)  

  double ARRAY_DECL_NNODE(shape_C2);                // C2 shapes (node index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dx_shape_C2));            // C2 shapes ( node index, coord index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dX_shape_C2));            // Corresponding Lagrangian version
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dS_shape_C2));            // Corresponding local coordinate version
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(d_dx_shape_dcoord_C2)))); // derivative of dx_shape_C2 w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index)
  double ******d2_dx2_shape_dcoord_C2; // second derivative of dx_shape_C2 w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index,deriv. coord node index2, deriv coord dir index2)

  double ARRAY_DECL_NNODE(ARRAY_DECL_NNODE(nodal_shape_C2)); // C2 shapes (node index, node index). In principle just delta_{i,j}

  double ARRAY_DECL_NNODE(shape_C2TB);                // C2TB shapes (node index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dx_shape_C2TB));            // C2TB shapes (node index, coord index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dX_shape_C2TB));            // Lagrangian version
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dS_shape_C2TB));            // Corresponding local coordinate version
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(d_dx_shape_dcoord_C2TB)))); // derivative of dx_shape_C2TB w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index)
  double ******d2_dx2_shape_dcoord_C2TB; // second derivative of dx_shape_C2TB w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index,deriv. coord node index2, deriv coord dir index2)

  double ARRAY_DECL_NNODE(ARRAY_DECL_NNODE(nodal_shape_C2TB)); // C2TB shapes (node index, node index). In principle just delta_{i,j}

  double ARRAY_DECL_NNODE(shape_C1);                // C1 shapes (node index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dx_shape_C1));            // C1 shapes (node index, coord index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dX_shape_C1));            // Corresponding Lagrangian version
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dS_shape_C1));            // Corresponding local coordinate version
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(d_dx_shape_dcoord_C1)))); // derivative of dx_shape_C1 w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index)
  double ******d2_dx2_shape_dcoord_C1; // second derivative of dx_shape_C2 w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index,deriv. coord node index2, deriv coord dir index2)
  double ARRAY_DECL_NNODE(ARRAY_DECL_NNODE(nodal_shape_C1)); // C1 shapes (node index, node index). In principle just delta_{i,j}
  
  double ARRAY_DECL_NNODE(shape_C1TB);                // C1TB shapes (node index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dx_shape_C1TB));            // C1TB shapes (node index, coord index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dX_shape_C1TB));            // Corresponding Lagrangian version
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dS_shape_C1TB));            // Corresponding local coordinate version
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(d_dx_shape_dcoord_C1TB)))); // derivative of dx_shape_C1TB w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index)
  double ******d2_dx2_shape_dcoord_C1TB; // second derivative of dx_shape_C2 w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index,deriv. coord node index2, deriv coord dir index2)
  double ARRAY_DECL_NNODE(ARRAY_DECL_NNODE(nodal_shape_C1TB)); // C1TB shapes (node index, node index). In principle just delta_{i,j}  

  double ARRAY_DECL_NNODE(shape_DL);                // DL shapes (node index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dx_shape_DL));            // DL shapes (node index, coord index)
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dX_shape_DL));            // Corresponding Lagrangian derivatives
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(dS_shape_DL));            // Corresponding local coordinate version
  double ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(d_dx_shape_dcoord_DL)))); // derivative of dx_shape_DL w/r to nodal coords (intpt,node index, coord index, deriv. coord node index, deriv coord dir index)
  double ******d2_dx2_shape_dcoord_DL; // second derivative of dx_shape_DL w/r to nodal coords (intpt,node index, coord index, deriv. coord node index, deriv coord dir index,deriv. coord node index2, deriv coord dir index2)
  double ARRAY_DECL_NNODE(ARRAY_DECL_NNODE(nodal_shape_DL)); // DL shapes (node index, node index). In principle just delta_{i,j}
  
  double ARRAY_DECL_NDIM(ARRAY_DECL_NDIM(ARRAY_DECL_NNODE(ARRAY_DECL_NDIM(d_dshape_dx_tensor))));
  
  #ifdef FIXED_SIZE_SHAPE_BUFFER
  double *shape_Pos; // Pos space shapes. These will be mapped to the dominant element space
  double (*dx_shape_Pos)[MAX_NODAL_DIM];
  double (*dX_shape_Pos)[MAX_NODAL_DIM];
  double (*dS_shape_Pos)[MAX_NODAL_DIM];
  double (*d_dx_shape_dcoord_Pos)[MAX_NODAL_DIM][MAX_NODES][MAX_NODAL_DIM];
  #else
  double *shape_Pos; // Pos space shapes. These will be mapped to the dominant element space
  double **dx_shape_Pos;
  double **dX_shape_Pos;
  double **dS_shape_Pos;
  double ****d_dx_shape_dcoord_Pos;  
  #endif
   double ******d2_dx2_shape_dcoord_Pos; // second derivative of dx_shape_DL w/r to nodal coords (node index, coord index, deriv. coord node index, deriv coord dir index,deriv. coord node index2, deriv coord dir index2)

  // double ** shape_D0; //DL shapes (intpt, "node" index) -> Actually always 1 //TODO: Simplify this
  // double *** dx_shape_D0; //DL shapes (intpt, "node" index,coord index) -> Actually always zero //TODO: Simplify this

  unsigned int jacobian_size;
  unsigned int mass_matrix_size;

  double ARRAY_DECL_NDIM(normal);            // direction //TODO: This does not allow for divergence of the normal etc.
  double ***d_normal_dcoord; // Derivative of the normal wrt. nodal coordinates [ipt][dir][coord node][coord dir]
  double *****d2_normal_d2coord; // Second order derivative of the normal wrt. nodal coordinates [ipt][dir][coord node 1][coord dir 1][coord node 2][coord dir 2]

  // double * dx_shape_at_center_C1; //Gradients of C1 space at center //Required for SUPG

  double ARRAY_DECL_NDT(t);
  double ARRAY_DECL_NDT(dt);                           // Current time buffer and desired [and history] time steps
  unsigned int timestepper_ntstorage;       // Number of timestepper weights
  double ARRAY_DECL_NDT(timestepper_weights_dt_BDF1);      // Weights for calculating \partial_t
  double ARRAY_DECL_NDT(timestepper_weights_dt_BDF2);      // Weights for calculating \partial_t
  double ARRAY_DECL_NDT(timestepper_weights_dt_Newmark2);  // Weights for calculating \partial_t
  double ARRAY_DECL_NDT(timestepper_weights_d2t_Newmark2); // Weights for calculating \partial^2_t

  // Possibly degraded variants
  double *timestepper_weights_dt_BDF2_degr;
  double *timestepper_weights_dt_Newmark2_degr;

  JITHangInfo_t ARRAY_DECL_NNODE(hanginfo_C1);   //[nodenum]
  JITHangInfo_t ARRAY_DECL_NNODE(hanginfo_C1TB); //[nodenum]  
  JITHangInfo_t ARRAY_DECL_NNODE(hanginfo_C2);   //[nodenum]
  JITHangInfo_t ARRAY_DECL_NNODE(hanginfo_C2TB); //[nodenum]
  JITHangInfo_t ARRAY_DECL_NNODE(hanginfo_Pos);

// Not really hanging, but used for bulk elements etc to remap the eqs  
  JITHangInfo_t ARRAY_DECL_NNODE(hanginfo_Discont);   


  struct JITShapeInfo *bulk_shapeinfo;
  // struct JITShapeInfo * otherbulk_shapeinfo; //Bulk element on the other side
  struct JITShapeInfo *opposite_shapeinfo; // Shape info on the other side
  int ARRAY_DECL_NNODE(opposite_node_index);                // Reindex the nodes on the opposite side (can be different due to orientation)
} JITShapeInfo_t;

typedef void (*JITFuncSpec_ResidualAndJacobian_FiniteElement)(const JITElementInfo_t *, const JITShapeInfo_t *, double *, double *, double *, unsigned);
typedef void (*JITFuncSpec_HessianVectorProduct_FiniteElement)(const JITElementInfo_t *, const JITShapeInfo_t *, const double *, double *, double *, unsigned, unsigned);
typedef void (*JITFuncSpec_GetZ2Fluxes_FiniteElement)(const JITElementInfo_t *, const JITShapeInfo_t *, double *);

typedef double (*JITFuncSpec_InitialCondition_FiniteElement)(const JITElementInfo_t *, int, double *, double *, double *,double, int, double);
typedef double (*JITFuncSpec_DirichletCondition_FiniteElement)(const JITElementInfo_t *, int, double *, double *, double *,double, double);
typedef double (*JITFuncSpec_EvalIntegralExpr_FiniteElement)(const JITElementInfo_t *, const JITShapeInfo_t *, unsigned);
typedef void (*JITFuncSpec_EvalTracerAdvection_FiniteElement)(const JITElementInfo_t *, const JITShapeInfo_t *, unsigned, double, double *);

typedef double (*JITFuncSpec_GeometricJacobian)(const JITElementInfo_t *, const double *);

typedef void (*JITFuncSpec_GeometricJacobianSpatialDerivative)(const JITElementInfo_t *, const double *, double *);

typedef struct JITFuncSpec_RequiredShapes_FiniteElement
{
  bool psi_C1,psi_C1TB, psi_C2, psi_C2TB, psi_DL, psi_D0;
  bool dx_psi_C1, dx_psi_C1TB, dx_psi_C2, dx_psi_C2TB, dx_psi_DL, dx_psi_D0; // Eulerian derivatives
  bool dX_psi_C1, dX_psi_C1TB,dX_psi_C2, dX_psi_C2TB, dX_psi_DL, dX_psi_D0; // Lagrangian derivatives
  bool psi_Pos, dx_psi_Pos, dX_psi_Pos;              // Position space. This is always the dominant element space, i.e. C2TB>C2>C1TB>C1. If an element has a "C2" and a "C1TB" space, it will be C2TB.
  bool normal_Pos;                                   // Normal required /( Normal is just considered to be defined on the Pos space)
  bool elemsize_Eulerian_Pos,elemsize_Lagrangian_Pos;
  bool elemsize_Eulerian_cartesian_Pos,elemsize_Lagrangian_cartesian_Pos;  
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


typedef struct JITFuncSpec_Table_FiniteElement
{
  unsigned int nodal_dim, lagr_dim;

  unsigned int numfields_C1, numfields_C1_bulk, numfields_C1_basebulk,numfields_C1_new; // Fields numfields_C1 are the total number of fields, numfields_C1_bulk are the ones which are defined on the bulk mesh (including the additional field of all parent interface meshes, numfields_C1_basebulk are indeed the fields that are directly implemented only at the lowest level. numfields_C1_new are the number of fields defined directly at this element level, i.e for lowest bulk level, numfields_C1_new=numfields_C1_bulk=numfields_C1_basebulk=numfields_C1. For interface elements, we get numfields_C1_new=numfields_C1-numfields_C1_bulk;
  char **fieldnames_C1;

  unsigned int numfields_C2, numfields_C2_bulk, numfields_C2_basebulk,numfields_C2_new;
  char **fieldnames_C2;

  unsigned int numfields_C2TB, numfields_C2TB_bulk, numfields_C2TB_basebulk,numfields_C2TB_new;
  char **fieldnames_C2TB;
  
  unsigned int numfields_C1TB, numfields_C1TB_bulk, numfields_C1TB_basebulk,numfields_C1TB_new;
  char **fieldnames_C1TB;  
  
  unsigned int numfields_D1, numfields_D1_bulk, numfields_D1_basebulk,numfields_D1_new; 
  char **fieldnames_D1;
  
  unsigned int numfields_D1TB, numfields_D1TB_bulk, numfields_D1TB_basebulk,numfields_D1TB_new;
  char **fieldnames_D1TB;    

  unsigned int numfields_D2, numfields_D2_bulk, numfields_D2_basebulk,numfields_D2_new;
  char **fieldnames_D2;

  unsigned int numfields_D2TB, numfields_D2TB_bulk, numfields_D2TB_basebulk,numfields_D2TB_new;
  char **fieldnames_D2TB;  

  unsigned int numfields_Pos;
  char **fieldnames_Pos;
  bool bulk_position_space_to_C1;
  
  int hangindex_C1,hangindex_C2,hangindex_C1TB,hangindex_C2TB,hangindex_Pos; //Hang indices

  unsigned int numfields_DL;
  char **fieldnames_DL;

  unsigned int numfields_D0;
  char **fieldnames_D0;

  unsigned int numfields_ED0;
  char **fieldnames_ED0;
  
  unsigned int nodal_offset_C2TB_basebulk,nodal_offset_C2_basebulk,nodal_offset_C1TB_basebulk,nodal_offset_C1_basebulk; //Offsets to the indices in the nodal values (basebulk fields only)
  unsigned int buffer_offset_C2TB_basebulk,buffer_offset_C2_basebulk,buffer_offset_C1TB_basebulk,buffer_offset_C1_basebulk; // Offsets in the nodal data buffer (basebulk fields only)
  unsigned int buffer_offset_C2TB_interf,buffer_offset_C2_interf,buffer_offset_C1TB_interf,buffer_offset_C1_interf; // Offsets in the nodal data buffer (interface fields only)
  
    
  
  unsigned int internal_offset_D2TB_new,internal_offset_D2_new,internal_offset_D1TB_new,internal_offset_D1_new; // Offset to the internal_data entries. These are only there on the current element level
  unsigned int buffer_offset_D2TB_basebulk,buffer_offset_D2_basebulk,buffer_offset_D1TB_basebulk,buffer_offset_D1_basebulk; // Offsets in the nodal data buffer (basebulk fields only)
  unsigned int buffer_offset_D2TB_interf,buffer_offset_D2_interf,buffer_offset_D1TB_interf,buffer_offset_D1_interf; // Offsets in the nodal data buffer (interface fields only)  
  
  unsigned int external_offset_D2TB_bulk,external_offset_D2_bulk,external_offset_D1TB_bulk,external_offset_D1_bulk; // Offset to the external_data entries. These refer to DG spaces on parent elements
  
  unsigned int internal_offset_DL,internal_offset_D0;
  unsigned int buffer_offset_DL,buffer_offset_D0; 
  unsigned int buffer_offset_ED0,external_offset_ED0;

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
  JITFuncSpec_ResidualAndJacobian_FiniteElement *ResidualAndJacobian_NoHang;
  bool * missing_residual_assembly; // Some residuals are not calculated (if not needed, e.g. for azimuthal eigenproblem). We cannot FD then!

  JITFuncSpec_HessianVectorProduct_FiniteElement *HessianVectorProduct;
  bool hessian_generated;

  unsigned num_Z2_flux_terms,num_Z2_flux_terms_for_eigen;
  JITFuncSpec_GetZ2Fluxes_FiniteElement GetZ2Fluxes,GetZ2FluxesForEigen;
  JITFuncSpec_RequiredShapes_FiniteElement_t shapes_required_Z2Fluxes;

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

