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

The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl

================================================================================*/


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

// No hanging macros
#define BEGIN_RESIDUAL_CONTINUOUS_SPACE(EQN, HANGINFO, NODALIND) BEGIN_RESIDUAL(EQN)
#define ADD_TO_RESIDUAL_CONTINUOUS_SPACE(RES) ADD_TO_RESIDUAL(RES)
#define END_RESIDUAL_CONTINUOUS_SPACE() END_RESIDUAL()
#define BEGIN_JACOBIAN_HANG(EQN, CONTRIB, HANGINFO, NODALIND) BEGIN_JACOBIAN_NOHANG(EQN, CONTRIB)
#define END_JACOBIAN_HANG() END_JACOBIAN_NOHANG()
#define ADD_TO_JACOBIAN_HANG_NOHANG(JPART) ADD_TO_JACOBIAN_NOHANG_NOHANG(JPART)
#define ADD_TO_JACOBIAN_NOHANG_HANG(JPART) ADD_TO_JACOBIAN_NOHANG_NOHANG(JPART)
#define ADD_TO_JACOBIAN_HANG_HANG(JPART) ADD_TO_JACOBIAN_NOHANG_NOHANG(JPART)

// Undef
#ifdef BEGIN_RESIDUAL_CONTINUOUS_SPACE
#undef BEGIN_RESIDUAL_CONTINUOUS_SPACE
#undef ADD_TO_RESIDUAL_CONTINUOUS_SPACE
#undef END_RESIDUAL_CONTINUOUS_SPACE
#undef BEGIN_JACOBIAN_HANG
#undef ADD_TO_JACOBIAN_HANG_NOHANG
#undef ADD_TO_JACOBIAN_NOHANG_HANG
#undef ADD_TO_JACOBIAN_HANG_HANG
#undef END_JACOBIAN_HANG
#endif

// Hanging macros
#define BEGIN_RESIDUAL_CONTINUOUS_SPACE(EQN, CONTRIB, HANGINFO, NODALIND, LINDEX)                                   \
  if (HANGINFO[LINDEX].nummaster)                                                                                   \
  {                                                                                                                 \
    nummaster = HANGINFO[LINDEX].nummaster;                                                                         \
  }                                                                                                                 \
  else                                                                                                              \
  {                                                                                                                 \
    nummaster = 1;                                                                                                  \
  }                                                                                                                 \
  _res_contrib = CONTRIB;                                                                                       \
  for (int m = 0; m < nummaster; m++)                                                                               \
  {                                                                                                                 \
    if (HANGINFO[LINDEX].nummaster)                                                                                 \
    {                                                                                                               \
      local_eqn = HANGINFO[LINDEX].masters[m].local_eqn[NODALIND];                                                  \
      hang_weight = HANGINFO[LINDEX].masters[m].weight;                                                             \
    }                                                                                                               \
    else                                                                                                            \
    {                                                                                                               \
      local_eqn = EQN;                                                                                              \
      hang_weight = 1.0;                                                                                            \
    }                                                                                                               \
    if (local_eqn >= 0) /*&& (!eleminfo->nullified_residual_dof || !eleminfo->nullified_residual_dof[local_eqn] )*/ \
    {

#define ADD_TO_RESIDUAL_CONTINUOUS_SPACE() \
  assert(local_eqn < eleminfo->ndof);      \
  residuals[local_eqn] += hang_weight * _res_contrib;

#define END_RESIDUAL_CONTINUOUS_SPACE() \
  }                                     \
  }

#define BEGIN_JACOBIAN_HANG(EQN, CONTRIB, HANGINFO, NODALIND, LINDEX)   \
  if (HANGINFO[LINDEX].nummaster)                                       \
  {                                                                     \
    nummaster2 = HANGINFO[LINDEX].nummaster;                            \
  }                                                                     \
  else                                                                  \
  {                                                                     \
    nummaster2 = 1;                                                     \
  }                                                                     \
  _J_contrib = CONTRIB;                                               \
  for (int m2 = 0; m2 < nummaster2; m2++)                               \
  {                                                                     \
    if (HANGINFO[LINDEX].nummaster)                                     \
    {                                                                   \
      local_unknown = HANGINFO[LINDEX].masters[m2].local_eqn[NODALIND]; \
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
  }

// Hanging macros
#define BEGIN_HESSIAN_TEST_LOOP_CONTINUOUS_SPACE(EQN, HANGINFO, NODALIND, LINDEX)                                     \
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
      local_eqn = HANGINFO[LINDEX].masters[m].local_eqn[NODALIND];                                                    \
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

#define BEGIN_HESSIAN_SHAPE_LOOP1_CONTINUOUS_SPACE(EQN, HANGINFO, NODALIND, LINDEX) \
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
      local_unknown = HANGINFO[LINDEX].masters[m2].local_eqn[NODALIND];             \
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

#define BEGIN_HESSIAN_SHAPE_LOOP2_CONTINUOUS_SPACE(EQN, CONTRIB, HANGINFO, NODALIND, LINDEX) \
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
      local_deriv = HANGINFO[LINDEX].masters[m3].local_eqn[NODALIND];                        \
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
