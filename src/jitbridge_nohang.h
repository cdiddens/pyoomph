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

// No hanging spaces or without considering hang infos
#define BEGIN_RESIDUAL(EQN)                                                                                           \
      local_eqn = EQN;                                                                                                \
      if (local_eqn >= 0) /*&& (!eleminfo->nullified_residual_dof || !eleminfo->nullified_residual_dof[local_eqn] )*/ \
      {

#define ADD_TO_RESIDUAL(RES) residuals[local_eqn] += RES;
#define END_RESIDUAL() }

#define BEGIN_JACOBIAN() \
      if (flag)          \
      {

#define ADD_TO_JACOBIAN_NOHANG_NOHANG(JPART) jacobian[local_eqn * shapeinfo->jacobian_size + local_unknown] += JPART;

#define ADD_TO_MASS_MATRIX_NOHANG_NOHANG(JPART)                                            \
      if (flag == 2)                                                                       \
      {                                                                                    \
            mass_matrix[local_eqn * shapeinfo->mass_matrix_size + local_unknown] += JPART; \
      }

#define END_JACOBIAN() }

#define BEGIN_JACOBIAN_NOHANG(EQN) \
      local_unknown = EQN;         \
      if (local_unknown >= 0)      \
      {

#define END_JACOBIAN_NOHANG() }

#define BEGIN_HESSIAN_TEST_LOOP(EQN)                                                                                    \
      local_eqn = EQN;                                                                                                  \
      if (local_eqn >= 0) /* && (!eleminfo->nullified_residual_dof || !eleminfo->nullified_residual_dof[local_eqn] ) */ \
      {

#define END_HESSIAN_TEST_LOOP() }

#define BEGIN_HESSIAN_SHAPE_LOOP1(EQN) \
      local_unknown = EQN;             \
      if (local_unknown >= 0)          \
      {

#define END_HESSIAN_SHAPE_LOOP1() }

#define BEGIN_HESSIAN_SHAPE_LOOP2(EQN, CONTRIB) \
      local_deriv = EQN;                        \
      if (local_deriv >= 0)                     \
      {                                         \
            _H_contrib = CONTRIB;

#define END_HESSIAN_SHAPE_LOOP2() }

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
#undef BEGIN_HESSIAN_TEST_LOOP_CONTINUOUS_SPACE
#undef END_HESSIAN_TEST_LOOP_CONTINUOUS_SPACE
#undef BEGIN_HESSIAN_SHAPE_LOOP1_CONTINUOUS_SPACE
#undef END_HESSIAN_SHAPE_LOOP1_CONTINUOUS_SPACE
#endif

// No hanging macros
#define BEGIN_RESIDUAL_CONTINUOUS_SPACE(EQN, HANGINFO, NODALIND) BEGIN_RESIDUAL(EQN)
#define ADD_TO_RESIDUAL_CONTINUOUS_SPACE(RES) ADD_TO_RESIDUAL(RES)
#define END_RESIDUAL_CONTINUOUS_SPACE() END_RESIDUAL()
#define BEGIN_JACOBIAN_HANG(EQN, HANGINFO, NODALIND) BEGIN_JACOBIAN_NOHANG(EQN)
#define END_JACOBIAN_HANG() END_JACOBIAN_NOHANG()
#define ADD_TO_JACOBIAN_HANG_NOHANG(JPART) ADD_TO_JACOBIAN_NOHANG_NOHANG(JPART)
#define ADD_TO_JACOBIAN_NOHANG_HANG(JPART) ADD_TO_JACOBIAN_NOHANG_NOHANG(JPART)
#define ADD_TO_JACOBIAN_HANG_HANG(JPART) ADD_TO_JACOBIAN_NOHANG_NOHANG(JPART)

#define BEGIN_HESSIAN_TEST_LOOP_CONTINUOUS_SPACE(EQN, HANGINFO, NODALIND, LINDEX) BEGIN_HESSIAN_TEST_LOOP(EQN)
#define END_HESSIAN_TEST_LOOP_CONTINUOUS_SPACE() END_HESSIAN_TEST_LOOP()

#define BEGIN_HESSIAN_SHAPE_LOOP1_CONTINUOUS_SPACE(EQN, HANGINFO, NODALIND, LINDEX) BEGIN_HESSIAN_SHAPE_LOOP1(EQN)
#define BEGIN_HESSIAN_SHAPE_LOOP2_CONTINUOUS_SPACE(EQN, CONTRIB, HANGINFO, NODALIND, LINDEX) BEGIN_HESSIAN_SHAPE_LOOP2(EQN, CONTRIB)
