#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2026  Christian Diddens & Duarte Rocha
# 
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
# 
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
# 
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
#
#  The authors may be contacted at c.diddens@utwente.nl and d.rocha@utwente.nl
#
# ========================================================================
 
from ..expressions.generic import is_zero,subexpression
import math
from .cb import *
from .coordsys import BaseCoordinateSystem,AxisymmetricCoordinateSystem,CartesianCoordinateSystem,AxisymmetryBreakingCoordinateSystem
from ..typings import *
from .generic import matrix,ExpressionOrNum,Expression, scale_factor

# Calling this object after creating takes a symmetric tensor T as argument
# It's call will return a matrix R and a diagonal matrix D so that
#   T=matproduct(R,matproduct(D,transpose(R)))
# i.e. T=R*D*R^t
# If you use dimensions, you must set the scale of the tensor entries as 'scale' argument.
# This scale will be in D, whereas R does not have any physical dimensions
class DiagonalizeSymmetricTensor(CustomMultiReturnExpression):
    def __init__(self,coordinate_system:BaseCoordinateSystem,dim:int,scale:Union[ExpressionOrNum,str]=1,fill_to_max_vector_dim:bool=True,use_FD:Union[bool,float]=False) -> None:
        super().__init__()
        if isinstance(coordinate_system,AxisymmetricCoordinateSystem):
            if isinstance(coordinate_system,AxisymmetryBreakingCoordinateSystem):
                raise RuntimeError("Not implemented for this coordinate system: "+str(coordinate_system))
            self.axisymmetric=True
        elif isinstance(coordinate_system,CartesianCoordinateSystem):
            self.axisymmetric=False
        else:
            raise RuntimeError("Not implemented for this coordinate system: "+str(coordinate_system))
    
        self.dim=dim
        if self.dim!=2:
            raise RuntimeError("Currently only implemented for 2 dimensional tensors")        
        if isinstance(scale,str):
            scale=scale_factor(scale)
        self.scale=scale
        self.fill_to_max_vector_dim=fill_to_max_vector_dim # Fill to 3x3 [Filled with 0] or keep it at dim x dim ?
        self.use_FD=use_FD
        self.FD_epsilon=1e-8
        if isinstance(self.use_FD,float):
            self.FD_epsilon=self.use_FD

    # Input arguments, i.e. the tensor, to scalar list
    def process_args_to_scalar_list(self,*args: ExpressionOrNum)->List[ExpressionOrNum]:
        assert len(args)==1
        M=args[0]
        assert isinstance(M,Expression)
        if not self.axisymmetric:
            if self.dim==2:
                return [M[0,0]/self.scale,M[0,1]/self.scale,M[1,1]/self.scale] # Nondimensional relevant matrix entries 
            else:
                raise RuntimeError("TODO: "+str(self.dim)+"-dimensional case here")
        else:
            return [M[0,0]/self.scale,M[0,1]/self.scale,M[1,1]/self.scale, M[2,2]/self.scale]
        

    # How many scalar values will be returned. You can also check the number of scalar input values here
    def get_num_returned_scalars(self,nargs:int)->int:
        if not self.axisymmetric:
            if self.dim==2:
                if nargs!=3:
                    raise RuntimeError("Expected 3 input arguments!")
                return 6
            else:
                raise RuntimeError("TODO: "+str(self.dim)+"-dimensional case here")
        else:
            return 7


    # Evaluate the function. Depending on the case, we will call a specific routine
    def eval(self,flag:int,arg_list:NPFloatArray,result_list:NPFloatArray,derivative_matrix:NPFloatArray):
        if not self.axisymmetric:
            if self.dim==2:
                self.eval_2d_cartesian(flag,arg_list,result_list,derivative_matrix) # Call the Python eval for 2d Cartesian
            else:
                raise RuntimeError("TODO: "+str(self.dim)+"-dimensional case here")
        else:
            self.eval_axisymmetric(flag,arg_list,result_list,derivative_matrix)
        
    # Get the C code
    def generate_c_code(self) -> str:
        if not self.axisymmetric:
            if self.dim==2:
                return self.generate_c_code_2d_cartesian() # Generate the C code for 2d Cartesian
            else:
                raise RuntimeError("TODO: "+str(self.dim)+"-dimensional case here")
        else:
            return self.generate_c_code_axisymmetric()

    # Convert the scalar result list back to matrices
    def process_result_list_to_results(self, result_list: List[Expression]) -> Tuple[ExpressionOrNum,...]:
        if not self.axisymmetric:
            if self.dim==2:
                # Rebuild matrices, also apply the scale again on the diagonal matrix
                R=matrix([[result_list[0],result_list[1]],[result_list[2],result_list[3]]],fill_to_max_vector_dim=self.fill_to_max_vector_dim)
                D=matrix([[result_list[4]*self.scale,0],[0,result_list[5]*self.scale]],fill_to_max_vector_dim=self.fill_to_max_vector_dim)
                return R,D
            else:
                raise RuntimeError("TODO: "+str(self.dim)+"-dimensional case here")
        else:
            R=matrix([[result_list[0],result_list[1],0],[result_list[2],result_list[3],0],[0,0,1]],fill_to_max_vector_dim=self.fill_to_max_vector_dim)
            D=matrix([[result_list[4]*self.scale,0,0],[0,result_list[5]*self.scale,0],[0,0,result_list[6]*self.scale]],fill_to_max_vector_dim=self.fill_to_max_vector_dim)
            return R,D
        
    # 2d Cartesian case, evaluation and Jacobian
    def eval_2d_cartesian(self,flag:int,arg_list:NPFloatArray,result_list:NPFloatArray,derivative_matrix:NPFloatArray):
        M11=arg_list[0]
        M12=arg_list[1]
        M22=arg_list[2]

        if M12*M12<1e-15:
            Rxx=Ryy=1.0
            Rxy=Ryx=0.0
            Lx,Ly=M11,M22
        else:
            T=M11+M22
            D=M11*M22-M12*M12
            Rxx=Rxy=M12
            Ryx=Ryy=-M11
            Lx=T/2+numpy.sqrt(T**2/4-D)
            Ryx+=Lx
            mod=numpy.sqrt(Rxx**2+Ryx**2)
            Rxx/=mod
            Ryx/=mod

            Ly=T/2-numpy.sqrt(T**2/4-D)
            Ryy+=Ly
            mod=numpy.sqrt(Rxy**2+Ryy**2)
            Rxy/=mod
            Ryy/=mod

            Ryy*=-1
            Ryx*=-1

        result_list[:]=numpy.array([Rxx,Rxy,Ryx,Ryy,Lx,Ly])[:]        

        if flag:
            if self.use_FD:
                self.fill_python_derivatives_by_FD(arg_list,result_list,derivative_matrix,self.FD_epsilon)
            else:
                if M12*M12<1e-15:
                    derivative_matrix[:,:]=numpy.array([[0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])[:]
                else:
                    x0 = M11/4 - M22/4
                    x1 = M12**2
                    x2 = numpy.sqrt(-M11*M22 + x1 + (M11 + M22)**2/4)
                    x3 = 1/x2
                    x4 = x0*x3
                    x5 = 2*x4
                    x6 = x5 - 1
                    x7 = M11/2
                    x8 = M22/2 + x2 - x7
                    x9 = x8**2
                    x10 = x1 + x9
                    x11 = x10**(-3/2)
                    x12 = x11/2
                    x13 = M12*x12*x8
                    x14 = 1/numpy.sqrt(x10)
                    x15 = M12*x3
                    x16 = x11*(-M12 - x15*x8)
                    x17 = -x0*x3
                    x18 = 2*x17
                    x19 = x18 + 1
                    x20 = -x5 - 1
                    x21 = M22/2 - x2 - x7
                    x22 = x21**2
                    x23 = x1 + x22
                    x24 = x23**(-3/2)
                    x25 = x24/2
                    x26 = M12*x21*x25
                    x27 = 1/numpy.sqrt(x23)
                    x28 = -M12 + x15*x21
                    x29 = 1 - x18
                    x30 = x4 - 1/2
                    x31 = x12*x9
                    x32 = x17 + 1/2
                    x33 = x4 + 1/2
                    x34 = x22*x25
                    x35 = 1/2 - x17
                    derivative_matrix[:,:]=numpy.array([[-x13*x6, M12*x16 + x14, -x13*x19], [-x20*x26, M12*x24*x28 + x27, -x26*x29], [x14*x30 - x31*x6, x14*x15 + x16*x8, x14*x32 - x19*x31], [-x20*x34 - x27*x33, -x15*x27 + x21*x24*x28, x27*x35 - x29*x34], [x33, x15, x32], [-x30, -x15, x35]])[:]

            #self.debug_python_derivatives_with_FD(arg_list,result_list,derivative_matrix,error_threshold=1e-6)

    def generate_c_code_2d_cartesian(self) -> str:
        ccode= """
// 2d Cartesian version of DiagonalizeSymmetricTensor
const double M11=arg_list[0];
const double M12=arg_list[1];
const double M22=arg_list[2];
double Rxx,Ryy,Rxy,Ryx,Lx,Ly;
if (M12*M12<1e-15)
{
   Rxx=Ryy=1.0;
   Rxy=Ryx=0.0;
   Lx=M11; Ly=M22;
}
else
{
   double T=M11+M22;
   double D=M11*M22-M12*M12;
   Rxx=Rxy=M12;
   Ryx=Ryy=-M11;
   Lx=T/2.0+sqrt(T*T/4.0-D);
   Ryx+=Lx;
   double mod=sqrt(Rxx*Rxx+Ryx*Ryx);
   Rxx/=mod;
   Ryx/=mod;
   Ly=T/2.0-sqrt(T*T/4.0-D);
   Ryy+=Ly;
   mod=sqrt(Rxy*Rxy+Ryy*Ryy);
   Rxy/=mod;
   Ryy/=mod;
}
result_list[0]=Rxx;
result_list[1]=Rxy;
result_list[2]=Ryx;
result_list[3]=Ryy;
result_list[4]=Lx;
result_list[5]=Ly;
"""
        if self.use_FD:
            ccode+="""
FILL_MULTI_RET_JACOBIAN_BY_FD("""+str(self.FD_epsilon)+ """)            
"""
        else:
            ccode+="""
if (flag)
{
   if (M12*M12<1e-15)
   {
      derivative_matrix[0]=derivative_matrix[1]=derivative_matrix[2]=0.0;
      derivative_matrix[3]=derivative_matrix[4]=derivative_matrix[5]=0.0;
      derivative_matrix[6]=derivative_matrix[7]=derivative_matrix[8]=0.0;
      derivative_matrix[9]=derivative_matrix[10]=derivative_matrix[11]=0.0;
		derivative_matrix[12]=1.0; derivative_matrix[13]=derivative_matrix[14]=0.0;
		derivative_matrix[15]=derivative_matrix[16]=0.0; derivative_matrix[17]=1.0;      		      
   }
   else
   {
       const double x0 = M11/4.0 - M22/4.0;
       const double x1 = M12*M12;
       const double x2 = sqrt(-M11*M22 + x1 + pow(M11 + M22,2)/4.0);
       const double x3 = 1.0/x2;
       const double x4 = x0*x3;
       const double x5 = 2.0*x4;
       const double x6 = x5 - 1.0;
       const double x7 = M11/2.0;
       const double x8 = M22/2.0 + x2 - x7;
       const double x9 = x8*x8;
       const double x10 = x1 + x9;
       const double x11 = pow(x10,-3.0/2.0);
       const double x12 = x11/2.0;
       const double x13 = M12*x12*x8;
       const double x14 = 1.0/sqrt(x10);
       const double x15 = M12*x3;
       const double x16 = x11*(-M12 - x15*x8);
       const double x17 = -x0*x3;
       const double x18 = 2*x17;
       const double x19 = x18 + 1.0;
       const double x20 = -x5 - 1.0;
       const double x21 = M22/2.0 - x2 - x7;
       const double x22 = x21*x21;
       const double x23 = x1 + x22;
       const double x24 = pow(x23,-3.0/2.0);
       const double x25 = x24/2.0;
       const double x26 = M12*x21*x25;
       const double x27 = 1.0/sqrt(x23);
       const double x28 = -M12 + x15*x21;
       const double x29 = 1.0 - x18;
       const double x30 = x4 - 1.0/2.0;
       const double x31 = x12*x9;
       const double x32 = x17 + 1.0/2.0;
       const double x33 = x4 + 1.0/2.0;
       const double x34 = x22*x25;
       const double x35 = 1.0/2.0 - x17;
       derivative_matrix[0]=-x13*x6; derivative_matrix[1]=M12*x16 + x14;derivative_matrix[2]= -x13*x19; 
       derivative_matrix[3]=-x20*x26;derivative_matrix[4]= M12*x24*x28 + x27;derivative_matrix[5]= -x26*x29;
       derivative_matrix[6]=x14*x30 - x31*x6;derivative_matrix[7]= x14*x15 + x16*x8;derivative_matrix[8]= x14*x32 - x19*x31;
       derivative_matrix[9]=-x20*x34 - x27*x33; derivative_matrix[10]=-x15*x27 + x21*x24*x28;derivative_matrix[11]= x27*x35 - x29*x34;
       derivative_matrix[12]=x33;derivative_matrix[13]= x15;derivative_matrix[14]= x32;
       derivative_matrix[15]=-x30;derivative_matrix[16]= -x15;derivative_matrix[17]= x35;
    }
}
        """
            return ccode
        
# Axisymmetric case, evaluation and Jacobian
    def eval_axisymmetric(self,flag:int,arg_list:NPFloatArray,result_list:NPFloatArray,derivative_matrix:NPFloatArray):
        M11=arg_list[0]
        M12=arg_list[1]
        M22=arg_list[2]
        M33=arg_list[3]

        if M12*M12<1e-15:
            Rxx=Ryy=1.0
            Rxy=Ryx=0.0
            Lx,Ly=M11,M22
        else:
            T=M11+M22
            D=M11*M22-M12*M12
            Rxx=Rxy=M12
            Ryx=Ryy=-M11
            Lx=T/2+numpy.sqrt(T**2/4-D)
            Ryx+=Lx
            mod=numpy.sqrt(Rxx**2+Ryx**2)
            Rxx/=mod
            Ryx/=mod

            Ly=T/2-numpy.sqrt(T**2/4-D)
            Ryy+=Ly
            mod=numpy.sqrt(Rxy**2+Ryy**2)
            Rxy/=mod
            Ryy/=mod

        result_list[:]=numpy.array([Rxx,Rxy,Ryx,Ryy,Lx,Ly,M33])[:]        

        if flag:
            if self.use_FD:
                self.fill_python_derivatives_by_FD(arg_list,result_list,derivative_matrix,self.FD_epsilon)
            else:
                if M12*M12<1e-15:
                    derivative_matrix[:,:]=numpy.array([[0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0], [0.0,0.0,1.0,0.0], [0.0,0.0,0.0,1.0]])[:]    
                else:
                    x0=M12**2
                    x1=4*x0
                    x2=M11*M22
                    x3=numpy.sqrt(x1 - 4*x2 + (M11 + M22)**2)
                    x4=-M11 + M22
                    x5=x3 + x4
                    x6=x5**2
                    x7=x1 + x6
                    x8=x7**(-3/2)
                    x9=1/x3
                    x10=2*x9
                    x11=M12*x10
                    x12=x11*x6*x8
                    x13=2*x3
                    x14=M11 - M22
                    x15=x14 + x3
                    x16=x15**2
                    x17=x1 + x16
                    x18=x17**(-3/2)
                    x19=x11*x18
                    x20=M11**2 + M22**2 + x1 - 2*x2
                    x21=numpy.sqrt(x20)
                    x22=M22*x21
                    x23=M11*x21
                    x24=1/numpy.sqrt(x20 + x22 - x23)
                    x25=numpy.sqrt(2)/x20
                    x26=x0*x25
                    x27=x24*x26
                    x28=M12*x14*x25
                    x29=1/numpy.sqrt(x20 - x22 + x23)
                    x30=x9/2
                    x31=x15*x30
                    x32=x30*x5
                    derivative_matrix[:,:]=numpy.array([[x12, x10*x8*(-x1*(x13 + x4) + x3*x7), -x12, 0], [-x16*x19, x10*x18*(-x1*(x13 + x14) + x17*x3), x15**2*x19, 0], [-x27, x24*x28, x27, 0], [-x26*x29, x28*x29, x1*x15*x18*x9, 0], [x31, x11, x32, 0], [x32, -x11, x31, 0], [0, 0, 0, 1]])[:]

            #self.debug_python_derivatives_with_FD(arg_list,result_list,derivative_matrix,error_threshold=1e-6)


    def generate_c_code_axisymmetric(self) -> str:
        ccode="""
// Axisymmetric version of DiagonalizeSymmetricTensor
const double M11=arg_list[0];
const double M12=arg_list[1];
const double M22=arg_list[2];
const double M33=arg_list[3];
double Rxx,Ryy,Rxy,Ryx,Lx,Ly;
if (M12*M12<1e-15)
{
   Rxx=Ryy=1.0;
   Rxy=Ryx=0.0;
   Lx=M11; Ly=M22;
}
else
{
   double T=M11+M22;
   double D=M11*M22-M12*M12;
   Rxx=Rxy=M12;
   Ryx=Ryy=-M11;
   Lx=T/2.0+sqrt(T*T/4.0-D);
   Ryx+=Lx;
   double mod=sqrt(Rxx*Rxx+Ryx*Ryx);
   Rxx/=mod;
   Ryx/=mod;
   Ly=T/2.0-sqrt(T*T/4.0-D);
   Ryy+=Ly;
   mod=sqrt(Rxy*Rxy+Ryy*Ryy);
   Rxy/=mod;
   Ryy/=mod;
}
result_list[0]=Rxx;
result_list[1]=Rxy;
result_list[2]=Ryx;
result_list[3]=Ryy;
result_list[4]=Lx;
result_list[5]=Ly;
result_list[6]=M33;
"""
        if self.use_FD:
            ccode+="""
FILL_MULTI_RET_JACOBIAN_BY_FD("""+str(self.FD_epsilon)+""")
"""
        else:
            ccode+="""
if (flag)
{
   if (M12*M12<1e-15)
   {
      derivative_matrix[0]=derivative_matrix[1]=derivative_matrix[2]=derivative_matrix[3]=0.0;
      derivative_matrix[4]=derivative_matrix[5]=derivative_matrix[6]=derivative_matrix[7]=0.0;
      derivative_matrix[8]=derivative_matrix[9]=derivative_matrix[10]=derivative_matrix[11]=0.0;
      derivative_matrix[12]=derivative_matrix[13]=derivative_matrix[14]=derivative_matrix[15]=0.0;
	  derivative_matrix[16]=1.0; derivative_matrix[17]=derivative_matrix[18]=derivative_matrix[19]=0.0;
      derivative_matrix[22]=1.0; derivative_matrix[20]=derivative_matrix[21]=derivative_matrix[23]=0.0;    		      
      derivative_matrix[27]=1.0; derivative_matrix[24]=derivative_matrix[25]=derivative_matrix[26]=0.0;
   }
   else
   {    
        const double x0 = pow(M12, 2);
        const double x1 = 4.0*x0;
        const double x2 = M11*M22;
        const double x3 = sqrt(x1 - 4*x2 + pow(M11 + M22, 2));
        const double x4 = -M11 + M22;
        const double x5 = x3 + x4;
        const double x6 = pow(x5, 2);
        const double x7 = x1 + x6;
        const double x8 = pow(x7, -3.0/2.0);
        const double x9 = 1.0/x3;
        const double x10 = 2*x9;
        const double x11 = M12*x10;
        const double x12 = x11*x6*x8;
        const double x13 = 2*x3;
        const double x14 = M11 - M22;
        const double x15 = x14 + x3;
        const double x16 = pow(x15, 2);
        const double x17 = x1 + x16;
        const double x18 = pow(x17, -3.0/2.0);
        const double x19 = x11*x18;
        const double x20 = pow(M11, 2) + pow(M22, 2) + x1 - 2*x2;
        const double x21 = sqrt(x20);
        const double x22 = M22*x21;
        const double x23 = M11*x21;
        const double x24 = pow(x20 + x22 - x23, -1.0/2.0);
        const double x25 = sqrt(2)/x20;
        const double x26 = x0*x25;
        const double x27 = x24*x26;
        const double x28 = M12*x14*x25;
        const double x29 = pow(x20 - x22 + x23, -1.0/2.0);
        const double x30 = (1.0/2.0)*x9;
        const double x31 = x15*x30;
        const double x32 = x30*x5;
        derivative_matrix[0]=x12; derivative_matrix[1]=x10*x8*(-x1*(x13 + x4) + x3*x7); derivative_matrix[2]=-x12; derivative_matrix[3]=0.0;
        derivative_matrix[4]=-x16*x19; derivative_matrix[5]=x10*x18*(-x1*(x13 + x14) + x17*x3); derivative_matrix[6]=pow(x15, 2)*x19; derivative_matrix[7]=0.0;
        derivative_matrix[8]=-x27; derivative_matrix[9]=x24*x28; derivative_matrix[10]=x27; derivative_matrix[11]=0.0;
        derivative_matrix[12]=-x26*x29; derivative_matrix[13]=x28*x29; derivative_matrix[14]=x1*x15*x18*x9; derivative_matrix[15]=0.0;
        derivative_matrix[16]=x31; derivative_matrix[17]=x11; derivative_matrix[18]=x32; derivative_matrix[19]=0.0;
        derivative_matrix[20]=x32; derivative_matrix[21]=-x11; derivative_matrix[22]=x31; derivative_matrix[23]=0.0;
        derivative_matrix[24]=0.0; derivative_matrix[25]=0.0; derivative_matrix[26]=0.0; derivative_matrix[27]=1.0;
    }
}
        """
        return ccode
        

# Expects the diagonalizing tensor R, the velocity gradient grad(u) and the diagonal matrix ev
# Only works in 2d Cartesian!
# Returns B tensor and Omega tensor with case distinguishment
# if the eigenvalues are degenerate, return B=1/2*sym(grad(u)), Omega=0
# Else perform the transformation of the paper
class LogConfTensorDecompositionCartesian2d(CustomMultiReturnExpression):
    def __init__(self,epsilon=1e-7,use_subexpression:bool=True) -> None:
        super().__init__()
        self.epsilon=epsilon
        self.use_subexpression=use_subexpression
    
    # Take the args R, grad(u) and ev and assemple it to a list
    def process_args_to_scalar_list(self, *args: "ExpressionOrNum") -> List["ExpressionOrNum"]:
        R=args[0]
        gradu=args[1]
        ev=args[2]
        return [R[0,0],R[0,1],R[1,0],R[1,1],gradu[0,0],gradu[0,1],gradu[1,0],gradu[1,1],ev[0,0],ev[1,1]]
    
    # To the calculations in python, including the derivatives
    def eval(self, flag: int, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray) -> None:
        R=numpy.array([[arg_list[0],arg_list[1]],[arg_list[2],arg_list[3]]])
        gradU=numpy.array([[arg_list[4],arg_list[5]],[arg_list[6],arg_list[7]]])
        ev0=arg_list[8]
        ev1=arg_list[9]        
        divisor=ev1-ev0
        if abs(divisor)<self.epsilon:
            B=(gradU+numpy.transpose(gradU))/2
            Omega=numpy.zeros((2,2),dtype=numpy.float64)
            if flag:
                derivative_matrix.fill(0.0)
                derivative_matrix[0,4] = 1.0
                derivative_matrix[1,5] = 1.0/2.0
                derivative_matrix[1,6] = 1.0/2.0
                derivative_matrix[2,5] = 1.0/2.0
                derivative_matrix[2,6] = 1.0/2.0
                derivative_matrix[3,7] = 1.0
        else:
            m=numpy.matmul(numpy.matmul(numpy.transpose(R),gradU),R)
            omega_val=(ev1*m[0,1]+ev0*m[1,0])/divisor
            B=numpy.matmul(numpy.matmul(R,numpy.array([[m[0,0],0],[0,m[1,1]]])),numpy.transpose(R))   
            Omega=numpy.matmul(numpy.matmul(R,numpy.array([[0,omega_val],[-omega_val,0]])),numpy.transpose(R))      
            if flag:
                x0 = R[0,0]**2
                x1 = R[1,0]*gradU[0,1]
                x2 = R[1,0]*gradU[1,0]
                x3 = R[0,0]*gradU[0,0]
                x4 = x1 + x2 + 2*x3
                x5 = x2 + x3
                x6 = R[0,0]*gradU[0,1]
                x7 = R[1,0]*gradU[1,1]
                x8 = x6 + x7
                x9 = R[0,0]*x5 + R[1,0]*x8
                x10 = R[0,0]*x9
                x11 = R[0,1]**2
                x12 = R[1,1]*gradU[0,1]
                x13 = R[1,1]*gradU[1,0]
                x14 = R[0,1]*gradU[0,0]
                x15 = x12 + x13 + 2*x14
                x16 = x13 + x14
                x17 = R[0,1]*gradU[0,1]
                x18 = R[1,1]*gradU[1,1]
                x19 = x17 + x18
                x20 = R[0,1]*x16 + R[1,1]*x19
                x21 = R[0,1]*x20
                x22 = R[0,0]*gradU[1,0]
                x23 = x22 + x6 + 2*x7
                x24 = R[0,1]*gradU[1,0]
                x25 = x17 + 2*x18 + x24
                x26 = R[0,0]**3*R[1,0] + R[0,1]**3*R[1,1]
                x27 = R[1,0]**2
                x28 = R[1,1]**2
                x29 = x0*x27 + x11*x28
                x30 = R[0,0]*R[1,0]
                x31 = R[1,0]*x9
                x32 = x30*x4 + x31
                x33 = R[0,1]*R[1,1]
                x34 = R[1,1]*x20
                x35 = x15*x33 + x34
                x36 = x10 + x23*x30
                x37 = x21 + x25*x33
                x38 = R[0,0]*R[1,0]**3 + R[0,1]*R[1,1]**3
                x39 = R[0,0]*R[1,1]
                x40 = -ev0 + ev1
                x41 = 1/x40
                x42 = x41*(ev0*x16 + ev1*(x12 + x14))
                x43 = R[0,1]*R[1,0]
                x44 = R[0,0]*x16 + R[1,0]*x19
                x45 = R[0,1]*x5 + R[1,1]*x8
                x46 = ev0*x44 + ev1*x45
                x47 = x41*x46
                x48 = R[1,1]*x47 + x39*x42 - x42*x43
                x49 = ev0*(x1 + x3) + ev1*x5
                x50 = -R[0,0]*R[1,1]*x41*x49 + R[1,0]*x47 + x41*x43*x49
                x51 = ev0*x19 + ev1*(x18 + x24)
                x52 = -R[0,0]*R[1,1]*x41*x51 + R[0,1]*x47 + x41*x43*x51
                x53 = x41*(ev0*(x22 + x7) + ev1*x8)
                x54 = R[0,0]*x47 + x39*x53 - x43*x53
                x55 = R[0,0]*R[0,1]
                x56 = x41*(ev0*x55 + ev1*x55)
                x57 = x39*x56 - x43*x56
                x58 = x41*(ev0*x43 + ev1*x39)
                x59 = x39*x58 - x43*x58
                x60 = x41*(ev0*x39 + ev1*x43)
                x61 = x39*x60 - x43*x60
                x62 = R[1,0]*R[1,1]
                x63 = x41*(ev0*x62 + ev1*x62)
                x64 = x39*x63 - x43*x63
                x65 = x41*x44
                x66 = x46/x40**2
                x67 = x39*x66
                x68 = x43*x66
                x69 = x39*x65 - x43*x65 + x67 - x68
                x70 = x41*x45
                x71 = x39*x70 - x43*x70 - x67 + x68
                #Jacobian entries:
                derivative_matrix.fill(0.0)
                derivative_matrix[0,0] = x0*x4 + 2*x10
                derivative_matrix[0,1] = x11*x15 + 2*x21
                derivative_matrix[0,2] = x0*x23
                derivative_matrix[0,3] = x11*x25
                derivative_matrix[0,4] = R[0,0]**4 + R[0,1]**4
                derivative_matrix[0,5] = x26
                derivative_matrix[0,6] = x26
                derivative_matrix[0,7] = x29
                derivative_matrix[1,0] = x32
                derivative_matrix[1,1] = x35
                derivative_matrix[1,2] = x36
                derivative_matrix[1,3] = x37
                derivative_matrix[1,4] = x26
                derivative_matrix[1,5] = x29
                derivative_matrix[1,6] = x29
                derivative_matrix[1,7] = x38
                derivative_matrix[2,0] = x32
                derivative_matrix[2,1] = x35
                derivative_matrix[2,2] = x36
                derivative_matrix[2,3] = x37
                derivative_matrix[2,4] = x26
                derivative_matrix[2,5] = x29
                derivative_matrix[2,6] = x29
                derivative_matrix[2,7] = x38
                derivative_matrix[3,0] = x27*x4
                derivative_matrix[3,1] = x15*x28
                derivative_matrix[3,2] = x23*x27 + 2*x31
                derivative_matrix[3,3] = x25*x28 + 2*x34
                derivative_matrix[3,4] = x29
                derivative_matrix[3,5] = x38
                derivative_matrix[3,6] = x38
                derivative_matrix[3,7] = R[1,0]**4 + R[1,1]**4
                derivative_matrix[4,0] = x48
                derivative_matrix[4,1] = -x50
                derivative_matrix[4,2] = -x52
                derivative_matrix[4,3] = x54
                derivative_matrix[4,4] = x57
                derivative_matrix[4,5] = x59
                derivative_matrix[4,6] = x61
                derivative_matrix[4,7] = x64
                derivative_matrix[4,8] = x69
                derivative_matrix[4,9] = x71
                derivative_matrix[5,0] = -x48
                derivative_matrix[5,1] = x50
                derivative_matrix[5,2] = x52
                derivative_matrix[5,3] = -x54
                derivative_matrix[5,4] = -x57
                derivative_matrix[5,5] = -x59
                derivative_matrix[5,6] = -x61
                derivative_matrix[5,7] = -x64
                derivative_matrix[5,8] = -x69
                derivative_matrix[5,9] = -x71

        result_list[0]=B[0,0]
        result_list[1]=B[0,1]
        result_list[2]=B[1,0]
        result_list[3]=B[1,1]
        result_list[4]=Omega[0,1]
        result_list[5]=Omega[1,0]


    # Do the calculations in C, including the derivatives
    def generate_c_code(self) -> str:
        return """
const double R00=arg_list[0];
const double R01=arg_list[1];
const double R10=arg_list[2];
const double R11=arg_list[3];
const double gradU00=arg_list[4];
const double gradU01=arg_list[5];
const double gradU10=arg_list[6];
const double gradU11=arg_list[7];
const double ev0=arg_list[8];
const double ev1=arg_list[9];

const unsigned nargs_fixed=10;
const unsigned nret_fixed=6;

double B00,B01,B10,B11,Omega01,Omega10;

const double divisor=ev1-ev0;
if (fabs(divisor)<"""+str(self.epsilon)+""")
{
   B00=gradU00;
   B10=B01=gradU01/2.0 + gradU10/2.0;   
   B11=gradU11;
   Omega01=Omega10=0.0;
   if (flag)
   {
       for (unsigned int i=0;i<nargs_fixed*nret;i++) derivative_matrix[i]=0.0;
       derivative_matrix[0*nargs_fixed+4] = derivative_matrix[3*nargs_fixed+7] =1.0;
       derivative_matrix[1*nargs_fixed+5] = derivative_matrix[1*nargs_fixed+6] = derivative_matrix[2*nargs_fixed+5] = derivative_matrix[2*nargs_fixed+6] =1.0/2.0;           
   }
}
else
{
   const double _temp1=R00*gradU01 + R10*gradU11;
   const double _temp2=R01*gradU00 + R11*gradU10;
   const double _temp3=R01*gradU01 + R11*gradU11;
   const double _temp4=R00*gradU00 + R10*gradU10;
	const double m00=R00*_temp4 + R10*_temp1;
	const double m01=R01*_temp4 + R11*_temp1;
	const double m10=R00*_temp2 + R10*_temp3;
	const double m11=R01*_temp2 + R11*_temp3;
   double omega_val=(ev1*m01+ev0*m10)/divisor;
   B00=R00*R00*(R00*_temp4 + R10*_temp1) + R01*R01*(R01*_temp2 + R11*_temp3);
	B01=R00*R10*(R00*_temp4 + R10*_temp1) + R01*R11*(R01*_temp2 + R11*_temp3);
	B10=R00*R10*(R00*_temp4 + R10*_temp1) + R01*R11*(R01*_temp2 + R11*_temp3);
	B11=R10*R10*(R00*_temp4 + R10*_temp1) + R11*R11*(R01*_temp2 + R11*_temp3);

	Omega01= R00*R11*(ev0*(R00*_temp2 + R10*_temp3) + ev1*(R01*_temp4 + R11*_temp1))/(-ev0 + ev1) - R01*R10*(ev0*(R00*_temp2 + R10*_temp3) + ev1*(R01*_temp4 + R11*_temp1))/(-ev0 + ev1);
	Omega10= -R00*R11*(ev0*(R00*_temp2 + R10*_temp3) + ev1*(R01*_temp4 + R11*_temp1))/(-ev0 + ev1) + R01*R10*(ev0*(R00*_temp2 + R10*_temp3) + ev1*(R01*_temp4 + R11*_temp1))/(-ev0 + ev1);

   if (flag)
   {
       const double x0 = R00*R00;
       const double x1 = R10*gradU01;
       const double x2 = R10*gradU10;
       const double x3 = R00*gradU00;
       const double x4 = x1 + x2 + 2*x3;
       const double x5 = x2 + x3;
       const double x6 = R00*gradU01;
       const double x7 = R10*gradU11;
       const double x8 = x6 + x7;
       const double x9 = R00*x5 + R10*x8;
       const double x10 = R00*x9;
       const double x11 = R01*R01;
       const double x12 = R11*gradU01;
       const double x13 = R11*gradU10;
       const double x14 = R01*gradU00;
       const double x15 = x12 + x13 + 2*x14;
       const double x16 = x13 + x14;
       const double x17 = R01*gradU01;
       const double x18 = R11*gradU11;
       const double x19 = x17 + x18;
       const double x20 = R01*x16 + R11*x19;
       const double x21 = R01*x20;
       const double x22 = R00*gradU10;
       const double x23 = x22 + x6 + 2*x7;
       const double x24 = R01*gradU10;
       const double x25 = x17 + 2*x18 + x24;
       const double x26 = pow(R00,3)*R10 + pow(R01,3)*R11;
       const double x27 = R10*R10;
       const double x28 = R11*R11;
       const double x29 = x0*x27 + x11*x28;
       const double x30 = R00*R10;
       const double x31 = R10*x9;
       const double x32 = x30*x4 + x31;
       const double x33 = R01*R11;
       const double x34 = R11*x20;
       const double x35 = x15*x33 + x34;
       const double x36 = x10 + x23*x30;
       const double x37 = x21 + x25*x33;
       const double x38 = R00*pow(R10,3) + R01*pow(R11,3);
       const double x39 = R00*R11;
       const double x40 = -ev0 + ev1;
       const double x41 = 1.0/x40;
       const double x42 = x41*(ev0*x16 + ev1*(x12 + x14));
       const double x43 = R01*R10;
       const double x44 = R00*x16 + R10*x19;
       const double x45 = R01*x5 + R11*x8;
       const double x46 = ev0*x44 + ev1*x45;
       const double x47 = x41*x46;
       const double x48 = R11*x47 + x39*x42 - x42*x43;
       const double x49 = ev0*(x1 + x3) + ev1*x5;
       const double x50 = -R00*R11*x41*x49 + R10*x47 + x41*x43*x49;
       const double x51 = ev0*x19 + ev1*(x18 + x24);
       const double x52 = -R00*R11*x41*x51 + R01*x47 + x41*x43*x51;
       const double x53 = x41*(ev0*(x22 + x7) + ev1*x8);
       const double x54 = R00*x47 + x39*x53 - x43*x53;
       const double x55 = R00*R01;
       const double x56 = x41*(ev0*x55 + ev1*x55);
       const double x57 = x39*x56 - x43*x56;
       const double x58 = x41*(ev0*x43 + ev1*x39);
       const double x59 = x39*x58 - x43*x58;
       const double x60 = x41*(ev0*x39 + ev1*x43);
       const double x61 = x39*x60 - x43*x60;
       const double x62 = R10*R11;
       const double x63 = x41*(ev0*x62 + ev1*x62);
       const double x64 = x39*x63 - x43*x63;
       const double x65 = x41*x44;
       const double x66 = x46/pow(x40,2);
       const double x67 = x39*x66;
       const double x68 = x43*x66;
       const double x69 = x39*x65 - x43*x65 + x67 - x68;
       const double x70 = x41*x45;
       const double x71 = x39*x70 - x43*x70 - x67 + x68;
       
       //Jacobian entries:       
       derivative_matrix[0*nargs_fixed+0] = x0*x4 + 2*x10;
       derivative_matrix[0*nargs_fixed+1] = x11*x15 + 2*x21;
       derivative_matrix[0*nargs_fixed+2] = x0*x23;
       derivative_matrix[0*nargs_fixed+3] = x11*x25;
       derivative_matrix[0*nargs_fixed+4] = pow(R00,4) + pow(R01,4);
       derivative_matrix[0*nargs_fixed+5] = x26;
       derivative_matrix[0*nargs_fixed+6] = x26;
       derivative_matrix[0*nargs_fixed+7] = x29;
       derivative_matrix[0*nargs_fixed+8] = 0.0;
       derivative_matrix[0*nargs_fixed+9] = 0.0;
       derivative_matrix[1*nargs_fixed+0] = x32;
       derivative_matrix[1*nargs_fixed+1] = x35;
       derivative_matrix[1*nargs_fixed+2] = x36;
       derivative_matrix[1*nargs_fixed+3] = x37;
       derivative_matrix[1*nargs_fixed+4] = x26;
       derivative_matrix[1*nargs_fixed+5] = x29;
       derivative_matrix[1*nargs_fixed+6] = x29;
       derivative_matrix[1*nargs_fixed+7] = x38;
       derivative_matrix[1*nargs_fixed+8] = 0.0;
       derivative_matrix[1*nargs_fixed+9] = 0.0;
       derivative_matrix[2*nargs_fixed+0] = x32;
       derivative_matrix[2*nargs_fixed+1] = x35;
       derivative_matrix[2*nargs_fixed+2] = x36;
       derivative_matrix[2*nargs_fixed+3] = x37;
       derivative_matrix[2*nargs_fixed+4] = x26;
       derivative_matrix[2*nargs_fixed+5] = x29;
       derivative_matrix[2*nargs_fixed+6] = x29;
       derivative_matrix[2*nargs_fixed+7] = x38;
       derivative_matrix[2*nargs_fixed+8] = 0.0;
       derivative_matrix[2*nargs_fixed+9] = 0.0;
       derivative_matrix[3*nargs_fixed+0] = x27*x4;
       derivative_matrix[3*nargs_fixed+1] = x15*x28;
       derivative_matrix[3*nargs_fixed+2] = x23*x27 + 2*x31;
       derivative_matrix[3*nargs_fixed+3] = x25*x28 + 2*x34;
       derivative_matrix[3*nargs_fixed+4] = x29;
       derivative_matrix[3*nargs_fixed+5] = x38;
       derivative_matrix[3*nargs_fixed+6] = x38;
       derivative_matrix[3*nargs_fixed+7] = pow(R10,4) + pow(R11,4);
       derivative_matrix[3*nargs_fixed+8] = 0.0;
       derivative_matrix[3*nargs_fixed+9] = 0.0;
       derivative_matrix[4*nargs_fixed+0] = x48;
       derivative_matrix[4*nargs_fixed+1] = -x50;
       derivative_matrix[4*nargs_fixed+2] = -x52;
       derivative_matrix[4*nargs_fixed+3] = x54;
       derivative_matrix[4*nargs_fixed+4] = x57;
       derivative_matrix[4*nargs_fixed+5] = x59;
       derivative_matrix[4*nargs_fixed+6] = x61;
       derivative_matrix[4*nargs_fixed+7] = x64;
       derivative_matrix[4*nargs_fixed+8] = x69;
       derivative_matrix[4*nargs_fixed+9] = x71;
       derivative_matrix[5*nargs_fixed+0] = -x48;
       derivative_matrix[5*nargs_fixed+1] = x50;
       derivative_matrix[5*nargs_fixed+2] = x52;
       derivative_matrix[5*nargs_fixed+3] = -x54;
       derivative_matrix[5*nargs_fixed+4] = -x57;
       derivative_matrix[5*nargs_fixed+5] = -x59;
       derivative_matrix[5*nargs_fixed+6] = -x61;
       derivative_matrix[5*nargs_fixed+7] = -x64;
       derivative_matrix[5*nargs_fixed+8] = -x69;
       derivative_matrix[5*nargs_fixed+9] = -x71;
    }
}

result_list[0]=B00;
result_list[1]=B01;
result_list[2]=B10;
result_list[3]=B11;
result_list[4]=Omega01;
result_list[5]=Omega10;
"""

    # We expect 10 scales (4 x R, 4 x grad(u), 2 x ev) and return 6 scalars (4 x B, 2 x Omega)
    def get_num_returned_scalars(self, nargs: int) -> int:
        assert nargs==10
        return 6

    # Assemble back to a list
    def process_result_list_to_results(self, result_list: List["Expression"]) -> Tuple["ExpressionOrNum", ...]:
        se= (lambda x:subexpression(x)) if self.use_subexpression else (lambda x:x)
        B=se(matrix([[result_list[0],result_list[1]],[result_list[2],result_list[3]]]))
        Omega=se(matrix([[0,result_list[4]],[result_list[5],0]]))
        return B,Omega
    



# Expects the diagonalizing tensor R, the velocity gradient grad(u) and the diagonal matrix ev
# Only works in 2d Cartesian!
# Returns B tensor and Omega tensor with case distinguishment
# if the eigenvalues are degenerate, return B=1/2*sym(grad(u)), Omega=0
# Else perform the transformation of the paper
class LogConfTensorDecompositionAxisymmetric(CustomMultiReturnExpression):
    def __init__(self,epsilon=1e-7,use_FD:Union[bool,float]=False,use_subexpression:bool=True) -> None:
        super().__init__()
        self.epsilon=epsilon
        self.use_FD=use_FD
        self.FD_epsilon=1e-8
        if isinstance(self.use_FD,float):
            self.FD_epsilon=self.use_FD
        self.use_subexpression=use_subexpression
    
    # Take the args R, grad(u) and ev and assemple it to a list
    def process_args_to_scalar_list(self, *args: "ExpressionOrNum") -> List["ExpressionOrNum"]:
        R=args[0]
        gradu=args[1]
        ev=args[2]
        return [R[0,0],R[0,1],R[1,0],R[1,1],gradu[0,0],gradu[0,1],gradu[1,0],gradu[1,1],gradu[2,2],ev[0,0],ev[1,1]]
    
    # To the calculations in python, including the derivatives
    def eval(self, flag: int, arg_list: NPFloatArray, result_list: NPFloatArray, derivative_matrix: NPFloatArray) -> None:
        R=numpy.array([[arg_list[0],arg_list[1],0],[arg_list[2],arg_list[3],0],[0,0,1]])
        gradU=numpy.array([[arg_list[4],arg_list[5],0],[arg_list[6],arg_list[7],0],[0,0,arg_list[8]]])
        ev0=arg_list[9]
        ev1=arg_list[10]       
        divisor=ev1-ev0
        if abs(divisor)<self.epsilon:
            B=(gradU+numpy.transpose(gradU))/2
            Omega=numpy.zeros((3,3),dtype=numpy.float64)
            if flag and not self.use_FD:
                derivative_matrix.fill(0.0)
                derivative_matrix[0,4] = 1.0
                derivative_matrix[1,5] = 1.0/2.0
                derivative_matrix[1,6] = 1.0/2.0
                derivative_matrix[2,5] = 1.0/2.0
                derivative_matrix[2,6] = 1.0/2.0
                derivative_matrix[3,7] = 1.0
                derivative_matrix[4,8] = 1.0
        else:
            m=numpy.matmul(numpy.matmul(numpy.transpose(R),gradU),R)
            omega_val=(ev1*m[0,1]+ev0*m[1,0])/divisor
            B=numpy.matmul(numpy.matmul(R,numpy.array([[m[0,0],0,0],[0,m[1,1],0],[0,0,gradU[2,2]]])),numpy.transpose(R))   
            Omega=numpy.matmul(numpy.matmul(R,numpy.array([[0,omega_val,0],[-omega_val,0,0],[0,0,0]])),numpy.transpose(R))      
            if flag and not self.use_FD:
                x0 = R[1,0]**2
                x1 = gradU[1,1]*x0
                x2 = R[0,0]**2
                x3 = gradU[0,0]*x2
                x4 = R[0,0]*gradU[0,1]
                x5 = 3*R[1,0]
                x6 = R[0,0]*gradU[1,0]
                x7 = x4*x5 + x5*x6
                x8 = R[1,1]**2
                x9 = gradU[1,1]*x8
                x10 = R[0,1]**2
                x11 = gradU[0,0]*x10
                x12 = R[0,1]*gradU[0,1]
                x13 = 3*R[1,1]
                x14 = R[0,1]*gradU[1,0]
                x15 = x12*x13 + x13*x14
                x16 = R[1,0]*gradU[1,1]
                x17 = R[1,1]*gradU[1,1]
                x18 = R[0,0]**3*R[1,0] + R[0,1]**3*R[1,1]
                x19 = x0*x2 + x10*x8
                x20 = 2*R[1,0]
                x21 = x20*x4 + x20*x6
                x22 = R[1,0]*(x1 + x21 + 3*x3)
                x23 = 2*R[1,1]
                x24 = x12*x23 + x14*x23
                x25 = R[1,1]*(3*x11 + x24 + x9)
                x26 = R[0,0]*(3*x1 + x21 + x3)
                x27 = R[0,1]*(x11 + x24 + 3*x9)
                x28 = R[0,0]*R[1,0]**3 + R[0,1]*R[1,1]**3
                x29 = R[1,0]*gradU[0,1]
                x30 = R[1,0]*gradU[1,0]
                x31 = R[0,0]*gradU[0,0]
                x32 = R[1,1]*gradU[0,1]
                x33 = R[1,1]*gradU[1,0]
                x34 = R[0,1]*gradU[0,0]
                x35 = ev0 - ev1
                x36 = 1/x35
                x37 = x33 + x34
                x38 = ev0*x37 + ev1*(x32 + x34)
                x39 = R[0,0]*R[1,1]
                x40 = x12 + x17
                x41 = R[0,0]*x37 + R[1,0]*x40
                x42 = x30 + x31
                x43 = x16 + x4
                x44 = R[0,1]*x42 + R[1,1]*x43
                x45 = ev0*x41 + ev1*x44
                x46 = -R[0,1]*R[1,0]*x38 + R[1,1]*x45 + x38*x39
                x47 = ev0*(x29 + x31) + ev1*x42
                x48 = R[0,1]*R[1,0]
                x49 = R[1,0]*x45 - x39*x47 + x47*x48
                x50 = ev0*x40 + ev1*(x14 + x17)
                x51 = R[0,1]*x45 - x39*x50 + x48*x50
                x52 = ev0*(x16 + x6) + ev1*x43
                x53 = R[0,0]*x45 - R[0,1]*R[1,0]*x52 + x39*x52
                x54 = ev0 + ev1
                x55 = -R[0,1]*R[1,0] + x39
                x56 = -x55
                x57 = x36*x56
                x58 = x54*x57
                x59 = R[0,0]*R[0,1]
                x60 = ev0*x48 + ev1*x39
                x61 = ev0*x39 + ev1*x48
                x62 = R[1,0]*R[1,1]
                x63 = x35**(-2)
                x64 = x35*x41
                x65 = x63*(x35*x44 + x45)
                x66 = x36*x55
                x67 = x54*x66
                
                #Jacobian entries:
                derivative_matrix.fill(0.0)
                derivative_matrix[0,0] = R[0,0]*(2*x1 + 4*x3 + x7)
                derivative_matrix[0,1] = R[0,1]*(4*x11 + x15 + 2*x9)
                derivative_matrix[0,2] = x2*(2*x16 + x4 + x6)
                derivative_matrix[0,3] = x10*(x12 + x14 + 2*x17)
                derivative_matrix[0,4] = R[0,0]**4 + R[0,1]**4
                derivative_matrix[0,5] = x18
                derivative_matrix[0,6] = x18
                derivative_matrix[0,7] = x19
                derivative_matrix[1,0] = x22
                derivative_matrix[1,1] = x25
                derivative_matrix[1,2] = x26
                derivative_matrix[1,3] = x27
                derivative_matrix[1,4] = x18
                derivative_matrix[1,5] = x19
                derivative_matrix[1,6] = x19
                derivative_matrix[1,7] = x28
                derivative_matrix[2,0] = x22
                derivative_matrix[2,1] = x25
                derivative_matrix[2,2] = x26
                derivative_matrix[2,3] = x27
                derivative_matrix[2,4] = x18
                derivative_matrix[2,5] = x19
                derivative_matrix[2,6] = x19
                derivative_matrix[2,7] = x28
                derivative_matrix[3,0] = x0*(x29 + x30 + 2*x31)
                derivative_matrix[3,1] = x8*(x32 + x33 + 2*x34)
                derivative_matrix[3,2] = R[1,0]*(4*x1 + 2*x3 + x7)
                derivative_matrix[3,3] = R[1,1]*(2*x11 + x15 + 4*x9)
                derivative_matrix[3,4] = x19
                derivative_matrix[3,5] = x28
                derivative_matrix[3,6] = x28
                derivative_matrix[3,7] = R[1,0]**4 + R[1,1]**4
                derivative_matrix[4,8] = 1.0
                derivative_matrix[5,0] = -x36*x46
                derivative_matrix[5,1] = x36*x49
                derivative_matrix[5,2] = x36*x51
                derivative_matrix[5,3] = -x36*x53
                derivative_matrix[5,4] = x58*x59
                derivative_matrix[5,5] = x57*x60
                derivative_matrix[5,6] = x57*x61
                derivative_matrix[5,7] = x58*x62
                derivative_matrix[5,9] = x63*(x45*x55 + x56*x64)
                derivative_matrix[5,10] = x56*x65
                derivative_matrix[6,0] = x36*x46
                derivative_matrix[6,1] = -x36*x49
                derivative_matrix[6,2] = -x36*x51
                derivative_matrix[6,3] = x36*x53
                derivative_matrix[6,4] = x59*x67
                derivative_matrix[6,5] = x60*x66
                derivative_matrix[6,6] = x61*x66
                derivative_matrix[6,7] = x62*x67
                derivative_matrix[6,9] = x63*(x45*x56 + x55*x64)
                derivative_matrix[6,10] = x55*x65


        result_list[0]=B[0,0]
        result_list[1]=B[0,1]
        result_list[2]=B[1,0]
        result_list[3]=B[1,1]
        result_list[4]=B[2,2]
        result_list[5]=Omega[0,1]
        result_list[6]=Omega[1,0]

        if flag and self.use_FD:
            self.fill_python_derivatives_by_FD(arg_list,result_list,derivative_matrix,fd_epsilion=self.FD_epsilon)


    # Do the calculations in C, including the derivatives
    def generate_c_code(self) -> str:
        code= """
const double R00=arg_list[0];
const double R01=arg_list[1];
const double R10=arg_list[2];
const double R11=arg_list[3];
const double gradU00=arg_list[4];
const double gradU01=arg_list[5];
const double gradU10=arg_list[6];
const double gradU11=arg_list[7];
const double gradU22=arg_list[8];
const double ev0=arg_list[9];
const double ev1=arg_list[10];

const unsigned nargs_fixed=11;
const unsigned nret_fixed=7;

double B00,B01,B10,B11,B22,Omega01,Omega10;
B22=gradU22;

const double divisor=ev1-ev0;
if (fabs(divisor)<"""+str(self.epsilon)+""")
{
   B00=gradU00;
   B10=B01=gradU01/2.0 + gradU10/2.0;   
   B11=gradU11;
   Omega01=Omega10=0.0;
"""
        if not self.use_FD:
            code+="""   if (flag)
   {
       for (unsigned int i=0;i<nargs_fixed*nret_fixed;i++) derivative_matrix[i]=0.0;
       derivative_matrix[0*nargs_fixed+4] = derivative_matrix[3*nargs_fixed+7] = derivative_matrix[4*nargs_fixed+8] = 1.0;
       derivative_matrix[1*nargs_fixed+5] = derivative_matrix[1*nargs_fixed+6] = derivative_matrix[2*nargs_fixed+5] = derivative_matrix[2*nargs_fixed+6] =1.0/2.0;           
   }
"""
        code+="""
}
else
{
   const double _temp1=R00*gradU01 + R10*gradU11;
   const double _temp2=R01*gradU00 + R11*gradU10;
   const double _temp3=R01*gradU01 + R11*gradU11;
   const double _temp4=R00*gradU00 + R10*gradU10;
	const double m00=R00*_temp4 + R10*_temp1;
	const double m01=R01*_temp4 + R11*_temp1;
	const double m10=R00*_temp2 + R10*_temp3;
	const double m11=R01*_temp2 + R11*_temp3;
   double omega_val=(ev1*m01+ev0*m10)/divisor;
   B00=R00*R00*(R00*_temp4 + R10*_temp1) + R01*R01*(R01*_temp2 + R11*_temp3);
	B01=R00*R10*(R00*_temp4 + R10*_temp1) + R01*R11*(R01*_temp2 + R11*_temp3);
	B10=R00*R10*(R00*_temp4 + R10*_temp1) + R01*R11*(R01*_temp2 + R11*_temp3);
	B11=R10*R10*(R00*_temp4 + R10*_temp1) + R11*R11*(R01*_temp2 + R11*_temp3);

	Omega01= R00*R11*(ev0*(R00*_temp2 + R10*_temp3) + ev1*(R01*_temp4 + R11*_temp1))/(-ev0 + ev1) - R01*R10*(ev0*(R00*_temp2 + R10*_temp3) + ev1*(R01*_temp4 + R11*_temp1))/(-ev0 + ev1);
	Omega10= -R00*R11*(ev0*(R00*_temp2 + R10*_temp3) + ev1*(R01*_temp4 + R11*_temp1))/(-ev0 + ev1) + R01*R10*(ev0*(R00*_temp2 + R10*_temp3) + ev1*(R01*_temp4 + R11*_temp1))/(-ev0 + ev1);
"""
        if not self.use_FD:
            code+="""   if (flag)
   {
       const double x0 = pow(R10, 2);
        const double x1 = gradU11*x0;
        const double x2 = pow(R00, 2);
        const double x3 = gradU00*x2;
        const double x4 = R00*gradU01;
        const double x5 = 3*R10;
        const double x6 = R00*gradU10;
        const double x7 = x4*x5 + x5*x6;
        const double x8 = pow(R11, 2);
        const double x9 = gradU11*x8;
        const double x10 = pow(R01, 2);
        const double x11 = gradU00*x10;
        const double x12 = R01*gradU01;
        const double x13 = 3*R11;
        const double x14 = R01*gradU10;
        const double x15 = x12*x13 + x13*x14;
        const double x16 = R10*gradU11;
        const double x17 = R11*gradU11;
        const double x18 = pow(R00, 3)*R10 + pow(R01, 3)*R11;
        const double x19 = x0*x2 + x10*x8;
        const double x20 = 2*R10;
        const double x21 = x20*x4 + x20*x6;
        const double x22 = R10*(x1 + x21 + 3*x3);
        const double x23 = 2*R11;
        const double x24 = x12*x23 + x14*x23;
        const double x25 = R11*(3*x11 + x24 + x9);
        const double x26 = R00*(3*x1 + x21 + x3);
        const double x27 = R01*(x11 + x24 + 3*x9);
        const double x28 = R00*pow(R10, 3) + R01*pow(R11, 3);
        const double x29 = R10*gradU01;
        const double x30 = R10*gradU10;
        const double x31 = R00*gradU00;
        const double x32 = R11*gradU01;
        const double x33 = R11*gradU10;
        const double x34 = R01*gradU00;
        const double x35 = ev0 - ev1;
        const double x36 = 1.0/x35;
        const double x37 = x33 + x34;
        const double x38 = ev0*x37 + ev1*(x32 + x34);
        const double x39 = R00*R11;
        const double x40 = x12 + x17;
        const double x41 = R00*x37 + R10*x40;
        const double x42 = x30 + x31;
        const double x43 = x16 + x4;
        const double x44 = R01*x42 + R11*x43;
        const double x45 = ev0*x41 + ev1*x44;
        const double x46 = -R01*R10*x38 + R11*x45 + x38*x39;
        const double x47 = ev0*(x29 + x31) + ev1*x42;
        const double x48 = R01*R10;
        const double x49 = R10*x45 - x39*x47 + x47*x48;
        const double x50 = ev0*x40 + ev1*(x14 + x17);
        const double x51 = R01*x45 - x39*x50 + x48*x50;
        const double x52 = ev0*(x16 + x6) + ev1*x43;
        const double x53 = R00*x45 - R01*R10*x52 + x39*x52;
        const double x54 = ev0 + ev1;
        const double x55 = -R01*R10 + x39;
        const double x56 = -x55;
        const double x57 = x36*x56;
        const double x58 = x54*x57;
        const double x59 = R00*R01;
        const double x60 = ev0*x48 + ev1*x39;
        const double x61 = ev0*x39 + ev1*x48;
        const double x62 = R10*R11;
        const double x63 = pow(x35, -2);
        const double x64 = x35*x41;
        const double x65 = x63*(x35*x44 + x45);
        const double x66 = x36*x55;
        const double x67 = x54*x66;
       
       //Jacobian entries:       
        derivative_matrix[0*nargs_fixed+0]=R00*(2*x1 + 4*x3 + x7);
        derivative_matrix[0*nargs_fixed+1]=R01*(4*x11 + x15 + 2*x9);
        derivative_matrix[0*nargs_fixed+2]=x2*(2*x16 + x4 + x6);
        derivative_matrix[0*nargs_fixed+3]=x10*(x12 + x14 + 2*x17);
        derivative_matrix[0*nargs_fixed+4]=pow(R00, 4) + pow(R01, 4);
        derivative_matrix[0*nargs_fixed+5]=x18;
        derivative_matrix[0*nargs_fixed+6]=x18;
        derivative_matrix[0*nargs_fixed+7]=x19;
        derivative_matrix[0*nargs_fixed+8]=0.0;
        derivative_matrix[0*nargs_fixed+9]=0.0;
        derivative_matrix[0*nargs_fixed+10]=0.0;
        derivative_matrix[1*nargs_fixed+0]=x22;
        derivative_matrix[1*nargs_fixed+1]=x25;
        derivative_matrix[1*nargs_fixed+2]=x26;
        derivative_matrix[1*nargs_fixed+3]=x27;
        derivative_matrix[1*nargs_fixed+4]=x18;
        derivative_matrix[1*nargs_fixed+5]=x19;
        derivative_matrix[1*nargs_fixed+6]=x19;
        derivative_matrix[1*nargs_fixed+7]=x28;
        derivative_matrix[1*nargs_fixed+8]=0.0;
        derivative_matrix[1*nargs_fixed+9]=0.0;
        derivative_matrix[1*nargs_fixed+10]=0.0;
        derivative_matrix[2*nargs_fixed+0]=x22;
        derivative_matrix[2*nargs_fixed+1]=x25;
        derivative_matrix[2*nargs_fixed+2]=x26;
        derivative_matrix[2*nargs_fixed+3]=x27;
        derivative_matrix[2*nargs_fixed+4]=x18;
        derivative_matrix[2*nargs_fixed+5]=x19;
        derivative_matrix[2*nargs_fixed+6]=x19;
        derivative_matrix[2*nargs_fixed+7]=x28;
        derivative_matrix[2*nargs_fixed+8]=0.0;
        derivative_matrix[2*nargs_fixed+9]=0.0;
        derivative_matrix[2*nargs_fixed+10]=0.0;
        derivative_matrix[3*nargs_fixed+0]=x0*(x29 + x30 + 2*x31);
        derivative_matrix[3*nargs_fixed+1]=x8*(x32 + x33 + 2*x34);
        derivative_matrix[3*nargs_fixed+2]=R10*(4*x1 + 2*x3 + x7);
        derivative_matrix[3*nargs_fixed+3]=R11*(2*x11 + x15 + 4*x9);
        derivative_matrix[3*nargs_fixed+4]=x19;
        derivative_matrix[3*nargs_fixed+5]=x28;
        derivative_matrix[3*nargs_fixed+6]=x28;
        derivative_matrix[3*nargs_fixed+7]=pow(R10, 4) + pow(R11, 4);
        derivative_matrix[3*nargs_fixed+8]=0.0;
        derivative_matrix[3*nargs_fixed+9]=0.0;
        derivative_matrix[3*nargs_fixed+10]=0.0;
        derivative_matrix[4*nargs_fixed+0]=0.0;
        derivative_matrix[4*nargs_fixed+1]=0.0;
        derivative_matrix[4*nargs_fixed+2]=0.0;
        derivative_matrix[4*nargs_fixed+3]=0.0;
        derivative_matrix[4*nargs_fixed+4]=0.0;
        derivative_matrix[4*nargs_fixed+5]=0.0;
        derivative_matrix[4*nargs_fixed+6]=0.0;
        derivative_matrix[4*nargs_fixed+7]=0.0;
        derivative_matrix[4*nargs_fixed+8]=1.0;
        derivative_matrix[4*nargs_fixed+9]=0.0;
        derivative_matrix[4*nargs_fixed+10]=0.0;
        derivative_matrix[5*nargs_fixed+0]=-x36*x46;
        derivative_matrix[5*nargs_fixed+1]=x36*x49;
        derivative_matrix[5*nargs_fixed+2]=x36*x51;
        derivative_matrix[5*nargs_fixed+3]=-x36*x53;
        derivative_matrix[5*nargs_fixed+4]=x58*x59;
        derivative_matrix[5*nargs_fixed+5]=x57*x60;
        derivative_matrix[5*nargs_fixed+6]=x57*x61;
        derivative_matrix[5*nargs_fixed+7]=x58*x62;
        derivative_matrix[5*nargs_fixed+8]=0.0;
        derivative_matrix[5*nargs_fixed+9]=x63*(x45*x55 + x56*x64);
        derivative_matrix[5*nargs_fixed+10]=x56*x65;
        derivative_matrix[6*nargs_fixed+0]=x36*x46;
        derivative_matrix[6*nargs_fixed+1]=-x36*x49;
        derivative_matrix[6*nargs_fixed+2]=-x36*x51;
        derivative_matrix[6*nargs_fixed+3]=x36*x53;
        derivative_matrix[6*nargs_fixed+4]=x59*x67;
        derivative_matrix[6*nargs_fixed+5]=x60*x66;
        derivative_matrix[6*nargs_fixed+6]=x61*x66;
        derivative_matrix[6*nargs_fixed+7]=x62*x67;
        derivative_matrix[6*nargs_fixed+8]=0.0;
        derivative_matrix[6*nargs_fixed+9]=x63*(x45*x56 + x55*x64);
        derivative_matrix[6*nargs_fixed+10]=x55*x65;
    }"""
        code+="""
}

result_list[0]=B00;
result_list[1]=B01;
result_list[2]=B10;
result_list[3]=B11;
result_list[4]=B22;
result_list[5]=Omega01;
result_list[6]=Omega10;
"""
        if self.use_FD:
            code+="""
FILL_MULTI_RET_JACOBIAN_BY_FD(1.0e-8)
"""
        return code

    # We expect 11 scales (4 x R, 5 x grad(u), 2 x ev) and return 7 scalars (5 x B, 2 x Omega)
    def get_num_returned_scalars(self, nargs: int) -> int:
        assert nargs==11
        return 7

    # Assemble back to a list
    def process_result_list_to_results(self, result_list: List["Expression"]) -> Tuple["ExpressionOrNum", ...]:
        se= (lambda x:subexpression(x)) if self.use_subexpression else (lambda x:x)
        B=se(matrix([[result_list[0],result_list[1],0],[result_list[2],result_list[3],0],[0,0,result_list[4]]]))
        Omega=se(matrix([[0,result_list[5],0],[result_list[6],0,0],[0,0,0]]))
        return B,Omega
    



class SymmetricMatrixExponential(CustomMultiReturnExpression):    
    def __init__(self,coordinate_system:BaseCoordinateSystem,dim:int,scale:Union[ExpressionOrNum,str]=1,fill_to_max_vector_dim:bool=True,use_FD:Union[bool,float]=False,use_subexpression:bool=True) -> None:
        super().__init__()
        if isinstance(coordinate_system,AxisymmetricCoordinateSystem):
            if isinstance(coordinate_system,AxisymmetryBreakingCoordinateSystem):
                raise RuntimeError("Not implemented for this coordinate system: "+str(coordinate_system))
            self.axisymmetric=True
        elif isinstance(coordinate_system,CartesianCoordinateSystem):
            self.axisymmetric=False
        else:
            raise RuntimeError("Not implemented for this coordinate system: "+str(coordinate_system))
    
        self.dim=dim
        if self.dim!=2:
            raise RuntimeError("Currently only implemented for 2 dimensional tensors")        
        if isinstance(scale,str):
            scale=scale_factor(scale)
        self.scale=scale
        self.fill_to_max_vector_dim=fill_to_max_vector_dim # Fill to 3x3 [Filled with 0] or keep it at dim x dim ?
        self.use_FD=use_FD
        self.FD_epsilon=1e-8
        if isinstance(self.use_FD,float):
            self.FD_epsilon=self.use_FD
        self.use_subexpression=use_subexpression

    # Input arguments, i.e. the tensor, to scalar list
    def process_args_to_scalar_list(self,*args: ExpressionOrNum)->List[ExpressionOrNum]:
        assert len(args)==1
        M=args[0]
        assert isinstance(M,Expression)
        if not self.axisymmetric:
            if self.dim==2:
                return [M[0,0]/self.scale,M[0,1]/self.scale,M[1,1]/self.scale] # Nondimensional relevant matrix entries 
            else:
                raise RuntimeError("TODO: "+str(self.dim)+"-dimensional case here")
        else:
            raise RuntimeError("TODO: Axisymmetric case here")
        

    # How many scalar values will be returned. You can also check the number of scalar input values here
    def get_num_returned_scalars(self,nargs:int)->int:
        if not self.axisymmetric:
            if self.dim==2:
                if nargs!=3:
                    raise RuntimeError("Expected 3 input arguments!")
                return 3
            else:
                raise RuntimeError("TODO: "+str(self.dim)+"-dimensional case here")
        else:
            raise RuntimeError("TODO: Axisymmetric case here")


    # Evaluate the function. Depending on the case, we will call a specific routine
    def eval(self,flag:int,arg_list:NPFloatArray,result_list:NPFloatArray,derivative_matrix:NPFloatArray):
        if not self.axisymmetric:
            if self.dim==2:
                self.eval_2d_cartesian(flag,arg_list,result_list,derivative_matrix) # Call the Python eval for 2d Cartesian
            else:
                raise RuntimeError("TODO: "+str(self.dim)+"-dimensional case here")
        else:
            raise RuntimeError("TODO: Axisymmetric case here")
        
    # 2d Cartesian case, evaluation and Jacobian
    def eval_2d_cartesian(self,flag:int,arg_list:NPFloatArray,result_list:NPFloatArray,derivative_matrix:NPFloatArray):

        a11=arg_list[0]
        a12=arg_list[1]
        a22=arg_list[2]

        mu_eps=0
        mu = math.sqrt((a11-a22)**2 + 4*a12**2)/2
        eApD2 = math.exp((a11+a22)/2)
        AmD2 = (a11 - a22)/2.
        coshMu = math.cosh(mu)
        if mu<=mu_eps:
            sinchMu=1
        else:
            sinchMu=math.sinh(mu)/mu
        result_list[0] = eApD2 * (coshMu + AmD2*sinchMu)
        result_list[1] = eApD2 * a12 * sinchMu
        result_list[2] = eApD2 * (coshMu - AmD2*sinchMu)

        
        if flag:
            if self.use_FD:
                self.fill_python_derivatives_by_FD(arg_list,result_list,derivative_matrix,self.FD_epsilon)
            else:
                if mu>mu_eps:
                    x0 = a12**2
                    x1 = 4*x0
                    x2 = a11 - a22
                    x3 = x1 + x2**2
                    x4 = math.sqrt(x3)
                    x5 = x4/2
                    x6 = math.cosh(x5)
                    x7 = 0.5*a11 - 0.5*a22
                    x8 = 1/x4
                    x9 = math.sinh(x5)
                    x10 = x8*x9
                    x11 = 2*x10
                    x12 = x11*x7
                    x13 = math.exp(a11/2 + a22/2)
                    x14 = x13/2
                    x15 = x14*(x12 + x6)
                    x16 = x10/2
                    x17 = 1.0*x10
                    x18 = x2*x7
                    x19 = x6/x3
                    x20 = -x2
                    x21 = x20*x7
                    x22 = x9/x3**(3/2)
                    x23 = 2*x22
                    x24 = x17 + x18*x19 + x21*x23
                    x25 = a12*x11
                    x26 = a12*x7
                    x27 = 8*x22
                    x28 = x26*x27
                    x29 = 4*x19*x26
                    x30 = -x17 + x18*x23 + x19*x21
                    x31 = a12*x13
                    x32 = x10*x31
                    x33 = x19*x31
                    x34 = x23*x31
                    x35 = x14*(-x12 + x6)
                    #Jacobian entries:
                    derivative_matrix[0,0] = x13*(x16*x2 + x24) + x15
                    derivative_matrix[0,1] = x13*(x25 - x28 + x29)
                    derivative_matrix[0,2] = x13*(x16*x20 + x30) + x15
                    derivative_matrix[1,0] = x2*x33 + x20*x34 + x32
                    derivative_matrix[1,1] = -x0*x13*x27 + x1*x13*x19 + x11*x13
                    derivative_matrix[1,2] = x2*x34 + x20*x33 + x32
                    derivative_matrix[2,0] = x13*(x2*x8*x9/2 - x24) + x35
                    derivative_matrix[2,1] = x13*(x25 + x28 - x29)
                    derivative_matrix[2,2] = x13*(x20*x8*x9/2 - x30) + x35
                else:
                    x0 = 0.5*a11
                    x1 = 0.5*a22
                    x2 = a11 - a22
                    x3 = math.sqrt(4*a12**2 + x2**2)
                    x4 = x3/2
                    x5 = math.cosh(x4)
                    x6 = math.exp(a11/2 + a22/2)
                    x7 = x6/2
                    x8 = x7*(x0 - x1 + x5)
                    x9 = 1
                    x10 = x9/2
                    x11 = x10*x2
                    x12 = 2*a12*x6*x9
                    x13 = -x10*x2
                    x14 = a12*x7
                    x15 = x7*(-x0 + x1 + x5)
                    #Jacobian entries:
                    derivative_matrix[0,0] = x6*(x11 + 0.5) + x8
                    derivative_matrix[0,1] = x12
                    derivative_matrix[0,2] = x6*(x13 - 0.5) + x8
                    derivative_matrix[1,0] = x14
                    derivative_matrix[1,1] = x6
                    derivative_matrix[1,2] = x14
                    derivative_matrix[2,0] = x15 + x6*(x11 - 0.5)
                    derivative_matrix[2,1] = x12
                    derivative_matrix[2,2] = x15 + x6*(x13 + 0.5)

    def generate_c_code_2d_cartesian(self)->str:
        res= """
        const double a11=arg_list[0];
        const double a12=arg_list[1];
        const double a22=arg_list[2];
        const double mu = sqrt(pow(a11-a22,2) + 4*a12*a12)/2.;
        const double eApD2 = exp((a11+a22)/2.);
        const double AmD2 = (a11 - a22)/2.;
        const double coshMu = cosh(mu);
        double sinchMu;
        const double mu_eps=1e-9;
        if (mu<mu_eps)
        {
         sinchMu=1.0;
        }
        else
        {
         sinchMu=sinh(mu)/mu;
        }            
        result_list[0] = eApD2 * (coshMu + AmD2*sinchMu);
        result_list[1] = eApD2 * a12 * sinchMu;
        result_list[2] = eApD2 * (coshMu - AmD2*sinchMu);
        """
        if not self.use_FD:
            res+="""
        if (flag)
        {
            if (mu>mu_eps)
            {
                const double x0 = pow(a12,2);
                const double x1 = 4*x0;
                const double x2 = a11 - a22;
                const double x3 = x1 + pow(x2,2);
                const double x4 = sqrt(x3);
                const double x5 = x4/2.0;
                const double x6 = cosh(x5);
                const double x7 = 0.5*a11 - 0.5*a22;
                const double x8 = 1.0/x4;
                const double x9 = sinh(x5);
                const double x10 = x8*x9;
                const double x11 = 2*x10;
                const double x12 = x11*x7;
                const double x13 = exp(a11/2.0 + a22/2.0);
                const double x14 = x13/2.0;
                const double x15 = x14*(x12 + x6);
                const double x16 = x10/2.0;
                const double x17 = 1.0*x10;
                const double x18 = x2*x7;
                const double x19 = x6/x3;
                const double x20 = -x2;
                const double x21 = x20*x7;
                const double x22 = x9/pow(x3,3.0/2.0);
                const double x23 = 2*x22;
                const double x24 = x17 + x18*x19 + x21*x23;
                const double x25 = a12*x11;
                const double x26 = a12*x7;
                const double x27 = 8*x22;
                const double x28 = x26*x27;
                const double x29 = 4*x19*x26;
                const double x30 = -x17 + x18*x23 + x19*x21;
                const double x31 = a12*x13;
                const double x32 = x10*x31;
                const double x33 = x19*x31;
                const double x34 = x23*x31;
                const double x35 = x14*(-x12 + x6);
                derivative_matrix[0] = x13*(x16*x2 + x24) + x15;
                derivative_matrix[1] = x13*(x25 - x28 + x29);
                derivative_matrix[2] = x13*(x16*x20 + x30) + x15;
                derivative_matrix[3] = x2*x33 + x20*x34 + x32;
                derivative_matrix[4] = -x0*x13*x27 + x1*x13*x19 + x11*x13;
                derivative_matrix[5] = x2*x34 + x20*x33 + x32;
                derivative_matrix[6] = x13*(x2*x8*x9/2 - x24) + x35;
                derivative_matrix[7] = x13*(x25 + x28 - x29);
                derivative_matrix[8] = x13*(x20*x8*x9/2 - x30) + x35;
            }
            else
            {                
                const double x0 = 0.5*a11;
                const double x1 = 0.5*a22;
                const double x2 = a11 - a22;
                const double x3 = sqrt(4*a12*a12 + x2*x2);
                const double x4 = x3/2.0;
                const double x5 = cosh(x4);
                const double x6 = exp(a11/2.0 + a22/2.0);
                const double x7 = x6/2.0;
                const double x8 = x7*(x0 - x1 + x5);
                const double x9 = 1;
                const double x10 = x9/2.0;
                const double x11 = x10*x2;
                const double x12 = 2*a12*x6*x9;
                const double x13 = -x10*x2;
                const double x14 = a12*x7;
                const double x15 = x7*(-x0 + x1 + x5);
                derivative_matrix[0] = x6*(x11 + 0.5) + x8;
                derivative_matrix[1] = x12;
                derivative_matrix[2] = x6*(x13 - 0.5) + x8;
                derivative_matrix[3] = x14;
                derivative_matrix[4] = x6;
                derivative_matrix[5] = x14;
                derivative_matrix[6] = x15 + x6*(x11 - 0.5);
                derivative_matrix[7] = x12;
                derivative_matrix[8] = x15 + x6*(x13 + 0.5);
            }
        }
        """
        else:
            res+="""
        FILL_MULTI_RET_JACOBIAN_BY_FD("""+str(self.FD_epsilon)+""")
        """
        return res

        
    # Get the C code
    def generate_c_code(self) -> str:
        if not self.axisymmetric:
            if self.dim==2:
                return self.generate_c_code_2d_cartesian() # Generate the C code for 2d Cartesian
            else:
                raise RuntimeError("TODO: "+str(self.dim)+"-dimensional case here")
        else:
            raise RuntimeError("TODO: Axisymmetric case here")

    def get_num_returned_scalars(self, nargs: int) -> int:
        assert nargs==3
        return 3

    # Assemble back to a list
    def process_result_list_to_results(self, result_list: List["Expression"]) -> Tuple["ExpressionOrNum", ...]:
        se= (lambda x:subexpression(x)) if self.use_subexpression else (lambda x:x)
        if not self.axisymmetric:
            if self.dim==2:
                return se(matrix([[result_list[0],result_list[1]],[result_list[1],result_list[2]]],fill_to_max_vector_dim=self.fill_to_max_vector_dim))
            else:
                raise RuntimeError("TODO: "+str(self.dim)+"-dimensional case here")
        else:
            raise RuntimeError("TODO: Axisymmetric case here")

    