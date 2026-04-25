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
 

import numpy

from ..expressions.generic import ExpressionOrNum, Expression, scale_factor
from ..typings import *
import scipy.interpolate #type:ignore
from .cb import CustomMathExpression,CustomMultiReturnExpression



class DeterministicRandomField(CustomMathExpression):
  """
  Creates a random field in 1d, 2d or 3d. The field is created by a random cloud of points, which is interpolated using scipy's RegularGridInterpolator. The field is then evaluated at the given point.  Ranges must be set via min_x=[xmin,<ymin>,<zmin>] and same for max_x.

  Args:
    min_x: Minimum values of the field in each dimension. If a single value is given, it is assumed that the field is 1D.
    max_x: Maximum values of the field in each dimension. If a single value is given, it is assumed that the field is 1D.
    amplitude: Amplitude of the random field.
    Nresolution: Number of points in each dimension of the random cloud.
    interpolation: Interpolation method. Default is "linear".    
  """
  def __init__(self ,min_x:Union[float,List[float]]=[0] ,max_x:Union[float,List[float]]=[1] ,amplitude:float=1.0 ,Nresolution:int=100,interpolation:str="linear"):
    super(DeterministicRandomField, self).__init__()
    self.min_x = min_x
    self.max_x= max_x
    if not isinstance(self.min_x ,(tuple ,list,numpy.ndarray)):
      self.min_x =[self.min_x]
    if not isinstance(self.max_x ,(tuple ,list,numpy.ndarray)):
      self.max_x =[self.max_x]
    if len(self.min_x) !=len(self.max_x):
      raise RuntimeError("Non-matching min and max dimensions")
    self.amplitude =amplitude
    self.Nresolution =Nresolution
    if not isinstance(self.Nresolution ,(tuple ,list,numpy.ndarray)):
      self.Nresolution =[self.Nresolution ] *len(self.min_x)
    if len(self.Nresolution ) != len(self.min_x):
      raise RuntimeError("Non-matching Nresolution array")
    random_cloud =(numpy.random.rand(*self.Nresolution ) -0.5 ) *self.amplitude
    coords =[] #type:ignore
    for direct in range(len(self.Nresolution)):
      coords.append(numpy.linspace(0 ,1 ,num=self.Nresolution[direct])) #type:ignore
    self.interp =scipy.interpolate.RegularGridInterpolator(tuple(coords) ,random_cloud ,bounds_error=False,fill_value=0,method=interpolation) #type:ignore
    self.min_x =numpy.array(list(map(float,self.min_x))) #type:ignore
    self.coord_denom =numpy.array(list(map(float,[ 1 /(self.max_x[i ] -self.min_x[i]) for i in range(len(self.min_x))]))) #type:ignore

  def eval(self ,arg_array:NPFloatArray)->float:
    return self.interp((arg_array -self.min_x ) *self.coord_denom)[0] #type:ignore
		
		
		


# After creation, a call of this object will return the following:
# Let A and C be a list of n entries
# Let B and eps be scalars
# A call with arguments A,B,eps,C will return A[:]/B if |B|>=eps
# Otherwise, it will return C[:]
# For dimensional problems, please define Ascale(=Cscale) and Bscale. epsilon will be always nondimensional!
class MultiSafeDivide(CustomMultiReturnExpression):
    def __init__(self,Ascale:Union[str,ExpressionOrNum]=1,Bscale:Union[str,ExpressionOrNum]=1) -> None:
          super().__init__()
          if isinstance(Ascale,str):
                Ascale=scale_factor(Ascale)
          if isinstance(Bscale,str):
                Bscale=scale_factor(Bscale)
          self.Ascale=Ascale
          self.Bscale=Bscale

    def check_arg_num(self,nargs):
        if nargs<4:
            raise RuntimeError("Requires at least four arguments: A,B,epsilon,C to calculate A/B if |B|>=epsilon, otherwise return C")
        if (nargs-2)%2!=0:
            raise RuntimeError("Unsupported number of arguments, expected: A[n],B,epsilon,C[n] to calculate A[:]/B if |B|>=epsilon, otherwise return C[:]")          

    # Before calling eval, we can decompose our arguments. E.g. tensors split into scalars. The returning list may not have any phyiscal dimensions
    def process_args_to_scalar_list(self, *args: "ExpressionOrNum") -> List["ExpressionOrNum"]:
        nargs=len(args)
        nret=self.get_num_returned_scalars(nargs)
        res=[]
        for i,arg in enumerate(args):
              if i<nret:
                    res.append(arg/self.Ascale) #A
              elif i==nret:
                    res.append(arg/self.Bscale) #B
              elif i==nret+1:
                    res.append(arg) #epsilon
              else:
                    res.append(arg/self.Ascale) #C (scales as A)
        return res

    # Before returning, we can assemble things back to e.g. tensors or multiple returnals
    def process_result_list_to_results(self, result_list: List["Expression"]) -> Tuple["ExpressionOrNum", ...]:
        fact=self.Ascale/self.Bscale
        return tuple(r*fact for r in result_list)          
        
    def get_num_returned_scalars(self,nargs:int)->int:
        self.check_arg_num(nargs)
        return (nargs-2)//2

    def eval(self,flag:int,arg_list:NPFloatArray,result_list:NPFloatArray,derivative_matrix:NPFloatArray):
        nret=len(result_list)
        Alist=arg_list[0:nret]
        B=arg_list[nret]
        eps=arg_list[nret+1]
        Clist=arg_list[nret+2:]
        assert len(Alist)==len(Clist)
        print(Alist,B)
        if abs(B)<eps:            
            result_list[:]=Clist[:]
            #print("Case small: "+str(B)+"<"+str(eps))
            if flag:
                derivative_matrix[:,:nret+2]=numpy.zeros((nret,nret+2),dtype=numpy.float64)[:,:]
                derivative_matrix[:,nret+2:]=numpy.identity(nret,dtype=numpy.float64)[:,:]
        else:
            #print("Case large: "+str(B)+">="+str(eps))
            result_list[:]=Alist[:]/B
            if flag:
                derivative_matrix[:,0:nret]=numpy.identity(nret)/B
                derivative_matrix[:,nret]=-Alist[:]/B**2
                derivative_matrix[:,nret+1:]=numpy.zeros((nret,nret+1),dtype=numpy.float64)[:,:]

    
        #if flag:
        #    self.debug_python_derivatives_with_FD(arg_list,result_list,derivative_matrix,fd_epsilion=1e-9,error_threshold=1e-1,stop_on_error=True)


    def generate_c_code(self) -> str:
        return """
        // Code of MultiSafeDivide
        const double B=arg_list[nret];
        const double eps=arg_list[nret+1];
        if (fabs(B)<eps)
        {
          for (unsigned int i=0;i<nret;i++)
          {
           result_list[i]=arg_list[i+2+nret];
          }
          if (flag)
          {
            for (unsigned int i=0;i<nret;i++)
            {
               for (unsigned int j=0;j<nret+2;j++)
               {
                 derivative_matrix[i*nargs+j]=0.0;   
               }            
              for (unsigned int j=nret+2;j<nargs;j++)
               {
                 derivative_matrix[i*nargs+j]=(i+nret+2==j ? 1.0 : 0.0);   
               }                           
            }
          }
        }
        else
        {
          for (unsigned int i=0;i<nret;i++)        
          {
           result_list[i]=arg_list[i]/B;
          }
          if (flag)
          {
            for (unsigned int i=0;i<nret;i++)
            {
                for (unsigned int j=0;j<nargs;j++)
               {
                 derivative_matrix[i*nargs+j]=0.0;   
               }                                       
               derivative_matrix[i*nargs+i]=1.0/(B);   
               derivative_matrix[i*nargs+nret]=-arg_list[i]/(B*B);                            
            }
          }
        }
        
        """
