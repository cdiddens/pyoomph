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
 
# This filter requires input written by MeshFileOutput(operator=MeshDataCombineWithEigenfunction(0))
# Probably best to use with tesselate_tri=True as well

# Load this file as filter in Paraview to use it

import vtk
import typing
import numpy
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk

from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa

# new module for ParaView-specific decorators.
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain


@smproxy.filter(label="Pyoomph Azimuthal Extrusion")
@smproperty.input(name="Input")
class PyoomphAzimuthalExtrusion(VTKPythonAlgorithmBase):
    def __init__(self):
        VTKPythonAlgorithmBase.__init__(self)
        self.m=1
        self.nextru=128
        self.angle_deg=360
        self.capping=1
        self.eigenfactor=0
        self.phi_shift=0

    @smproperty.intvector(name="Azimuthal Mode", default_values=1)
    @smdomain.intrange(min=0, max=20)    
    def SetM(self, m):
        self.m=m
        self.Modified()

    @smproperty.intvector(name="Resolution", default_values=128)
    @smdomain.intrange(min=2, max=500)    
    def SetNExtru(self, nextru):
        self.nextru=nextru
        self.Modified()        

    @smproperty.doublevector(name="Angle", default_values=360)
    @smdomain.doublerange(min=1, max=360)    
    def SetAngleDeg(self, angle_deg):
        self.angle_deg=angle_deg
        self.Modified()                

    @smproperty.doublevector(name="Angle Shift", default_values=0)
    @smdomain.doublerange(min=0, max=360)    
    def SetAngleShift(self, angle_shift):
        self.phi_shift=angle_shift/180*numpy.pi
        self.Modified()                

    @smproperty.doublevector(name="Eigen_perturbation", default_values=0)
    @smdomain.doublerange(min=0, max=1000)    
    def SetEigenFactor(self, eigenfactor):
        self.eigenfactor=eigenfactor
        self.Modified()                

    @smproperty.intvector(name="Capping", default_values=1)
    @smdomain.intrange(min=0, max=1)    
    def SetCapping(self, capping):
        self.capping=capping
        self.Modified()                        

    def RequestData(self, request, inInfo, outInfo):
        input0 = dsa.WrapDataObject(vtkDataSet.GetData(inInfo[0]))

        to_poly=vtk.vtkDataSetSurfaceFilter()        
        to_poly.SetInputDataObject(0,input0.VTKObject)
        to_poly.Update()

        extrusion=vtk.vtkRotationalExtrusionFilter()
        extrusion.SetInputDataObject(0,to_poly.GetOutputDataObject(0))
        extrusion.SetRotationAxis(0,1,0)
        extrusion.SetResolution(self.nextru)        
        extrusion.SetCapping(self.capping)
        extrusion.SetAngle(self.angle_deg)
        extrusion.Update()

        extr=extrusion.GetOutput()
        assert isinstance(extr,vtk.vtkPolyData)
        pd=extr.GetPointData()
        arr_to_index:typing.Dict[typing.AnyStr,int]={}
        for i in range(pd.GetNumberOfArrays()):
            arr_to_index[pd.GetArrayName(i)]=i

        pts=extr.GetPoints()
        assert isinstance(pts,vtk.vtkPoints)
        ptdata=pts.GetData()
        ptarray=vtk_to_numpy(ptdata)
        phi=numpy.arctan2(ptarray[:,2],ptarray[:,0])
        csphi=numpy.cos(phi)
        snphi=numpy.sin(phi)
        csmphi=numpy.cos(self.m*(phi-self.phi_shift))
        snmphi=numpy.sin(self.m*(phi-self.phi_shift))
        torem=[] # Remove fields
        toren={} # Rename fields: old->new
        toren_rev={} # Rename fields: new->old
        new_indices={}
        for n,i in arr_to_index.items():    
            if n.startswith("Eigen"):
                continue

            # Rotate also the base vector data correctly
            if n+"_phi" in arr_to_index.keys():
                arr=vtk_to_numpy(pd.GetArray(arr_to_index[n]))
                if len(arr.shape)>1:
                    Vr=arr[:,0]
                    Vy=arr[:,1]
                    Vphi=vtk_to_numpy(pd.GetArray(arr_to_index[n+"_phi"]))
                    Vx=csphi*Vr+snphi*Vphi
                    Vz=snphi*Vr-csphi*Vphi                    
                    base=numpy.array([Vx,Vy,Vz]).transpose()      
                    
                    newArr=numpy_to_vtk(base)    
                    newArr.SetName("__dummy_"+n)
                    torem.append(n)
                    torem.append(n+"_phi")  
                    dummyname="__dummy_"+n
                    toren[dummyname]=n
                    toren_rev[n]=dummyname
                    new_indices[dummyname]=pd.AddArray(newArr)            
            if not "EigenRe_"+n in arr_to_index.keys():
                continue
            if not "EigenIm_"+n in arr_to_index.keys():
                continue
            re=vtk_to_numpy(pd.GetArray(arr_to_index["EigenRe_"+n]))
            torem.append("EigenRe_"+n)
            im=vtk_to_numpy(pd.GetArray(arr_to_index["EigenIm_"+n]))
            torem.append("EigenIm_"+n)
            if len(re.shape)==1:  # Scalar data                
                arr=re*csmphi+im*snmphi        
            else:
                
                
                if not "EigenRe_"+n+"_phi" in arr_to_index.keys():
                    continue
                if not "EigenIm_"+n+"_phi" in arr_to_index.keys():
                    continue
                RePhi=vtk_to_numpy(pd.GetArray(arr_to_index["EigenRe_"+n+"_phi"]))
                ImPhi=vtk_to_numpy(pd.GetArray(arr_to_index["EigenIm_"+n+"_phi"]))
                torem.append("EigenRe_"+n+"_phi")
                torem.append("EigenIm_"+n+"_phi")
                torem.append("Eigen_"+n+"_phi")
                ReR=re[:,0]
                ImR=im[:,0]
                ReZ=re[:,1]
                ImZ=im[:,1]
                Vx=csphi*(csmphi*ReR+snmphi*ImR)+snphi*(csmphi*RePhi+snmphi*ImPhi)        
                Vz=snphi*(csmphi*ReR+snmphi*ImR)-csphi*(csmphi*RePhi+snmphi*ImPhi)        
                Vy=ReZ*csmphi+ImZ*snmphi        
                arr=numpy.array([Vx,Vy,Vz]).transpose()        
            if self.eigenfactor>0:
                if n in toren_rev:
                    base=vtk_to_numpy(pd.GetArray(new_indices[toren_rev[n]]))
                else:
                    base=vtk_to_numpy(pd.GetArray(arr_to_index[n]))
                arr=base+self.eigenfactor*arr
            newArr=numpy_to_vtk(arr)    
            newArr.SetName("Eigen_"+n)
            pd.AddArray(newArr)

        torem=list(set(torem))
        while len(torem)>0:
            n=torem[-1]
            names=[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
            pd.RemoveArray(names.index(n))
            torem.pop()

        for old,new in toren.items():            
            names=[pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
            if old in names:
                pd.GetArray(names.index(old)).SetName(new)


        extrusion.Update()
        self.GetOutputDataObject(0).DeepCopy(extrusion.GetOutputDataObject(0))
        # add to output
        #output = dsa.WrapDataObject(vtkDataSet.GetData(outInfo))
        #output.DeepCopy(extrusion)
        return 1






