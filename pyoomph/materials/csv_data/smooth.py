import numpy 
from scipy.interpolate import UnivariateSpline

data=numpy.loadtxt("surftens_water_12hexanediol.csv")
def ign(x):
    global data
    closest_index=numpy.argmin(numpy.absolute(data[:,0]-x))
    print(closest_index)
    data=numpy.delete(data, closest_index,axis=0)
ign(0.019855)
samp=numpy.linspace(0,1,1001,endpoint=True)
smooth=UnivariateSpline(data[:,0],data[:,1],s=0.05)(samp)
numpy.savetxt("surftens_water_12hexanediol_smoothed.csv",numpy.array([samp,smooth]).transpose())
