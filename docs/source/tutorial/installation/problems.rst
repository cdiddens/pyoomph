Typical problems with the installation
--------------------------------------

In the following, there are some typical errors listed, which may occur after installing and trying pyoomph:

.. container:: tcolorbox
	
	**distutils.errors.DistutilsPlatformError: Unable to find vcvarsall.bat**
			
	This happens on Windows if you have not installed MS Build Tools (cf. :numref:`secinstallationmsbuild`). Either install it or set the compiler to TinyC, which can be done by calling the method ``set_c_compiler("tcc")`` of your problem instance or by passing the command line argument *--tcc*.
   
   
