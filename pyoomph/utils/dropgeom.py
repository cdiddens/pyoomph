#  @file
#  @author Christian Diddens <c.diddens@utwente.nl>
#  @author Duarte Rocha <d.rocha@utwente.nl>
#  
#  @section LICENSE
# 
#  pyoomph - a multi-physics finite element framework based on oomph-lib and GiNaC 
#  Copyright (C) 2021-2025  Christian Diddens & Duarte Rocha
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
 
from _pyoomph import Expression
import numpy
from ..output.meshio import IntegralObservableOutput
from ..typings import List
from ..expressions import square_root,pi,asin,sin,cos,absolute,rational_num,weak,dot,testfunction,scale_factor,div,grad,vector,acos,ExpressionNumOrNone,ExpressionOrNum,cartesian,Expression,CustomMathExpression,subexpression,log,is_zero,atan2
from ..expressions.interpol import InterpolateSpline1d
from ..expressions.units import meter,milli,newton,kilogram,second,degree
from .. import Equations,Problem,var,var_and_test,GlobalLagrangeMultiplier,WeakContribution,LineMesh,InitialCondition,TestScaling,Scaling,DirichletBC,IntegralObservables,TextFileOutput
from ..typings import *
from scipy import integrate


YoungLaplaceFixationEnum=Literal["contact_angle","volume","base_radius","apex_height"]
YoungLaplaceFixationsType=Union[Set[YoungLaplaceFixationEnum],Dict[YoungLaplaceFixationEnum,ExpressionOrNum]]

class DropletGeometry:
    """
    A helper class to calculate the geometry of a droplet from a set of parameters. You must specify exactly two of the following parameters:
    
    Args:
        volume: The volume of the droplet
        base_radius: The base radius (contact line radius) of the droplet
        contact_angle: The contact angle of the droplet
        apex_height: The apex height of the droplet
        curv_radius: The curvature radius of the droplet
        ambiguous_low_contact_angle: If you set base_radius and curv_radius, you must specify if the contact angle is below or above 90°
        evalf: If True, the result will be evaluated to a float. If False, the result will be kept as an expression
        rivulet_instead: If True, the droplet is assumed to be a rivulet, i.e. it is just a 2d cross section of a 3d droplet. The volume is then the area of this cross section.
    """
    def __init__(self,*,volume:ExpressionNumOrNone=None,base_radius:ExpressionNumOrNone=None,contact_angle:ExpressionNumOrNone=None,apex_height:ExpressionNumOrNone=None,curv_radius:ExpressionNumOrNone=None,ambiguous_low_contact_angle:Optional[bool]=None,evalf:bool=True,rivulet_instead:bool=False):
        #: The contact angle of the droplet
        self.contact_angle:ExpressionNumOrNone=None #type:ignore
        #: The volume of the droplet
        self.volume:ExpressionNumOrNone=None #type:ignore
        #: The apex height of the droplet
        self.apex_height:ExpressionNumOrNone=None #type:ignore
        #: The base radius of the droplet
        self.base_radius:ExpressionNumOrNone=None #type:ignore
        #: The curvature radius of the droplet
        self.curv_radius:ExpressionNumOrNone=None #type:ignore
        numgiven=0
        settings:Dict[str,ExpressionNumOrNone]= {}
        self._sampled_gravity_shape:Optional[Tuple[NPFloatArray,ExpressionOrNum]]=None
        self._sampled_gravity_shape_reference_pressure:ExpressionOrNum=None
        
        self.rivulet_instead=rivulet_instead

        def setprop(name:str,val:ExpressionNumOrNone)->int:
            settings[name]=val
            if val is not None:
                setattr(self,name,val)
                return 1
            else:
                return 0
            
        numgiven+=setprop("volume",volume)
        numgiven+=setprop("base_radius", base_radius)
        numgiven+=setprop("contact_angle", contact_angle)
        numgiven+=setprop("apex_height", apex_height)
        numgiven+=setprop("curv_radius", curv_radius)

        if numgiven!=2:
            raise RuntimeError("Specify exactly two of the following parameters: "+", ".join(settings.keys())+" but got: "+str(settings))
        if self.contact_angle is not None and evalf:
            if float(self.contact_angle)<0:
                raise RuntimeError("Negative contact angle!")
            elif float(self.contact_angle/pi)>=1:
                raise RuntimeError("Contact angle too large")

        self._settings=settings.copy()

        r0=self.base_radius
        h0=self.apex_height
        v0=self.volume
        if evalf:
            ca=float(self.contact_angle) if self.contact_angle is not None else None
        else:
            ca=self.contact_angle if self.contact_angle is not None else None
        rc=self.curv_radius

        if r0 is not None:
            if h0 is not None:
                pass
            elif v0 is not None:
                #print("v0")
                #h0 = 1.0 / pi * square_root((3.0 * v0 + square_root(pi ** 2 * r0 ** 6 + 9 * v0 ** 2) ) * pi ** 2,3) - r0 ** 2 * pi / square_root((3.0 * v0 + square_root(pi ** 2 * r0 ** 6 + 9 * v0 ** 2)) * pi ** 2,3)
                #print("H01",float(h0/meter))
                if self.rivulet_instead:
                    raise RuntimeError("Not yet implemented. Calculate the height of a rivulet from base radius and volume[area]")
                else:
                    h0 = 1.0 / pi * ((3 * v0 + (pi ** 2 * r0 ** 6 + 9 * v0 ** 2) ** rational_num(1, 2)) * pi ** 2) ** rational_num(1,3) - r0 ** 2 * pi / ((3 * v0 + (pi ** 2 * r0 ** 6 + 9 * v0 ** 2) ** rational_num(1, 2)) * pi ** 2) ** rational_num(1, 3)
                #print("H02",float(h0/meter))
            elif ca is not None:
                #if float(ca) <= float(0.5 * pi):
                    #h0 = (- cos(ca)+1) / absolute(sin(ca)) * r0 # TODO: I don't think there is any difference
                    h0 = (- cos(ca)+1) / (sin(ca)) * r0 # TODO: I don't think there is any difference
                #else:
                #    h0 = (1.0 + cos(pi - ca)) / absolute(sin(ca)) * r0 # TODO: I don't think there is any difference
            elif rc is not None:
                if ambiguous_low_contact_angle is None:
                    raise RuntimeError("Set ambiguous_low_contact_angle to either True or False if passing the base and curvature radius")
                if ambiguous_low_contact_angle:
                    h0 = rc - (rc ** 2 - r0 ** 2) ** (1.0 / 2.0)
                else:
                    h0 = rc + (rc ** 2 - r0 ** 2) ** (1.0 / 2.0)
            else:
                raise RuntimeError("base_radius > curv_radius")
        elif h0 is not None:
            if v0 is not None:
                if self.rivulet_instead:
                    raise RuntimeError("Not yet implemented. Calculate the base radius of a rivulet from height and volume[area]")
                else:
                    r0=square_root(3.0)*square_root((6*(v0/h0) - float(pi)*h0**2))/(3*square_root(float(pi)))
            elif ca is not None:
                if float(ca) <= float(pi/2):
                    r0 = h0 / ((1 - cos(ca)) / absolute(sin(ca)))
                else:
                    r0 = h0 / ((1 + cos(pi - ca)) / absolute(sin(ca)))
            elif rc is not None:
                r0 = square_root(rc ** 2 - (rc - h0) ** 2) 

        elif v0 is not None:            
            if ca is not None:
                if self.rivulet_instead:
                    h_by_c=(1-(cos(ca)))/(2*sin(ca))
                    theta=2*ca
                    a=v0                    
                    c=8*square_root(2)*square_root(a)*h_by_c*square_root(1/(16*h_by_c**4*theta + 32*h_by_c**3 + 8*h_by_c**2*theta - 8*h_by_c + theta))
                    r0=c/2
                    h0=h_by_c*c                                        
                else:
                    h0 = square_root(v0 * 3.0 / pi / (3.0 / (1.0 - cos(ca)) - 1.0),3)
                    r0 = square_root((v0 * 6.0 / pi / h0 - h0 * h0) / 3.0)
            elif rc is not None:
                raise RuntimeError("Not yet implemented. Calculations based on volume and curvature radius")
        elif ca is not None:
            if rc is not None:
                r0 = rc * sin(ca)
                h0 = rc * (1.0 - cos(ca))

        if (r0 is None) or (h0 is None):
            raise RuntimeError("Not sufficiently specified droplet geometry")

        if self.rivulet_instead:
            #raise RuntimeError("Check the result below!")
            #self.volume = ((r0**2+h0**2)**2*acos((r0**2-h0**2)/(r0**2+h0**2))-r0*(r0**2-h0**2))/(2*h0)
            c=2*r0
            self.volume=((c**2+4*h0**2)/(8*h0))**2*acos((c**2-4*h0**2)/(c**2+4*h0**2))-c/(16*h0)*(c**2-4*h0**2)
        else:
            self.volume = pi * h0 / 6.0 * (3.0 * r0 ** 2 + h0 ** 2) if self.volume is None else self.volume
        self.curv_radius = (r0 ** 2 + h0 ** 2) / (2.0 * h0) if self.curv_radius is None else self.curv_radius
        if self.contact_angle is None:
            if float(h0/r0) > 1:
                self.contact_angle = pi - asin((r0 / self.curv_radius)) if self.contact_angle is None else self.contact_angle
            else:
                self.contact_angle = asin((r0 / self.curv_radius)) if self.contact_angle is None else self.contact_angle
        self.base_radius=r0 if self.base_radius is None else self.base_radius
        self.apex_height = h0 if self.apex_height is None else self.apex_height

        if evalf:
            if isinstance(self.volume,Expression): #type:ignore
                self.volume=self.volume.evalf()
            if isinstance(self.base_radius,Expression): #type:ignore
                self.base_radius=self.base_radius.evalf()
            if isinstance(self.apex_height,Expression): #type:ignore
                self.apex_height=self.apex_height.evalf()
            if isinstance(self.contact_angle,Expression): #type:ignore
                self.contact_angle=self.contact_angle.evalf()
            if isinstance(self.curv_radius,Expression): #type:ignore
                self.curv_radius=self.curv_radius.evalf()

        #print(self.contact_angle)
        self.volume:ExpressionOrNum=self.volume
        self.base_radius:ExpressionOrNum=self.base_radius
        self.apex_height:ExpressionOrNum=self.apex_height
        self.contact_angle:ExpressionOrNum=self.contact_angle
        self.curv_radius:ExpressionOrNum=self.curv_radius
        self.surface_area:ExpressionOrNum=2*pi*self.curv_radius*self.apex_height

    @overload
    def get_point_at_interface_by_slerp(self,rel_apex_dist:float)-> List[ExpressionOrNum]: ...

    @overload
    def get_point_at_interface_by_slerp(self,rel_apex_dist:NPFloatArray)-> List[List[ExpressionOrNum]]: ...

    def get_point_at_interface_by_slerp(self,rel_apex_dist:Union[float,NPFloatArray])-> Union[List[List[ExpressionOrNum]],List[ExpressionOrNum]]:
        import scipy.spatial 
        start=numpy.array([0,float(self.apex_height/self.curv_radius)]) #type:ignore
        end = numpy.array([float(self.base_radius/self.curv_radius),0]) #type:ignore
        center = numpy.array([0,float((self.apex_height-self.curv_radius)/self.curv_radius)]) #type:ignore
        s=start-center #type:ignore
        e=end-center #type:ignore
        r=numpy.linalg.norm(s) #type:ignore
        s,e=s/r,e/r #type:ignore
        slerps=scipy.spatial.geometric_slerp(s,e,rel_apex_dist) #type:ignore
        if isinstance(rel_apex_dist,float):
            slerps=[slerps]
        res=[]
        for s in slerps:
            res.append(list((r*s+center)*self.curv_radius))
        if isinstance(rel_apex_dist,float):
            return res[0]
        else:
            return res


    # Relaxes the shape by gravity
    # returns an array of r and z positions and a scale factor to multiply the results with to get the right scaling
    def sample_gravity_shape(self,surface_tension:ExpressionOrNum,delta_rho_times_g:ExpressionOrNum,output_dir:str,fixations:Optional[YoungLaplaceFixationsType]=None,update_params:bool=True,N:int=200,output_text:bool=True,compiler:Any=None,ignore_command_line:bool=False,globally_convergent_newton:bool=False)->Tuple[NPFloatArray,ExpressionOrNum]:
        if self.rivulet_instead:
            raise RuntimeError("Not yet implemented")
        with YoungLaplaceDropletShape(self,sigma=surface_tension,rho_g_ez=delta_rho_times_g,fixations=fixations,N=N) as problem:
            problem.logfile_name=None # Do not change the log file here
            problem.set_output_directory(output_dir)
            problem.ignore_command_line=ignore_command_line
            if compiler is not None:
                problem.set_c_compiler(compiler)
            problem.relax_by_gravity(output_text=True,globally_convergent_newton=globally_convergent_newton)
            dom=problem.get_mesh("domain")
            rs:List[float]=[]
            zs:List[float]=[]
            for n in dom.nodes():
                rs.append(n.x(0))
                zs.append(n.x(1))
            spatscal=problem.get_scaling("spatial")
            assert spatscal is not None
            if update_params:
                self.apex_height=zs[0]*spatscal
                self.base_radius=rs[-1]*spatscal
                # curv_radius is not meaningful here
                self.volume=problem.get_mesh("domain").evaluate_observable("volume")
                self.contact_angle=problem.get_mesh("domain/right").evaluate_observable("contact_angle")
                self.surface_area=problem.get_mesh("domain").evaluate_observable("area")
            self._sampled_gravity_shape_reference_pressure=problem.get_ode("globals").get_value("p0")

        # Store it. You might require it some day...
        self._sampled_gravity_shape=cast(Tuple[NPFloatArray,ExpressionOrNum],(numpy.transpose(numpy.array([rs, zs])), spatscal)) # type:ignore
        
        return self._sampled_gravity_shape


    def get_sampled_gravity_shape(self):
        return self._sampled_gravity_shape # return [rs,zs] and a scale factor. If None, you must call sample_gravity_shape first



class DropletEvaporationHelper:
    # f_theta is angle factor that let you calculate the mass loss of a droplet by
    # dM/dt = -pi*R_base*D_vap*(c_interf-c_infty)*f(theta)
    # see e.g. Eq. (2.3), PhD thesis of Hanneke Gelderblom
    # f(theta=0)=4/pi holds for quite some theta range
    def generate_f_theta_function(self,numsamples:int=100):
        from scipy import integrate #type:ignore

        # integrant=lambda x : numpy.tanh(arg_prefactor*x)/(cosh(arg_prefactor*x)*numpy.sqrt(numpy.cosh(x)-numpy.cosh(tau)))
        def integrant(tau:NPFloatArray,theta:NPFloatArray):
            try:
                res = (1+numpy.cosh(2*theta * tau)) /numpy.sinh(2*numpy.pi*tau) * numpy.tanh((numpy.pi-theta)*tau)
            except:
                res = 0
            return res

        thetas:NPFloatArray=numpy.linspace(0,numpy.pi,numsamples,endpoint=False) #type:ignore
        f_thetas:List[List[float]]=[]
        for theta in thetas:
            add_term=numpy.sin(theta)/(1+numpy.cos(theta))
            integral = integrate.quad(lambda tau : integrant(tau,theta), 0, 30) #type:ignore
            f_thetas.append([theta,add_term+4*integral[0]])
        f_thetasA:NPFloatArray=numpy.array(f_thetas) #type:ignore

        from ..expressions.interpol import InterpolateSpline1d
        return InterpolateSpline1d(f_thetasA)



############ PROBLEM CLASS TO RELAX A DROPLET SHAPE BY GRAVITY #######################


# Solving the equilibrium shape of a droplet by moving nodes of a curved line mesh
# Normally, the nodes are moved so that the Young Laplace equation holds
# Tangentially, the nodes would be free to move, i.e. no unique solution possible
# Therefore, we move the nodes tangentially on the same fraction of the arc length as initially
class YoungLaplaceEquations(Equations):
    def __init__(self,p_ref:ExpressionOrNum,sigma:ExpressionOrNum,additional_pressure:ExpressionOrNum=0,blend_factor:ExpressionOrNum=1):
        super(YoungLaplaceEquations, self).__init__()
        self.sigma=sigma  # Surface tension
        self.p_ref=p_ref # reference pressure
        self.additional_pressure=additional_pressure # e.g. gravity
        self.blend_factor=blend_factor # Blend factor for the mesh dynamics

    def define_fields(self):
        self.define_vector_field("_norm","C2") # Projected normal => Required for the curvature=div(n)
        self.define_scalar_field("_curv","C2",scale=1/scale_factor("spatial"),testscale=scale_factor("spatial")) # curvature
        # arc length to shift the nodes tangentially in an equidistance fashion
        self.define_scalar_field("_s","C2",scale=scale_factor("spatial"),testscale=1/scale_factor("spatial"))
        self.activate_coordinates_as_dofs() # moving mesh
        # The mesh positions are solved by the Young-Laplace eq., which has a dimension of pressure
        # We must absorp this scale in the test function of the mesh
        self.set_test_scaling(mesh=1/scale_factor("pressure"))

    def define_residuals(self):
        real_n=var("normal")
        norm,norm_test=var_and_test("_norm")
        curv,curv_test=var_and_test("_curv")
        self.add_residual(weak(norm-real_n,norm_test,coordinate_system=cartesian)) # project normal, so that we can derive it

        self.add_residual(weak(curv-div(norm),curv_test)) # get curvature

        # In normal direction, we must make sure that sigma*curv+additional_pressure=p_ref
        _,xtest=var_and_test("mesh")
        self.add_residual(weak(self.sigma*curv+self.additional_pressure-self.p_ref,self.blend_factor*dot(real_n,xtest),coordinate_system=cartesian))

        # We solve the normalized arclength (Dirichlet _s=0 and _s=smax at boundaries required)
        s,stest=var_and_test("_s")
        self.add_residual(weak(grad(s,coordsys=cartesian,nondim=True),grad(stest,coordsys=cartesian,nondim=True),coordinate_system=cartesian))

        # And we shift the nodes tangentially so that they keep an equidistance arclength distance. Otherwise, they would be free to move tangentially
        sdest=var("lagrangian_x") # desired position is given by the initial arclength
        tang=vector(real_n[1],-real_n[0])
        self.add_residual(weak(s-sdest,self.blend_factor*dot(xtest,tang)*scale_factor("pressure")/scale_factor("spatial")))



# Problem to solve the equilibrium droplet shape with gravity
class YoungLaplaceDropletShape(Problem):
    def __init__(self,drop_geom:DropletGeometry,*,sigma:ExpressionOrNum=72 * milli * newton / meter,rho_g_ez:ExpressionOrNum=-9.81 * meter / second ** 2 * 1000 * kilogram / meter ** 3,fixations:Optional[YoungLaplaceFixationsType]=None,N:int=200):
        super(YoungLaplaceDropletShape, self).__init__()
        self.N=N
        self.drop_geom=drop_geom
        self.sigma = sigma
        self.rho_g_ez = rho_g_ez
        self.volume_factor=self.define_global_parameter(volume_factor=1) # To find the critical volume if desired

        # Fixations are the two parameters (base_radius, apex_height, volume, (microscopic) contact_angle) that are kept constant
        # if not explicitly set, it will take over the ones you set in the constructor if the DropletGeometry object passed to drop_geom
        if fixations is None:
            self.fixations=set(k for k,v in drop_geom._settings.items() if v is not None) #type:ignore
        else:
            self.fixations=fixations

        # To find the equilibirum shape, we will blend in the gravitational force by a parameter from 0 to 1
        self.force_factor = self.get_global_parameter("force_factor")



    def define_problem(self):
        # We must have exactly _TWO_ of the following constraints:
        #   fixed base radius
        #   fixed apex height
        #   fixed volume
        #   fixed _local_ contact angle

        # depending on the choice of these two, we get different scenarios
        #   fixed base radius & fixed volume => volume via p_ref, base radius pinned
        #   fixed base radius & fixed apex height => base radius pinned, apex height via p_ref
        #   fixed base radius & fixed contact angle => base radius pinned, contact angle via p_ref

        #   fixed apex height & fixed volume => volume via p_ref, apex height pinned
        #   fixed apex height & fixed contact angle => apex height via p_ref, contact angle via base radius

        #   fixed volume & fixed contact angle => volume via p_Ref, contact angle via base radius

        if len(self.fixations)!=2:
            raise RuntimeError("The set fixations must have exactly two fixed quantities out of: base_radius, apex_height, volume, contact_angle. Got "+str(self.fixations))

        pin_base_radius="base_radius" in self.fixations
        enforce_theta_via_base_radius=False
        pin_apex_height=False
        p_ref_mode="volume" 
        if pin_base_radius:
            if "volume" in self.fixations:
                p_ref_mode="volume"
            elif "apex_height" in self.fixations:
                p_ref_mode="apex_height"
            elif "contact_angle" in self.fixations:
                p_ref_mode="contact_angle"
            else:
                raise RuntimeError("The set fixations must have exactly two fixed quantities out of: base_radius, apex_height, volume, contact_angle. Got " + str(self.fixations))
        elif "apex_height" in self.fixations:
            pin_apex_height=True
            if "volume" in self.fixations:
                p_ref_mode = "volume"
            elif "contact_angle" in self.fixations:
                p_ref_mode="contact_angle"
            else:
                raise RuntimeError("The set fixations must have exactly two fixed quantities out of: base_radius, apex_height, volume, contact_angle. Got " + str(self.fixations))
        elif "volume" in self.fixations and "contact_angle" in self.fixations:
            p_ref_mode = "volume"
            enforce_theta_via_base_radius=True

        geom = self.drop_geom

        dest_contact_angle=geom.contact_angle
        dest_base_radius=geom.base_radius
        dest_volume=geom.volume*self.volume_factor
        dest_apex_height=geom.apex_height
        if isinstance(self.fixations,dict):
            dest_contact_angle=self.fixations.get("contact_angle",dest_contact_angle)
            dest_base_radius = self.fixations.get("base_radius", dest_base_radius)
            dest_volume = self.fixations.get("volume", dest_volume)
            dest_apex_height = self.fixations.get("apex_height", dest_apex_height)


        # Now we have all the required information to setup the problem

        # Coordinate system is axisymmetric
        self.set_coordinate_system("axisymmetric")



        self.set_scaling(spatial=geom.curv_radius, pressure=2 * self.sigma / geom.curv_radius) # Nondimensionalize with reasonable scales

        # Create a line mesh, but with 2d points, parameter is the arc length from the apex to the contact line
        mesh=LineMesh(N=self.N,nodal_dimension=2,size=geom.curv_radius*geom.contact_angle)
        self.add_mesh(mesh)

        # Setup the initial spherical cap shape
        zcenter = geom.apex_height - geom.curv_radius
        phi = var("lagrangian_x") / geom.curv_radius
        # Deform the mesh to form the initial shape
        eqs = InitialCondition(mesh_x=geom.curv_radius * sin(phi), mesh_y=zcenter+geom.curv_radius * cos(phi))  # Undeformed droplet shape
        # Setting curvature and arclength IC
        eqs += InitialCondition(_curv=2 / geom.curv_radius, _s=var("lagrangian_x"))

        # The reference pressure in the Young-Laplace equation selects the actual solution
        # depending on the fixations, we will adjust p_ref (=p0) depending on the fixations
        p0 = var("p0", domain="globals")
        if p_ref_mode=="volume":
            # Enforcing the volume via the reference pressure
            peq=GlobalLagrangeMultiplier(p0=-dest_volume) # solve the reference pressure to match the initial volume
            peq += TestScaling(p0=1 / dest_volume) # We solve a volumetric equation V-V0=0. The equation is nondimensionalized by V0
            # Integrate to add up to the volume constraint to get p_ref, i.e. p_ref is correct if V=V0
            eqs += WeakContribution(1 / 3 * dot(var("mesh"), var("normal")), testfunction(p0), dimensional_dx=True)
        elif p_ref_mode=="apex_height":
            # P_ref will enforce h(r=0)=h_apex
            peq = GlobalLagrangeMultiplier(p0=-dest_apex_height) # select p0 so that h=h0
            peq += TestScaling(p0=1 / scale_factor("spatial")) # nondimensionalize h-h0=0 by the spatial dimension
            eqs += WeakContribution(var("mesh_y"),testfunction(p0),coordinate_system=cartesian)@"left" # and add h to the constraint equation
        else: # contact_angle
            peq = GlobalLagrangeMultiplier(p0=-cos(dest_contact_angle)) # we adjust p0 so that n_z-cos(theta)=0 holds
            eqs += WeakContribution(var("_norm_y"), testfunction(p0), coordinate_system=cartesian) @ "right"

        # Setting reference pressure without gravity as initial condition
        peq+=Scaling(p0=scale_factor("pressure"))
        peq += InitialCondition(p0=2 * self.sigma / geom.curv_radius)
        self.add_equations(peq@"globals")

        #eqs+=TextFileOutput()
        #eqs+=MeshFileOutput(hide_underscore=False)
        # Solve the Young-Laplace equation
        eqs+=YoungLaplaceEquations(p0,self.sigma,additional_pressure=-self.force_factor.get_symbol()*var("mesh_y")*self.rho_g_ez) 
        # Fix r=0 at the axis of symmetry
        eqs+=DirichletBC(mesh_x=0)@"left"
        # and z=0 at the substrate
        eqs += DirichletBC(mesh_y=0) @ "right"        

        # Depending on the fixations, we might want to fix some mesh positions
        if pin_base_radius:
            eqs += DirichletBC(mesh_x=dest_base_radius) @ "right"
        elif pin_apex_height:
            eqs += DirichletBC(mesh_y=dest_apex_height) @ "left"
        elif enforce_theta_via_base_radius:
            # Or, if volume is enforced by p0, the contact angle must be enforced by adjusting the base radius accordingly
            teq = GlobalLagrangeMultiplier(ca_by_r=-cos(dest_contact_angle)) # solve n_z-cos(theta)=0 by adjusting mesh_x(z=0)
            eqs += WeakContribution(var("_norm_y"), testfunction("ca_by_r",domain="globals"), coordinate_system=cartesian) @ "right"
            # Add the feedback of this Lagrange multiplier
            eqs += WeakContribution(var("ca_by_r", domain="globals"),testfunction("mesh_x"),coordinate_system=cartesian) @ "right"
            self.add_equations(teq @ "globals")

        # Fix the arclength calculation boundaries for the tangential shifting of the nodes
        eqs+=DirichletBC(_s=0)@"left"
        eqs += DirichletBC(_s=geom.curv_radius*geom.contact_angle) @ "right"


        # Measures
        eqs+=IntegralObservables(volume=1/3* dot(var("mesh"), var("normal")),area=1)
        eqs+=IntegralObservables(cartesian,_contact_angle=acos(var("_norm_y")), _ca_denom=1,)@"right"
        eqs+=IntegralObservables(contact_angle=lambda _contact_angle,_ca_denom:_contact_angle/_ca_denom)@"right"
        eqs+=IntegralObservableOutput()+IntegralObservableOutput()@"right"
        self.define_named_var(force_factor=self.force_factor.get_symbol())

        self.add_equations(eqs@"domain")


    def relax_by_gravity(self,output_text:bool=False,globally_convergent_newton:bool=True):
        #self.set_c_compiler("tcc")
        #self.quiet()
        if output_text:
            self.additional_equations+=TextFileOutput()@"domain"
            if not self.is_initialised():
                self.initialise()
            self.output()
        self.solve(max_newton_iterations=1000, globally_convergent_newton=globally_convergent_newton)
        print("RELAXING DROPLET SHAPE BY GRAVITY")
        self.go_to_param(force_factor=1, final_adaptive_solve=False)
        print("DROPLET SHAPE RELAXED BY GRAVITY","REFERENCE PRESSURE:",self.get_ode("globals").get_value("p0"))        
        if output_text:
            self.output_at_increased_time()
        #self.quiet(False)




####### ANALYTICAL DROPLET EVAPORATION RATE ########



class _PopovEvaporationRateByTau(CustomMathExpression):
    def __init__(self,contact_angle:ExpressionOrNum):
        super(_PopovEvaporationRateByTau, self).__init__()
        self.contact_angle=float(contact_angle)


    def update_contact_angle(self,contact_angle:ExpressionOrNum):
        self.contact_angle=float(contact_angle)

    def eval(self,arg_array:NPFloatArray)->float:        
        tau=arg_array[0]
        Nprime=numpy.sqrt(2.0)*(numpy.cosh(tau)+numpy.cos(self.contact_angle))**(3.0/2.0)
        factor=numpy.pi*Nprime/(2.0*numpy.sqrt(2.0)*(numpy.pi-self.contact_angle)**2)
        arg_prefactor=numpy.pi/(2*(numpy.pi-self.contact_angle))
        def integrand(x:float)->float:
            try:
                res=numpy.tanh(arg_prefactor*x)/(numpy.cosh(arg_prefactor*x)*numpy.sqrt(numpy.cosh(x)-numpy.cosh(tau)))
            except:
                res=0
            return res
        integral=integrate.quad(integrand,tau,numpy.inf)
        return factor*integral[0]

class _PrecachedPopovEvaporationRateByTau(InterpolateSpline1d):
    def __init__(self,contact_angle:ExpressionOrNum,numpoints:int=2000,maxtau:int=100):
        tau_array=numpy.linspace(0,maxtau,numpoints,endpoint=True)
        popov=_PopovEvaporationRateByTau(contact_angle=contact_angle)
        ev_array=0*tau_array
        print("Precaching evaporation rate")
        for i,t in enumerate(tau_array):
            ev_array[i]=popov.eval([t])
        print("Done")
        data=numpy.transpose(numpy.vstack([tau_array,ev_array]))
        super(_PrecachedPopovEvaporationRateByTau, self).__init__(data)


class _LebedevEvaporationRateByPhi(CustomMathExpression):
    def __init__(self,contact_angle:ExpressionOrNum):
        super(_LebedevEvaporationRateByPhi, self).__init__()
        self.contact_angle=float(contact_angle)

    def eval(self,arg_array:NPFloatArray)->float:        
        phi=arg_array[0]
        theta=self.contact_angle
        
        expo=numpy.pi/(2*(numpy.pi-theta))
        cosphi=numpy.cos(phi)
        cost=numpy.cos(theta)
        def integrand(beta:float)->float:                        
            A=(1-numpy.cos(theta+beta))**expo
            B=(1-numpy.cos(theta-beta))**expo
            cosb=numpy.cos(beta)
            numer=(A-B)*(cosb-cost)**(expo-0.5)
            denom=(A+B)**2*numpy.sqrt(cosphi-cosb)
            return numer/denom
            
        integral=integrate.quad(integrand,phi,theta)
        factor=pi/(pi-theta)**2 * (sin(theta))**3/(cos(phi)-cos(theta))
        return factor*integral[0]


class _PrecachedLebedevEvaporationRateByPhi(InterpolateSpline1d):
    def __init__(self,contact_angle:ExpressionOrNum,numpoints:int=2000):
        phi_array=numpy.linspace(0,float(contact_angle),numpoints,endpoint=float(contact_angle)>numpy.pi)
        lebedev=_LebedevEvaporationRateByPhi(contact_angle=contact_angle)
        ev_array=0*phi_array
        print("Precaching evaporation rate")
        for i,t in enumerate(phi_array):
            ev_array[i]=lebedev.eval([t])
        print("Done")
        data=numpy.transpose(numpy.vstack([phi_array,ev_array]))
        super(_PrecachedLebedevEvaporationRateByPhi, self).__init__(data)


# For special cases, there exists a simpler result
# For 90°, 135° and 150°, there are other expressions without using torodial integrations. These were developed by Peter Lebedev-Stepanov and Olga Savenko
def _get_j_lebedev(contact_angle,base_radius,with_subexpressions:bool=True,precached:bool=True,precache_points:int=1000,only_special:bool=True):
    x,y=var(["coordinate_x","coordinate_y"])
    geom=DropletGeometry(base_radius=base_radius,contact_angle=contact_angle)
    y+=geom.curv_radius-geom.apex_height
    phi=atan2(x,y)    
    se=(lambda x : subexpression(x)) if with_subexpressions else (lambda x:x)        
    phi=se(phi)
    if is_zero(contact_angle-90*degree):
        return 1
    elif is_zero(contact_angle-135*degree):
        return 1/square_root(2)-0.25/(rational_num(3,2)+square_root(2)*cos(phi))**rational_num(3,2)
    elif is_zero(contact_angle-150*degree):
        return (1+(7+4*square_root(3)*cos(phi))**rational_num(-3,2)-2*(4+2*square_root(3)*cos(phi))**rational_num(-3,2))/2
    elif not only_special:
        if precached:
            return _PrecachedLebedevEvaporationRateByPhi(contact_angle,precache_points)(phi)
        else:
            return _LebedevEvaporationRateByPhi(contact_angle)(phi)
    else:
        return None
    
# Calculate the evaporation rate of a spherical cap shaped droplet, assuming constant vapor c_sat at the interface and far field vapor c_far
# Requires the contact_angle of the droplet, the base_radius as well as the vapor diffusivity Dvap
# At the moment, it only works if the droplet is at the origin of an axisymmetric coordinate system
# When using precached=True (default), we will calculate the evaporation rate once initially and interpolate it during the simulation
# You can choose the number of sample points with precache_points and the max. toroidal tau coordinate with precache_max_tau
# Works only for a constant numerical contact angle
# For 90°, 135° and 150°, there are other expressions without using torodial integrations. These were developed by Peter Lebedev-Stepanov and Olga Savenko
# These cases can be activated by passing prefer_lebedev=True
def get_analytical_popov_evaporation_rate(contact_angle:ExpressionOrNum,base_radius:ExpressionOrNum,c_sat:ExpressionOrNum=1,c_far:ExpressionOrNum=0,Dvap:ExpressionOrNum=1, precached:bool=True,precache_points:int=1000,precache_max_tau:int=100,axisymmetric:bool=True,with_subexpressions:bool=True,prefer_lebedev:Union[bool,Literal["only_special"]]=False)->Expression:
    if not axisymmetric:
        raise RuntimeError("Can only do axisymmetric right now")
    lebedev_factor=None
    if prefer_lebedev:
        lebedev_factor=_get_j_lebedev(contact_angle,base_radius,with_subexpressions=with_subexpressions,precached=precached,precache_points=precache_points,only_special=(prefer_lebedev=="only_special"))
    se=(lambda x : subexpression(x)) if with_subexpressions else (lambda x:x)        
    if lebedev_factor is not None:
        return se(lebedev_factor/base_radius*(c_sat-c_far)*Dvap)  
    else:
        if precached:
            evap_by_tau=_PrecachedPopovEvaporationRateByTau(contact_angle,precache_points,precache_max_tau)
        else:
            evap_by_tau=_PopovEvaporationRateByTau(contact_angle)
        r,z = var(["coordinate_x","coordinate_y"])
        d1_sqr = (r + base_radius) ** 2 + z ** 2
        d2_sqr = (r - base_radius) ** 2 + z ** 2
        tau_toro = se(log(square_root(d1_sqr / d2_sqr)))        
        return se(evap_by_tau(tau_toro)/base_radius*(c_sat-c_far)*Dvap)    
