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
 
from ..expressions import *
from ..expressions.units import meter,second,joule,newton,degree,milli
import scipy.optimize #type:ignore
from ..generic.codegen import EquationTree,InterfaceEquations,GlobalLagrangeMultiplier,WeakContribution,BaseEquations,ODEEquations,Equations
from ..equations.generic import InitialCondition,DependentIntegralObservable
from ..meshes.mesh import InterfaceMesh, ODEStorageMesh, AnyMesh
from .multi_component import MultiComponentNavierStokesInterface

_DefaultCLSpeed=1e-5*meter/second




class GenericContactLineModel:
    """
        Represents a generic contact line model. This class is inherited by the specific contact line models (e.g. pinned, unpinned, stick-slip, etc.).
        It contains the basic methods that are common to all contact line models.
    """
    def __init__(self):
        self._dyn_cl_equation:Optional[Union["DynamicContactLineEquations","SimplePopovContactLineEquations"]]=None
        self._starts_pinned:bool=False
        self._ic_for_dyncl:Dict[str,ExpressionOrNum]={}

    def _setup_for_equation(self,dyn_cl_eq:Union["DynamicContactLineEquations","SimplePopovContactLineEquations"]):
        if (self._dyn_cl_equation is not None) and self._dyn_cl_equation!=dyn_cl_eq:
            raise RuntimeError("Contact line model used for different equations")
        self._dyn_cl_equation=dyn_cl_eq

    def set_missing_information(self,initial_contact_angle:ExpressionOrNum,initial_surface_tension:ExpressionOrNum,initial_contact_line_position:ExpressionOrNum):
        self._ic_for_dyncl["initial_contact_angle"]=initial_contact_angle
        self._ic_for_dyncl["initial_surface_tension"]=initial_surface_tension
        self._ic_for_dyncl["initial_contact_line_position"]=initial_contact_line_position
        pass

    def define_residuals(self, dyncl:"DynamicContactLineEquations"):
        pass

    def define_fields(self, dyncl:"DynamicContactLineEquations"):
        pass

    def before_assigning_equations_postorder(self,dyncl:"DynamicContactLineEquations",mesh:InterfaceMesh):
        pass

    def add_additional_functions(self,dyncl:"DynamicContactLineEquations"):
        pass


class PinnedContactLine(GenericContactLineModel):
    """
        Represents a pinned contact line model. 
        It uses a Lagrange multiplier to enforce the contact line to be pinned.

        Args:
            enforcing_condition: Optional. An expression that can be used to enforce the contact line to be pinned.
    """        

    def __init__(self,enforcing_condition:Optional[Expression]=None):
        super(PinnedContactLine, self).__init__()
        self._starts_pinned=True
        self.enforcing_condition=enforcing_condition

    def define_residuals(self,dyncl:"DynamicContactLineEquations"):
        X = var("mesh")  # contact line position
        vl, vl_test = var_and_test(dyncl.velocity_enforcing_name)
        u_test=testfunction("velocity")
        if self.enforcing_condition is None:
            pos_velo_constraint = dot(mesh_velocity(scheme="BDF1"), dyncl.wall_tangent)
        else:
            pos_velo_constraint=self.enforcing_condition
        dyncl.add_residual(weak(pos_velo_constraint , vl_test))
        dyncl.add_residual(weak(vl, dot(u_test, dyncl.wall_tangent)))


class PinnedContactLineWithCollapsePrevention(PinnedContactLine):
    """
        In case of regular Marangoni flow (e.g. glycerol+water), the pinned contact line can collapse, i.e. the local contact angle can go to zero. 
        We allow to move the contact line here to prevent this collapse, while replicating a pinned contact line.

        This class is a subclass of PinnedContactLine and inherits all its arguments.                    

        Args:
            min_local_contact_angle: The minimum local contact angle that is allowed. Default is 5 degrees.
            contact_line_position: Optional. The position of the contact line. If not set, it will be set to the initial contact line position. Default is None.
            shift_velocity: The velocity with which the contact line is shifted to prevent collapse. Default is 0.1 mm/s.
    """
    def __init__(self,min_local_contact_angle:ExpressionOrNum=5*degree,contact_line_position:Optional[ExpressionOrNum]=None,shift_velocity:ExpressionOrNum=0.1*milli*meter/second):
        super(PinnedContactLine, self).__init__()
        self.min_local_contact_angle=min_local_contact_angle
        self.contact_line_position=contact_line_position
        self.shift_velocity=shift_velocity

    def set_missing_information(self, initial_contact_angle:ExpressionOrNum, initial_surface_tension:ExpressionOrNum, initial_contact_line_position:ExpressionOrNum):
        super(PinnedContactLineWithCollapsePrevention, self).set_missing_information(initial_contact_angle, initial_surface_tension,
                                                                 initial_contact_line_position)
        if self.contact_line_position is None:
            self.contact_line_position = initial_contact_line_position

    def define_fields(self, dyncl:"DynamicContactLineEquations"):
        # Augmented pinned condition
        X,theta = var(["mesh",dyncl.actual_theta_name])  # contact line position
        blend=minimum(maximum(1-theta/self.min_local_contact_angle,0),1) # 0 if contact angle is higher, 1 if contact angle goes to 0
        shift_factor=self.shift_velocity*blend/(1-blend) # Really intense in the limit of vanishing contact angle
        assert self.contact_line_position is not None
        assert dyncl.wall_tangent is not None
        cond_pinned=dot((1-blend)*(X-self.contact_line_position)/scale_factor("temporal")+shift_factor*dyncl.wall_tangent, dyncl.wall_tangent)
        self.enforcing_condition=cond_pinned
        super(PinnedContactLineWithCollapsePrevention, self).define_fields(dyncl)


class UnpinnedContactLine(GenericContactLineModel):
    """
        Represents an unpinned contact line model using Cox-Voinov theory (doi:10.1017/S0022112086000332).

        Args:
            theta_eq: Optional. The equilibrium contact angle. If not set, it has to be passed in the set_missing_information method. Default is None.
            cl_speed_scale: Optional. The speed with which the contact line moves towards the equilibrium contact angle. Default is 1e-5 m/s.
            cl_speed_exponent: Optional. The exponent of the speed with which the contact line moves towards the equilibrium contact angle. Default is 1, use 3 for Cox-Voinov.
    """
        
    def __init__(self,theta_eq:ExpressionNumOrNone=None,cl_speed_scale:ExpressionNumOrNone=_DefaultCLSpeed,cl_speed_exponent:int=1):
        super(UnpinnedContactLine, self).__init__()
        self.theta_eq=theta_eq
        self.cl_speed_scale=cl_speed_scale
        self.cl_speed_exponent=1 # set to 3 for Cox-Voinov

    def get_unpinned_motion_velocity_expression(self,dyncl:Optional["DynamicContactLineEquations"],theta_act_for_popov:ExpressionNumOrNone=None) -> Expression:
        """
        Get the velocity with which the contact line moves towards the equilibrium contact angle. In this method, we just return a Cox-Voinov-like expression:
        The velocity of the contact line is proportional to the difference between the equilibrium contact angle and the actual contact angle, possibly raised to a power given by ``cl_speed_exponent``.

        Args:
            dyncl: The dynamic contact line equations. 
            theta_act_for_popov: If we do not have the dynamic contact line equations, we can pass the actual contact angle here (can be used for more simple models). Default is None.

        Returns:
            The velocity with which the contact line moves towards the equilibrium contact angle.
        """
        theta_eq=self.get_equilibrium_contact_angle_expression(dyncl)
        if dyncl is None:
            theta_act=theta_act_for_popov
        else:
            theta_act=dyncl.get_actual_contact_angle_expression()
        assert theta_act is not None
        assert self.cl_speed_scale is not None
        if self.cl_speed_exponent==1:
            return self.cl_speed_scale*(theta_eq-theta_act)
        else:
            return self.cl_speed_scale*(theta_eq**self.cl_speed_exponent-theta_act**self.cl_speed_exponent)

    def get_unpinned_indicator(self,dyncl:Optional["DynamicContactLineEquations"],simple_popov_unpinned_indicator_name:Optional[str]=None)->Expression:
        return Expression(1) # Always unpinned

    def get_equilibrium_contact_angle_expression(self,dyncl:Optional["DynamicContactLineEquations"]) -> Expression:
        if self.theta_eq is None:
            raise RuntimeError("Unpinned contact line has no equilibrium contact angle set. Set it in the constructor or use the set_missing_information to pass the initial contact angle as default value in the Problem.define_problem method")
        return subexpression(self.theta_eq)

    def equations_for_simple_popov_model(self,theta_act_name:str,rc_name:str)->Equations:
        eqs=GlobalLagrangeMultiplier(theta_eq=var("theta_eq")-self.get_equilibrium_contact_angle_expression(None))
        theta_eq=var("theta_eq")
        theta_act=var(theta_act_name)
        eqs+=InitialCondition(theta_eq=self._ic_for_dyncl["initial_contact_angle"])
        if self.cl_speed_scale is None:
            eqs+=WeakContribution(theta_act-theta_eq,testfunction(theta_act)/test_scale_factor(theta_act_name))
        else:
            upind = self.get_unpinned_indicator(None,"unpinned_indicator")
            rcl=var(rc_name)
            # Degrade automatically to BDF1 when pinning mode has changes
            must_degrade = heaviside(absolute(upind - evaluate_in_past(upind)) - 0.5)
            dXdt = must_degrade * partial_t(rcl, scheme="BDF1",ALE=False) + (1 - must_degrade) * partial_t(rcl, scheme="BDF2",ALE=False)
            actual_cl_velo = dXdt
            desired_cl_velo = upind * self.get_unpinned_motion_velocity_expression(None,var(theta_act_name))

            eqs+=WeakContribution(actual_cl_velo - desired_cl_velo, testfunction(theta_act)/test_scale_factor(theta_act_name)*scale_factor("temporal")/scale_factor("spatial"))
        return eqs
        pass

    def define_residuals(self,dyncl:"DynamicContactLineEquations"):
        X = var("mesh")  # contact line position
        vl, vl_test = var_and_test(dyncl.velocity_enforcing_name)
        u_test = testfunction("velocity")
        upind=self.get_unpinned_indicator(dyncl)


        # Impose weakly the equilibrium contact angle (e.g. in case of stationary solves)
        sigma_value=dyncl.get_surface_tension_at_cl_expression()
        theta_eq=self.get_equilibrium_contact_angle_expression(dyncl)
        m = sin(theta_eq) * dyncl.wall_normal + cos(theta_eq) * dyncl.wall_tangent
        dyncl.add_residual(weak(sigma_value, dot(m, u_test)))


        if self.cl_speed_scale is None:
            # Impose contact angle directly via Lagrange
            theta_desired=self.get_equilibrium_contact_angle_expression(dyncl)
            theta_present=dyncl.get_actual_contact_angle_expression()
            dyncl.add_residual(weak(theta_present-theta_desired,vl_test*scale_factor("velocity")))
            dyncl.add_residual(weak(vl, dot(u_test, dyncl.wall_tangent)))
        else:
            # Degrade automatically to BDF1 when pinning mode has changes
            must_degrade=heaviside(absolute(upind-evaluate_in_past(upind))-0.5)
            dXdt=must_degrade*mesh_velocity(scheme="BDF1")+(1-must_degrade)*partial_t(X, scheme="BDF2",ALE=False)
            actual_cl_velo = dot(dXdt, dyncl.wall_tangent)
            desired_cl_velo=upind*self.get_unpinned_motion_velocity_expression(dyncl)
            dyncl.add_residual(weak(actual_cl_velo-desired_cl_velo, vl_test))
            dyncl.add_residual(weak(vl, dot(u_test, dyncl.wall_tangent)))



    def set_missing_information(self,initial_contact_angle:ExpressionOrNum,initial_surface_tension:ExpressionOrNum,initial_contact_line_position:ExpressionOrNum):
        super(UnpinnedContactLine, self).set_missing_information(initial_contact_angle,initial_surface_tension,initial_contact_line_position)
        if self.theta_eq is None:
            self.theta_eq=initial_contact_angle

    def add_additional_functions(self,dyncl:"DynamicContactLineEquations"):
        nd=dyncl.get_nodal_dimension()
        dx=dyncl.get_dx()
        dyncl.add_integral_function("_theta_eq_integral",self.get_equilibrium_contact_angle_expression(dyncl)*dx)
        dyncl.add_dependent_integral_function("eq_contact_angle",lambda _theta_eq_integral,_cl_integral:_theta_eq_integral/_cl_integral)
        if nd>0:
            dyncl.add_integral_function("_clpos_integral_x",var("coordinate_x")*dx)
            dyncl.add_dependent_integral_function("cl_pos_x",lambda _clpos_integral_x,_cl_integral:_clpos_integral_x/_cl_integral)
            if nd>1:
                dyncl.add_integral_function("_clpos_integral_y", var("coordinate_y") * dx)
                dyncl.add_dependent_integral_function("cl_pos_y", lambda _clpos_integral_y,_cl_integral: _clpos_integral_y / _cl_integral)
                if nd>2:
                    dyncl.add_integral_function("_clpos_integral_z", var("coordinate_z") * dx)
                    dyncl.add_dependent_integral_function("cl_pos_z", lambda _clpos_integral_z,_cl_integral: _clpos_integral_z / _cl_integral)




class StickSlipContactLine(UnpinnedContactLine):
    """
        Represents a stick-slip contact line model, where the contact line can be pinned or unpinned depending on the contact angle dynamics.

        This class is a subclass of UnpinnedContactLine and inherits all its arguments.                    

        Args:
            theta_eq: Optional. The equilibrium contact angle. If not set, it has to be passed in the set_missing_information method. Default is None.
            cl_speed_scale: Optional. The speed with which the contact line moves towards the equilibrium contact angle. Default is 1e-5 m/s.
            cl_speed_exponent: Optional. The exponent of the speed with which the contact line moves towards the equilibrium contact angle. Default is 1.
    """

    def __init__(self,theta_eq:ExpressionNumOrNone=None,cl_speed_scale:ExpressionNumOrNone=_DefaultCLSpeed,cl_speed_exponent:int=1):
        super(StickSlipContactLine, self).__init__(theta_eq=theta_eq,cl_speed_scale=cl_speed_scale,cl_speed_exponent=cl_speed_exponent)
        self._initial_pin_info=[1,0] # unpinned, but not forced
        self._dynamics:Dict[Tuple[bool,bool,int],Tuple[ExpressionOrNum,bool,bool,float]]= {} # maps keys (unpinned,requid dtheta/dt sign) -> angle, factor


    def _set_dynamics_info(self,unpin:bool,above:bool,angle:ExpressionOrNum,explicit:bool,as_factor:bool,required_dthetasign:int,heaviside_smoothing:float):
        entry:Tuple[bool,bool,int]=(unpin,above,required_dthetasign)
        if angle is None:
            if entry in self._dynamics.keys():
                del self._dynamics[entry]
        else:
            self._dynamics[entry]=cast(Tuple[ExpressionOrNum,bool,bool,float],(angle,explicit,as_factor,heaviside_smoothing))

    def set_receding_unpin_below_angle(self,angle:ExpressionOrNum,explicit:bool=True,as_factor:bool=False,only_if_decaying:bool=True,heaviside_smoothing:float=0.0):
        """
        Set the dynamics for the contact line to unpin when the contact angle is below a certain angle.

        Args:
            angle: Angle or relative factor to the equilibrium contact angle.
            explicit: Handle the dynamics after each successful time step. 
            as_factor: ``angle`` is treated as numerical factor times the equilibrium contact angle.
            only_if_decaying: Dynamics are only applied if the contact angle is decaying.
            heaviside_smoothing: Smoothing parameter for the Heaviside function for an implicit treatment of the dynamics.

        Returns:
            The StickSlipContactLine itself for chaining.
        """
        self._set_dynamics_info(True,False,angle,explicit,as_factor,-1 if only_if_decaying else 0,heaviside_smoothing)
        return self  # To concatenate commands

    def set_receding_pin_above_angle(self, angle:ExpressionOrNum, explicit:bool=True,as_factor:bool=False,only_if_growing:bool=True,heaviside_smoothing:float=0.0):
        """
        Set the dynamics for the contact line to pin when the contact angle is above a certain angle.

        Args:
            angle: Angle or relative factor to the equilibrium contact angle.
            explicit: Handle the dynamics after each successful time step. 
            as_factor: ``angle`` is treated as numerical factor times the equilibrium contact angle.
            only_if_growing: Dynamics are only applied if the contact angle is growing.
            heaviside_smoothing: Smoothing parameter for the Heaviside function for an implicit treatment of the dynamics.

        Returns:
            The StickSlipContactLine itself for chaining.
        """        
        self._set_dynamics_info(False, True,angle, explicit,as_factor,1 if only_if_growing else 0,heaviside_smoothing)
        return self # To concatenate commands

    def set_advancing_pin_below_angle(self,angle:ExpressionOrNum,explicit:bool=True,as_factor:bool=False,only_if_decaying:bool=True,heaviside_smoothing:float=0.0):
        """
        Set the dynamics for the contact line to pin when the contact angle is below a certain angle.

        Args:
            angle: Angle or relative factor to the equilibrium contact angle.
            explicit: Handle the dynamics after each successful time step. 
            as_factor: ``angle`` is treated as numerical factor times the equilibrium contact angle.
            only_if_decaying: Dynamics are only applied if the contact angle is decaying.
            heaviside_smoothing: Smoothing parameter for the Heaviside function for an implicit treatment of the dynamics.

        Returns:
            The StickSlipContactLine itself for chaining.
        """        
        self._set_dynamics_info(False,False,angle,explicit,as_factor,-1 if only_if_decaying else 0,heaviside_smoothing)
        return self  # To concatenate commands

    def set_advancing_unpin_above_angle(self,angle:ExpressionOrNum,explicit:bool=True,as_factor:bool=False,only_if_growing:bool=True,heaviside_smoothing:float=0.0):
        """
        Set the dynamics for the contact line to unpin when the contact angle is above a certain angle.

        Args:
            angle: Angle or relative factor to the equilibrium contact angle.
            explicit: Handle the dynamics after each successful time step. 
            as_factor: ``angle`` is treated as numerical factor times the equilibrium contact angle.
            only_if_growing: Dynamics are only applied if the contact angle is growing.
            heaviside_smoothing: Smoothing parameter for the Heaviside function for an implicit treatment of the dynamics.

        Returns:
            The StickSlipContactLine itself for chaining.
        """                
        self._set_dynamics_info(True,True,angle,explicit,as_factor,1 if only_if_growing else 0,heaviside_smoothing)
        return self  # To concatenate commands

    def define_fields(self, dyncl:"DynamicContactLineEquations"):
        dyncl.define_scalar_field(dyncl.unpinned_indicator_name,"C2",scale=1,testscale=1)
        dyncl.define_scalar_field(dyncl.override_dynamics_name, "C2", scale=1, testscale=1)

    def equations_for_simple_popov_model(self, theta_act_name:str, rc_name:str)->Equations:
        eqs=super(StickSlipContactLine, self).equations_for_simple_popov_model(theta_act_name,rc_name)
        eqs+=GlobalLagrangeMultiplier(unpinned_indicator=0,override_cl_dynamics=0)
        addeqs=self.define_stick_slip_dynamics_residuals(None,theta_act_name)
        assert addeqs is not None
        eqs+=addeqs
        return eqs


    def get_unpinned_indicator(self,dyncl:Optional["DynamicContactLineEquations"],simple_popov_unpinned_indicator_name:Optional[str]=None):
        if dyncl is not None:
            return var(dyncl.unpinned_indicator_name)
        else:
            assert simple_popov_unpinned_indicator_name is not None
            return var(simple_popov_unpinned_indicator_name)

    def define_stick_slip_dynamics_residuals(self,dyncl:Optional["DynamicContactLineEquations"],simple_popov_theta_act_name:Optional[str]=None)->Optional[BaseEquations]:
        if dyncl is not None:
            up, up_test = var_and_test(dyncl.unpinned_indicator_name)
            oc = var(dyncl.override_dynamics_name)
            theta_act = dyncl.get_actual_contact_angle_expression()
        else:
            up, up_test = var_and_test("unpinned_indicator")
            oc = var("override_cl_dynamics")
            assert simple_popov_theta_act_name is not None
            theta_act = var(simple_popov_theta_act_name)
        # if oc==1: forcing of unpinned dynamics
        # if oc==-1: forcing of pinned dynamics
        # if oc==0: governed by adv. / rec. contact angles or leave as it is
        pin_value = 0
        unpin_value = 1
        as_it_is_value = evaluate_in_past(up)

        pin_when = None
        unpin_when = None

        theta_eq = self.get_equilibrium_contact_angle_expression(dyncl)

        # Now add the dynamics
        for entry, info in self._dynamics.items():
            if self.cl_speed_scale is None:
                raise RuntimeError("If you set cl_speed_scale to None, i.e. enforcing the contact angle to be at equilibrium, you cannot have stick slip dynamics")
            unpin, above, dtheta_sign = entry
            angle, explicit, as_factor, heaviside_smoothing = info
            if as_factor:
                angle = angle * theta_eq
            theta_act_val = theta_act
            if explicit:
                theta_act_val = evaluate_in_past(theta_act)
            diff = theta_act_val - angle
            if not above:
                diff = -diff
            if heaviside_smoothing > 0:
                factor = atan(diff / heaviside_smoothing) / pi + 0.5
            else:
                factor = heaviside(diff)
            if dtheta_sign != 0:
                dt_sign = partial_t(theta_act, nondim=True, scheme="BDF1")
                if explicit:
                    dt_sign = evaluate_in_past(theta_act) - evaluate_in_past(theta_act, 2)
                factor *= heaviside(dtheta_sign * dt_sign)
            if unpin:
                if unpin_when is None:
                    unpin_when=factor
                else:
                    unpin_when = maximum(unpin_when, factor)
            else:
                if pin_when is None:
                    pin_when=factor
                else:
                    pin_when = maximum(pin_when, factor)

        if pin_when is None:
            pin_when=heaviside(-0.5 - oc)
        else:
            pin_when=maximum(heaviside(-0.5 - oc),heaviside(-oc+0.5)*pin_when)

        if unpin_when is None:
            unpin_when=heaviside(oc - 0.5)
        else:
            unpin_when=maximum(heaviside(oc - 0.5),heaviside(oc+0.5)*unpin_when)

        pin_when = subexpression(pin_when)
        unpin_when = subexpression(unpin_when)
        as_it_is_when = subexpression(1 - maximum(pin_when, unpin_when)+minimum(pin_when, unpin_when))

        dynamics = pin_value * pin_when + unpin_value * unpin_when + as_it_is_when * as_it_is_value
        dynamics = subexpression(dynamics)
        if dyncl is not None:
            dyncl.add_residual(weak(up - dynamics, up_test, coordinate_system=cartesian))
            if self._initial_pin_info is not None:
                dyncl.set_initial_condition(dyncl.unpinned_indicator_name, self._initial_pin_info[0], True)
                dyncl.set_initial_condition(dyncl.override_dynamics_name, self._initial_pin_info[1], True)
            return
        else:
            eqs=WeakContribution(up-dynamics,up_test)
            if self._initial_pin_info is not None:
                eqs+=InitialCondition(unpinned_indicator= self._initial_pin_info[0],override_cl_dynamics=self._initial_pin_info[1])
            return eqs


    def define_residuals(self,dyncl:"DynamicContactLineEquations"):
        super(StickSlipContactLine, self).define_residuals(dyncl)
        self.define_stick_slip_dynamics_residuals(dyncl)

    def before_assigning_equations_postorder(self,dyncl:"DynamicContactLineEquations",mesh:InterfaceMesh):
        oc_index = mesh.has_interface_dof_id(dyncl.override_dynamics_name)
        for n in mesh.nodes():
            oc_ind = n.additional_value_index(oc_index)
            n.pin(oc_ind)
            #n.set_value(unpinned_ind, pinval)

    def _handle_pin_unpin(self,upval:int,forced:bool):
        forcval=0 if not forced else (1 if upval==0 else -1)
        assert self._dyn_cl_equation is not None
        if self._dyn_cl_equation._is_ode():
            ode_mesh=self._dyn_cl_equation._on_mesh 
            assert isinstance(ode_mesh,ODEStorageMesh)
            ode_mesh.set_value(override_cl_dynamics=forcval,unpinned_indicator=upval)
        else:
            cleq=self._dyn_cl_equation
            assert isinstance(cleq,DynamicContactLineEquations)
            mesh=cleq._on_mesh 
            assert isinstance(mesh,InterfaceMesh)            
            oc_index = mesh.has_interface_dof_id(cleq.override_dynamics_name)
            up_index = mesh.has_interface_dof_id(cleq.unpinned_indicator_name)
            for n in mesh.nodes():
                oc_ind = n.additional_value_index(oc_index)
                up_ind=n.additional_value_index(up_index)
                n.set_value(oc_ind,forcval)
                n.set_value(up_ind,upval)


    def pin(self,forced:bool=False):
        """
        Pins the contact line. If forced is set to True, the contact line will be pinned even if the dynamics would not pin it.

        Args:
            forced: Force the contact line to be pinned. Default is False.
        """
        if self.cl_speed_scale is None:
            raise RuntimeError("If you set cl_speed_scale to None, i.e. enforcing the contact angle to be at equilibrium, you cannot pin")
        if self._dyn_cl_equation is None:
            self._initial_pin_info=[0,-1 if forced else 0]
        else:
            self._handle_pin_unpin(0,forced)

    def unpin(self,forced:bool=False):
        """
        Unpins the contact line. If forced is set to True, the contact line will be depinned even if the dynamics would pin it.

        Args:
            forced: Force the contact line to be unpinned. Default is False.
        """
        if self._dyn_cl_equation is None:
            self._initial_pin_info=[0,1 if forced else 0]
        else:
            self._handle_pin_unpin(1,forced)

    def add_additional_functions(self,dyncl:"DynamicContactLineEquations"):
        super(StickSlipContactLine, self).add_additional_functions(dyncl)
        theta_eq = self.get_equilibrium_contact_angle_expression(dyncl)
        dx=dyncl.get_dx()
        for entry, info in self._dynamics.items():
            unpin, above, _ = entry
            angle, _, as_factor, _ = info
            if as_factor:
                angle = angle * theta_eq
            name:str="angle_"+("unpin" if unpin else "pin")+"_"+("above" if above else "below")
            dyncl.add_integral_function("_"+name+"_integral",angle*dx)
            dyncl.add_dependent_integral_function(name,DependentIntegralObservable(lambda a,b:a/b,"_"+name+"_integral","_cl_integral"))


class YoungDupreContactLine(StickSlipContactLine):
    """
        Represents a contact line model using the Young-Dupre law for the contact angle. It is useful to use this model when the equilibrum contact angle is composition-dependent.

        This class is a subclass of StickSlipContactLine and inherits all its arguments.

        Args:
            sigma_sg: Optional. The surface tension of the gas-liquid interface. If not set, it has to be passed in the set_missing_information method. Default is None.
            sigma_sl: Optional. The surface tension of the solid-liquid interface. If not set, it has to be passed in the set_missing_information method. Default is None.
            cl_speed_scale: Optional. The speed with which the contact line moves towards the equilibrium contact angle. Default is 1e-5 m/s.
            cl_speed_exponent: Optional. The exponent of the speed with which the contact line moves towards the equilibrium contact angle. Default is 1.
            line_tension: Optional. The line tension. Default is None.
    """

    def __init__(self,sigma_sg:ExpressionNumOrNone=None,sigma_sl:ExpressionNumOrNone=None,cl_speed_scale:ExpressionNumOrNone=_DefaultCLSpeed,cl_speed_exponent:int=1,line_tension:ExpressionNumOrNone=None):
        super(YoungDupreContactLine, self).__init__(theta_eq=None,cl_speed_scale=cl_speed_scale,cl_speed_exponent=cl_speed_exponent)
        self.sigma_sg=sigma_sg
        self.sigma_sl = sigma_sl
        self._delta_sigma:ExpressionNumOrNone=None
        self.line_tension=line_tension

    def get_equilibrium_contact_angle_expression(self,dyncl:Optional["DynamicContactLineEquations"]):
        assert self._delta_sigma is not None
        assert dyncl is not None
        if self.line_tension is None:
            line_tension=0
        else:
            line_tension=self.line_tension
        return acos(maximum(-1, (self._delta_sigma+line_tension/var("coordinate_x")) / dyncl.get_surface_tension_at_cl_expression()))

    def set_missing_information(self,initial_contact_angle:ExpressionOrNum,initial_surface_tension:ExpressionOrNum,initial_contact_line_position:ExpressionOrNum):
        super(YoungDupreContactLine, self).set_missing_information(initial_contact_angle,initial_surface_tension,initial_contact_line_position)
        if self.theta_eq is None:
            self.theta_eq = initial_contact_angle

        if self.sigma_sg is not None and self.sigma_sl is not None:
            # All specified
            self._delta_sigma=self.sigma_sg-self.sigma_sl
        else: # Calc sigma_sg-sigma_sl from initial contact angle and surface tension
            self._delta_sigma=cos(initial_contact_angle)*initial_surface_tension



class KwokNeumannContactLine(StickSlipContactLine):
    """
        Represents a contact line model using the Kwok-Neumann law for the contact angle. It is useful to use this model when the equilibrum contact angle is composition-dependent.

        This class is a subclass of StickSlipContactLine and inherits all its arguments.

        Args:
            cl_speed_scale: Optional. The speed with which the contact line moves towards the equilibrium contact angle. Default is 1e-5 m/s.
            beta: Optional. The fit parameter beta. Default is 124.7 m^4/J^2.
            sigma_sg_0: Optional. The surface tension of the gas-liquid interface at the contact line. If not set, it will be estimated from the initial contact angle and surface tension. Default is None.
            cl_speed_exponent: Optional. The exponent of the speed with which the contact line moves towards the equilibrium contact angle. Default is 1.
    """
    def __init__(self,cl_speed_scale:ExpressionNumOrNone=_DefaultCLSpeed,beta:ExpressionOrNum=124.7*meter**4/joule**2,sigma_sg_0:ExpressionNumOrNone=None,cl_speed_exponent:int=1):
        super(KwokNeumannContactLine, self).__init__(theta_eq=None,cl_speed_scale=cl_speed_scale,cl_speed_exponent=cl_speed_exponent)
        self.beta=beta
        self.sigma_sg_0:ExpressionNumOrNone=sigma_sg_0

    def get_equilibrium_contact_angle_expression(self,dyncl:Optional["DynamicContactLineEquations"]):
        assert dyncl is not None
        sigma = dyncl.get_surface_tension_at_cl_expression()
        assert self.sigma_sg_0 is not None
        #arccos_arg = -1 + 2 * square_root(maximum(0, self.sigma_sg_0 / sigma)) * exp(-self.beta * (self.sigma_sg_0 - sigma) ** 2)
        arccos_arg = -1 + 2 * square_root(self.sigma_sg_0/ sigma) * exp(-self.beta * (self.sigma_sg_0 - sigma) ** 2)
        return acos(arccos_arg)

    def set_missing_information(self,initial_contact_angle:ExpressionOrNum,initial_surface_tension:ExpressionOrNum,initial_contact_line_position:ExpressionOrNum):
        super(KwokNeumannContactLine, self).set_missing_information(initial_contact_angle,initial_surface_tension,initial_contact_line_position)
        if self.sigma_sg_0 is None:
            #Find root: However, we must cast things to numpy/scipy
            ica=float(initial_contact_angle)
            sigma0=float(initial_surface_tension/(newton/meter))
            beta=float(self.beta/(meter**4/joule**2))
            def rootfunc(sigma_sg_0:float):
                return numpy.cos(ica) - (-1 + 2 * numpy.sqrt(sigma_sg_0 / sigma0) * numpy.exp(-beta * (sigma_sg_0 - sigma0) ** 2))
            opt=scipy.optimize.root(rootfunc,0) #type:ignore
            if not opt.success: #type:ignore
                raise RuntimeError("Cannot estimate the required fit parameter sigma_sg_0 from the initial contact line information")
            if opt.x[0]<0: #type: ignore
                opt.x[0]=0 #type: ignore
            self.sigma_sg_0=float(opt.x[0])*newton/meter #type:ignore
            print("Kwok-Neumann contact line estimated sigma_sg_0="+str(self.sigma_sg_0))



class WenzelContactLine(YoungDupreContactLine):
    """
        See https://www.researchgate.net/profile/Gene-Whyman/publication/239161424_The_rigorous_derivation_of_Young_Cassie-Baxter_and_Wenzel_equations_and_the_analysis_of_the_contact_angle_hysteresis_phenomenon/links/56def81a08ae6a46a1849b06/The-rigorous-derivation-of-Young-Cassie-Baxter-and-Wenzel-equations-and-the-analysis-of-the-contact-angle-hysteresis-phenomenon.pdf for more information.

        This class is a subclass of YoungDupreContactLine and inherits all its arguments.
    """
    def __init__(self,roughness:ExpressionOrNum=1,sigma_sg:ExpressionNumOrNone=None,sigma_sl:ExpressionNumOrNone=None,cl_speed_scale:ExpressionNumOrNone=_DefaultCLSpeed,cl_speed_exponent:int=1):
        super(WenzelContactLine, self).__init__(sigma_sg=sigma_sg,sigma_sl=sigma_sl,cl_speed_scale=cl_speed_scale,cl_speed_exponent=cl_speed_exponent)
        self.roughness=roughness

    def get_equilibrium_contact_angle_expression(self, dyncl:Optional["DynamicContactLineEquations"]):
        assert self._delta_sigma is not None
        assert dyncl is not None
        return acos(maximum(-1, self.roughness*self._delta_sigma / dyncl.get_surface_tension_at_cl_expression()))

    def set_missing_information(self,initial_contact_angle:ExpressionOrNum,initial_surface_tension:ExpressionOrNum,initial_contact_line_position:ExpressionOrNum):
        if self.sigma_sg is not None and self.sigma_sl is not None:
            super(WenzelContactLine, self).set_missing_information(initial_contact_angle,initial_surface_tension,initial_contact_line_position)
        else:
            super(WenzelContactLine, self).set_missing_information(initial_contact_angle, initial_surface_tension,initial_contact_line_position)
            assert self._delta_sigma is not None
            self._delta_sigma/=self.roughness
        


class CassieBaxterContactLine(YoungDupreContactLine):
    """
        This assumes that we have pillars with air pockets. There is a more general version of Cassie-Baxter for arbitrary heterogenous substrates. See
        https://www.researchgate.net/profile/Gene-Whyman/publication/239161424_The_rigorous_derivation_of_Young_Cassie-Baxter_and_Wenzel_equations_and_the_analysis_of_the_contact_angle_hysteresis_phenomenon/links/56def81a08ae6a46a1849b06/The-rigorous-derivation-of-Young-Cassie-Baxter-and-Wenzel-equations-and-the-analysis-of-the-contact-angle-hysteresis-phenomenon.pdf for more information.

        This class is a subclass of YoungDupreContactLine and inherits all its arguments.
    """
    def __init__(self,pillar_fraction:ExpressionOrNum=0.5,sigma_sg:ExpressionNumOrNone=None,sigma_sl:ExpressionNumOrNone=None,cl_speed_scale:ExpressionNumOrNone=_DefaultCLSpeed,cl_speed_exponent:int=1):
        super(CassieBaxterContactLine, self).__init__(sigma_sg=sigma_sg, sigma_sl=sigma_sl,cl_speed_scale=cl_speed_scale,cl_speed_exponent=cl_speed_exponent)
        self.pillar_fraction = pillar_fraction # between 0 (all air) and 1 (all solid)

    def get_equilibrium_contact_angle_expression(self,dyncl:Optional["DynamicContactLineEquations"]):
        assert dyncl is not None
        sigma_cl=dyncl.get_surface_tension_at_cl_expression()        
        assert self._delta_sigma is not None
        return acos(maximum(-1,self.pillar_fraction*(self._delta_sigma)/sigma_cl-(1-self.pillar_fraction)))

    def set_missing_information(self,initial_contact_angle:ExpressionOrNum,initial_surface_tension:ExpressionOrNum,initial_contact_line_position:ExpressionOrNum):
        if self.sigma_sg is not None and self.sigma_sl is not None:
            super(CassieBaxterContactLine, self).set_missing_information(initial_contact_angle,initial_surface_tension,initial_contact_line_position)
        else:
            super(CassieBaxterContactLine, self).set_missing_information(initial_contact_angle, initial_surface_tension,initial_contact_line_position)
            self._delta_sigma=(cos(initial_contact_angle)*initial_surface_tension+(1-self.pillar_fraction)*initial_surface_tension)/self.pillar_fraction




class DynamicContactLineEquations(InterfaceEquations):
    """
        Represents the boundary conditions to be set at the moving contact line.

        This class requires the parent equations to be of type DynamicContactLineEquations, meaning that if DynamicContactLineEquations (or subclasses) are not defined in the parent domain, an error will be raised.

        Args:
            model: The model for the contact line. This model must be a subclass of GenericContactLineModel.
            wall_normal: The normal vector of the wall. Default is vector(0,1).
            wall_tangent: The tangent vector of the wall. Default is None, meaning it is calculated from the wall normal and the normal of the free surface.
            unpinned_indicator_name: Optional. The name of the unpinned indicator. Default is "_is_unpinned".
            velocity_enforcing_name: Optional. The name of the velocity enforcing field. Default is "_cl_velo_lagr".
            actual_theta_name: Optional. The name of the actual contact angle. Default is "measured_contact_angle".
            surface_tension_name: Optional. The name of the surface tension. Default is "surf_tens_at_cl".
            override_dynamics_name: Optional. The name of the override dynamics field. Default is "_override_cl_dynamics".
            with_observables: Optional. If True, the observables for the contact line will be added. Default is False.
    """
    required_parent_type = MultiComponentNavierStokesInterface

    def __init__(self,model:GenericContactLineModel,wall_normal:ExpressionOrNum=vector(0,1),wall_tangent:ExpressionOrNum=None,unpinned_indicator_name:str="_is_unpinned",velocity_enforcing_name:str="_cl_velo_lagr",actual_theta_name:str="measured_contact_angle",surface_tension_name:str="surf_tens_at_cl",override_dynamics_name:str="_override_cl_dynamics",with_observables:bool=False):
        super(DynamicContactLineEquations, self).__init__()
        self.model=model
        self.wall_normal=wall_normal
        self.wall_tangent=wall_tangent
        if self.wall_tangent is None:
            # bac-cab rule assuming the wall_normal is normalized. Pointing inward to the bulk domain, along the substrate (i.e. orthogonal to the wall normal)
            self.wall_tangent=self.wall_normal*dot(self.wall_normal,var("normal",domain=".."))-var("normal",domain="..")
            self.wall_tangent=subexpression(self.wall_tangent/square_root(dot(self.wall_tangent,self.wall_tangent)))
        self.unpinned_indicator_name=unpinned_indicator_name
        self.override_dynamics_name:str = override_dynamics_name
        self.velocity_enforcing_name=velocity_enforcing_name
        self.actual_theta_name=actual_theta_name
        self.surface_tension_name=surface_tension_name
        self.enforce_proj_interface_velo_for_surfs_name="_enforce_uinterf_proj"
        self.project_surface_tension=True
        self.with_observables=with_observables
        self.model._setup_for_equation(self) 
        self._on_mesh:Optional[InterfaceMesh]=None

    def define_fields(self):
        self.define_scalar_field(self.velocity_enforcing_name,"C2",scale=1/test_scale_factor("velocity"),testscale=1/scale_factor("velocity"))
        self.define_scalar_field(self.actual_theta_name, "C2", scale=1,testscale=1)
        if self.project_surface_tension:
            self.define_scalar_field(self.surface_tension_name,"C2",scale=1 / test_scale_factor("velocity"),testscale=1/scale_factor(self.surface_tension_name))
        self.model.define_fields(self)
        inter=self.get_parent_equations()
        assert isinstance(inter,MultiComponentNavierStokesInterface)
        if len(inter.interface_props._surfactants)>0: 
            self.define_vector_field(self.enforce_proj_interface_velo_for_surfs_name,inter.surfactant_advect_velo_space,scale=1/test_scale_factor(inter.surfactant_advect_velo_name),testscale=1/scale_factor(inter.surfactant_advect_velo_name))

    def get_surface_tension_at_cl_expression(self)->ExpressionOrNum:
        if self.project_surface_tension:
            return var(self.surface_tension_name)
        else:
            peqs=self.get_parent_equations()
            assert isinstance(peqs,MultiComponentNavierStokesInterface)
            sigm=peqs.interface_props.surface_tension
            return sigm
            #return evaluate_in_domain(sigm,domain=self.get_parent_domain())

    def get_actual_contact_angle_expression(self):
        return var(self.actual_theta_name)

    def define_residuals(self):
        theta_act, theta_act_test = var_and_test(self.actual_theta_name)

        # "Measure" the actual contact angle
        N=var("normal")
        self.add_residual(weak(theta_act - acos(-dot(N, self.wall_tangent)), theta_act_test))
        init_ca=self.model._ic_for_dyncl.get("initial_contact_angle", None)
        if init_ca is not None: 
            self.set_initial_condition(self.actual_theta_name,init_ca,degraded_start=True) 

        parent=self.get_parent_equations()
        assert isinstance(parent,MultiComponentNavierStokesInterface)
        # Project surface tension if demanded
        if self.project_surface_tension:
            sigma_proj, sigma_proj_test = var_and_test(self.surface_tension_name)            
            sigm_real = parent.interface_props.surface_tension
            self.add_residual(weak(sigma_proj - sigm_real,sigma_proj_test))
            if self.model._ic_for_dyncl.get("initial_surface_tension",None): 
                sigma0=self.model._ic_for_dyncl.get("initial_surface_tension",None) 
                assert sigma0 is not None
                self.set_initial_condition(self.surface_tension_name,sigma0,degraded_start=True)

        # In case of surfactants, enforce the velocity of the contact line for the projected velocity        
        if len(parent.interface_props._surfactants) > 0: 
            upro,upro_test=var_and_test(parent.surfactant_advect_velo_name)
            enf_upro,enf_upro_test=var_and_test(self.enforce_proj_interface_velo_for_surfs_name)
            self.add_residual(weak(enf_upro,upro_test))
            self.add_residual(weak(upro-mesh_velocity(),enf_upro_test))

        # Model specifics
        self.model.define_residuals(self)


    def before_assigning_equations_postorder(self, mesh:AnyMesh):
        assert isinstance(mesh,InterfaceMesh)
        super(DynamicContactLineEquations, self).before_assigning_equations_postorder(mesh)
        self._on_mesh = mesh
        self.model.before_assigning_equations_postorder(self,mesh)


    # Update the mesh if necessary
    def after_remeshing(self,eqtree:EquationTree):
        assert isinstance(eqtree._mesh,InterfaceMesh) 
        self._on_mesh=eqtree._mesh 

    def _init_output(self,eqtree:EquationTree,continue_info:Optional[Dict[str,Any]],rank:int): # Not for the output, but used to store the mesh information here
        super()._init_output(eqtree,continue_info,rank)
        assert isinstance(eqtree._mesh,InterfaceMesh) 
        self._on_mesh = eqtree._mesh 

    def define_additional_functions(self):
        if self.with_observables:
            dx = self.get_dx()
            self.add_integral_function("_cl_integral", 1 * dx) # Area of the contact line (1 if 0d, arc length if 1d, etc)
            self.add_integral_function("_ca_integral", acos(-dot(var("normal"), self.wall_tangent))*dx)
            self.add_dependent_integral_function("contact_angle",lambda _ca_integral,_cl_integral:_ca_integral/_cl_integral)
            if self.project_surface_tension:
                self.add_integral_function("_sigma_integral", var(self.surface_tension_name) * dx)
                self.add_dependent_integral_function("surface_tension",lambda _sigma_integral, _cl_integral: _sigma_integral / _cl_integral)
            self.model.add_additional_functions(self)


class SimplePopovContactLineEquations(ODEEquations):
    def __init__(self):
        super(SimplePopovContactLineEquations, self).__init__()
        self._on_mesh:Optional[ODEStorageMesh]=None

    def _init_output(self,eqtree:EquationTree,continue_info:Optional[Dict[str,Any]],rank:int): # Not for the output, but used to store the mesh information here
        super()._init_output(eqtree,continue_info,rank)
        assert isinstance(eqtree._mesh,ODEStorageMesh) 
        self._on_mesh = eqtree._mesh 