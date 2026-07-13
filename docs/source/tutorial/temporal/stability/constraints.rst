Stability of second order ODEs or in the presence of constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us revert back to the pendulum equation of :numref:`secODEpendulum`. There, the dynamics has been implemented in two ways, namely (i) by solving the second order ODE

.. math:: \partial_t^2\phi+\frac{g}{L}\sin(\phi)=0\,,

or alternatively the first order ODE system with a Lagrange multiplier :math:`\lambda` enforcing the pendulum constraint

.. math::

   \begin{aligned}
   m\partial_t w&=-\lambda \frac{x}{\sqrt{x^2+y^2}}\\
   m\partial_t z&=-mg-\lambda \frac{y}{\sqrt{x^2+y^2}}\\
   \partial_t x&=w\\
   \partial_t y&=z\\
   0&=\sqrt{x^2+y^2}-L\,.
   \end{aligned}

Both variants are somewhat special regarding the calculation of the stability, i.e. of the eigenvalues. The first one is a second order ODE, for which pyoomph cannot calculate the eigenvalues directly. Hence, we have to cast it to a first order system as usual:

.. code:: python

   from pyoomph import * # Import pyoomph 
   from pyoomph.expressions import * # Import some additional things to express e.g. partial_t

   class PendulumEquations(ODEEquations):
       # Lets assume g=L=1
       def define_fields(self):
           self.define_ode_variable("phi","psi") # angle and angular velocity

       def define_residuals(self):
           phi,phi_test=var_and_test("phi")
           psi, psi_test = var_and_test("psi")
           self.add_residual((partial_t(psi)+sin(phi))*phi_test) # psi'=phi''=-sin(phi)
           self.add_residual((partial_t(phi)-psi) * psi_test) # psi=dot(phi)


   class PendulumProblem(Problem):
       def define_problem(self):
           eqs=PendulumEquations() #No output or initial condition required
           self.add_equations(eqs@"pendulum")

       # A function to investigate the stability of solutions
       def investigate_stability_close_to(self,phi_guess):
           ode=self.get_ode("pendulum")
           ode.set_value(phi=phi_guess,psi=0) # set the guess
           self.solve() # stationary solve
           eigvals,eigvects=self.solve_eigenproblem(2) # get eigenvectors
           phi_in_terms_of_pi=(ode.get_value("phi")/pi).evalf() # output phi as multiple as pi
           print(f"Eigenvalues at phi={phi_in_terms_of_pi}*pi are {eigvals[0]} and {eigvals[1]}")

   if __name__=="__main__":
       with PendulumProblem() as problem:
           problem.quiet()
           problem.investigate_stability_close_to(phi_guess=0.01) # pendulum is almost hanging straight down
           problem.investigate_stability_close_to(phi_guess=0.9*pi)  # pendulum is almost at the apex

Indeed, the result is expected: A marginally stable solution at :math:`\phi=0` with imaginary eigenvalues :math:`\pm i`, i.e. an undamped oscillatory motion and an unstable solution at :math:`\phi=\pi`. Since there are two degrees of freedom, we solve for :math:`2` eigenvalues. Furthermore, note the method ``evalf`` applied to :math:`\phi/\pi` to express :math:`\pi` in terms of :math:`\pi`. Without ``evalf``, :math:`\pi` is treated as a constant. Also, we have added some custom functionality to our problem by providing the method ``investigate_stability_close_to``.



.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <pendulum_gencoord_eigenvalues.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		

Now, let's turn towards the system with the Lagrange multiplier. Naively, one might anticipate that we can find :math:`5` eigenvalues, since we have :math:`5` degrees of freedom. Let's see:

.. code:: python

   #We can reuse the old class, since it is already a system of first order ODEs
   from pendulum_lagrange_multiplier import *


   if __name__=="__main__":
   	with PendulumProblem() as problem:
   		problem.quiet()
   		ode = problem.get_ode("pendulum")

   		problem.solve(timestep=0.001) #Make one little time step to find a good guess for lambda
   		problem.solve()
   		x,y,lambd,xdot,ydot=ode.get_value(["x","y","lambda_pendulum","xdot","ydot"])
   		print(f"Solution: x={x}, y={y}, lambda={lambd}, xdot={xdot}, ydot={ydot}")
   		eig_vals,eig_vects=problem.solve_eigenproblem(5)
   		print(eig_vals)


   		# Just flip the solution upside down
   		ode.set_value(x=0.0,y=-y,lambda_pendulum=-lambd) # We also need to flip the rod tension lambda
   		problem.solve() # We can still solve
   		x, y, lambd, xdot, ydot = ode.get_value(["x", "y", "lambda_pendulum", "xdot", "ydot"])
   		print(f"Solution: x={x}, y={y}, lambda={lambd}, xdot={xdot}, ydot={ydot}")
   		eig_vals, eig_vects = problem.solve_eigenproblem(5)
   		print(eig_vals)

First, we use the fact that our system in :numref:`secODEpendulum` was already converted into first order form. Hence, we can just import the previous classes from :download:`pendulum_lagrange_multiplier.py <../pendulum_lagrange_multiplier.py>` and reuse them directly. However, the initial guess of :math:`\lambda` is not good and the problem might not converge with a simple :py:meth:`~pyoomph.generic.problem.Problem.solve` command. This can be done by performing a single short time step. Within a time step, the :math:`\partial_t`-terms ensure that the degrees :math:`x` and :math:`y` do not change too much within the small time interval. Thereby, :math:`\lambda` can relax better to a value which is close to the stationary solution near the top. Then, we solve for the stationary solution, get up to :math:`5` eigenvalues and finally we flip the pendulum upside down and repeat this. Note that :math:`\lambda` has also been flipped, since it represents the normal force stemming from the rod of the pendulum. If it is at the top, we need a pushing force, if it is at the bottom, we need the same pulling force.

.. only:: html

	.. container:: downloadbutton

		:download:`Download this example <pendulum_lagrange_eigenvalues.py>`
		
		:download:`Download all examples <../../tutorial_example_scripts.zip>`   	
		

While everything runs smoothly, instead of :math:`5` eigenvalues, only :math:`2` are returned, which are exactly the same as the ones in the simple system for the angle :math:`\phi` before. To see why this is the case and what is actually going on in pyoomph, we can go through the calculation of the eigenvalues analytically. First, the system is written as

.. math:: \mathbf{M}\partial_t \vec{U}=\vec{F}\left(\vec{U}\right)

where :math:`\vec{U}=(w,z,x,y,\lambda)` is the vector of unknowns and

.. math:: \vec{F}(\vec{U})=\begin{pmatrix} -\lambda \frac{x}{\sqrt{x^2+y^2}} \\ -mg-\lambda \frac{y}{\sqrt{x^2+y^2}}\\ w \\ z \\ \sqrt{x^2+y^2}-L  \end{pmatrix}

is the right hand side. The matrix :math:`\mathbf{M}` is called *mass matrix* and it reads

.. math:: \mathbf{M}=\begin{pmatrix} m & 0 & 0 & 0 &0 \\ 0 & m & 0 &0 &0 \\ 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 \end{pmatrix}

This matrix reflects the fact that in particular for the equation of the constraint no time derivatives are present. The stationary solutions :math:`\vec{U}_0` for the parameters :math:`m=L=g=1` read

.. math:: \vec{U}_0^{+}=\begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \\ -1 \end{pmatrix} \quad \text{and} \quad \vec{U}_0^{-}=\begin{pmatrix} 0 \\ 0 \\ 0 \\ -1 \\ 1 \end{pmatrix}

To determine the eigenvalues, the right hand side :math:`\vec{F}(\vec{U})` is linearized around :math:`\vec{U}_0^{\pm}`, which gives the *Jacobian*

.. math:: \mathbf{J}^{\pm}=\left.\frac{\partial \vec{F}}{\partial \vec{U}}\right|_{\vec{U}_0^{\pm}}=\begin{pmatrix}0 & 0 & \pm 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & \mp 1 \\ 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & \pm 1 & 0\end{pmatrix}

As usual in linear stability analysis, we consider a perturbed solution :math:`\vec{U}^\pm=\vec{U}_0^\pm+\delta\vec{U}^\pm`. The small perturbation :math:`\delta\vec{U}^\pm` hence evolves according to

.. math:: \mathbf{M}\partial_t\delta\vec{U}^\pm=\mathbf{J}^\pm \delta\vec{U}^\pm

And finally, given the linearity of this system, the ansatz :math:`\vec{U}^\pm=\vec{v}\exp(\mu t)` is chosen, where :math:`\mu` and :math:`\vec{v}` are the eigenvalue-eigenvector pair we want to solve for. This gives rise to the *generalized eigenvalue problem*

.. math:: \mu\mathbf{M}\vec{v}=\mathbf{J} \vec{v}\,,

where the indices :math:`\pm` were omitted for brevity. This equation is exactly the one which is solved within pyoomph, whenever :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem` is called. The mass matrix :math:`\mathbf{M}` is calculated based on the occurrences of the :py:func:`~pyoomph.expressions.generic.partial_t`-expressions in the added residuals and the Jacobian :math:`\mathbf{J}` is analytically calculated and numerically evaluated at the stationary solution.

.. warning::

   When solving for the generalized eigenproblem, most solvers internally used require that the mass matrix :math:`\mathbf{M}` is positive (semi-)definite or at least that the diagonal entries are positive. This means that you always should implement your residuals in such a way that the sign of the :py:func:`~pyoomph.expressions.generic.partial_t` terms is positive. Therefore, one should prefer

   .. container:: center

      ``add_residual((partial_t(u)-rhs))*testfunction(u))``

   over

   .. container:: center

      ``add_residual((rhs-partial_t(u)))*testfunction(u))``

Analogous to the conventional eigenvalue problem, which arises for the case of :math:`\mathbf{M}=\mathbf{1}`, we demand that

.. math:: \det\left|\mathbf{J}-\mu\mathbf{M}\right|=0

In fact, if we calculate this determinant, the characteristic polynomial is indeed only of second order, not of order :math:`5`. We get

.. math:: \det\left|\mathbf{J}^\pm-\mu\mathbf{M}\right|=-\mu^2\pm 1

which has exactly the pairs of solutions for the eigenvalues :math:`\mu` we got from the numerical calculation via pyoomph, namely :math:`\mu=\pm 1` for :math:`\vec{U}_0^+`, i.e. the pendulum at the apex and :math:`\mu=\pm i` for :math:`\vec{U}_0^-`, i.e. the pendulum at the equilibrium position at the bottom.

Obviously, in generalized eigenvalue problems, it is not true that the sum of the *algebraic multiplicity* of each eigenvalue yields the dimension of the matrix (here :math:`5`). Instead, we can get fewer, so that the dynamics in the system with the constraint enforced by the Lagrange multiplier exactly resembles the dynamics of the simple system expressed in the generalized coordinate :math:`\phi`.

In pyoomph, both situations are well treated by the method :py:meth:`~pyoomph.generic.problem.Problem.solve_eigenproblem`.
