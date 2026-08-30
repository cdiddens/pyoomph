.. _secelementaryfuncs:

Elementary functions
--------------------

The following elementary mathematical functions are implemented and work on scalar expressions and numbers. The constant ``pi`` is available as well.

The transcendental functions require a **dimensionless** argument, since e.g. :math:`\sin(x)` is only meaningful for a dimensionless :math:`x`. Divide by a scale first, i.e. ``sin(2*pi*x/(1*meter))`` instead of ``sin(2*pi*x)``. The functions of the second and third table below are the exception: they either pass the unit of their argument on to the result or, like :py:func:`~pyoomph.expressions.signum`, only look at its sign.

.. list-table:: Elementary mathematical functions
    :widths: 50 50
    :header-rows: 0

    *   - :py:func:`square_root(x,[order=2]) <pyoomph.expressions.square_root>`
        - :math:`\text{order}`-th root :math:`\sqrt[\text{order}]{x}`. The only one of these that takes a dimensional argument: the unit is raised to the power :math:`1/\text{order}` along with it
    *   - :py:func:`exp(x) <pyoomph.expressions.exp>`
        - Exponential function :math:`\exp(x)`
    *   - :py:func:`log(x) <pyoomph.expressions.log>`
        - Natural logarithm :math:`\log(x)`
    *   - :py:func:`sin(x) <pyoomph.expressions.sin>`
        - Sine :math:`\sin(x)`
    *   - :py:func:`cos(x) <pyoomph.expressions.cos>`
        - Cosine :math:`\cos(x)`
    *   - :py:func:`tan(x) <pyoomph.expressions.tan>`
        - Tangent :math:`\tan(x)`
    *   - :py:func:`asin(x) <pyoomph.expressions.asin>`
        - Inverse sine :math:`\operatorname{asin}(x)`
    *   - :py:func:`acos(x) <pyoomph.expressions.acos>`
        - Inverse cosine :math:`\operatorname{acos}(x)`
    *   - :py:func:`atan(x) <pyoomph.expressions.atan>`
        - Inverse tangent :math:`\operatorname{atan}(x)`
    *   - :py:func:`atan2(y,x) <pyoomph.expressions.atan2>`
        - Inverse tangent with case distinguishment :math:`\operatorname{atan2}(y,x)`. A shared unit of the two arguments is *not* divided out automatically, so pass ``atan2(y/(1*meter),x/(1*meter))``
    *   - :py:func:`sinh(x) <pyoomph.expressions.sinh>`
        - Hyperbolic sine :math:`\sinh(x)`
    *   - :py:func:`cosh(x) <pyoomph.expressions.cosh>`
        - Hyperbolic cosine :math:`\cosh(x)`
    *   - :py:func:`tanh(x) <pyoomph.expressions.tanh>`
        - Hyperbolic tangent :math:`\tanh(x)`
    *   - :py:func:`asinh(x) <pyoomph.expressions.asinh>`
        - Inverse hyperbolic sine :math:`\operatorname{asinh}(x)`
    *   - :py:func:`acosh(x) <pyoomph.expressions.acosh>`
        - Inverse hyperbolic cosine :math:`\operatorname{acosh}(x)`, real-valued for :math:`x\geq 1`
    *   - :py:func:`atanh(x) <pyoomph.expressions.atanh>`
        - Inverse hyperbolic tangent :math:`\operatorname{atanh}(x)`, real-valued for :math:`|x|<1`
    *   - :py:func:`erf(x) <pyoomph.expressions.erf>`
        - Error function :math:`\operatorname{erf}(x)`
    *   - :py:func:`erfc(x) <pyoomph.expressions.erfc>`
        - Complementary error function :math:`\operatorname{erfc}(x)=1-\operatorname{erf}(x)`, which unlike the latter stays accurate for large :math:`x`

The next functions are not smooth. They may be used in residuals nonetheless, but mind what their derivative is, since that is what the Newton solver sees: :py:func:`~pyoomph.expressions.heaviside` and :py:func:`~pyoomph.expressions.signum` deliberately differentiate to zero everywhere, i.e. the jump does not contribute a delta function to the Jacobian, and the branching functions differentiate branch-wise.

.. list-table:: Non-smooth and branching functions
    :widths: 50 50
    :header-rows: 0

    *   - :py:func:`absolute(x) <pyoomph.expressions.absolute>`
        - Absolute value :math:`|x|`, keeping the unit of :math:`x`. Differentiates as :math:`\operatorname{sign}(x)\,\mathrm{d}x`
    *   - :py:func:`signum(x) <pyoomph.expressions.signum>`
        - Sign of :math:`x`, i.e. :math:`\pm 1` and exactly :math:`0` at :math:`x=0`. Takes any unit, returns a dimensionless result
    *   - :py:func:`heaviside(x) <pyoomph.expressions.heaviside>`
        - Step function, i.e. :math:`1` for :math:`x>0`, :math:`0` for :math:`x<0` and :math:`1/2` at :math:`x=0`. Takes any unit, returns a dimensionless result
    *   - :py:func:`maximum(x,y) <pyoomph.expressions.maximum>`
        - :math:`\max(x,y)`. Both arguments must agree in units, which are then the units of the result
    *   - :py:func:`minimum(x,y) <pyoomph.expressions.minimum>`
        - :math:`\min(x,y)`, with the same unit rule as :py:func:`~pyoomph.expressions.maximum`
    *   - :py:func:`piecewise_geq0(cond,iftrue,iffalse) <pyoomph.expressions.piecewise_geq0>`
        - ``iftrue`` if :math:`\text{cond}\geq 0`, else ``iffalse``. The unit of ``cond`` is arbitrary, as only its sign matters, whereas the two branches must agree in units

For :py:func:`~pyoomph.expressions.maximum`, :py:func:`~pyoomph.expressions.minimum` and :py:func:`~pyoomph.expressions.piecewise_geq0`, a plain ``0`` is accepted as an argument irrespective of the unit of the other one, exactly as it is in a sum.

Residuals must be real-valued, but complex expressions are useful to formulate them, e.g. for a Helmholtz problem or for an eigenmode. The following functions split such an expression into its real and imaginary part, both of which keep the unit of the argument:

.. list-table:: Complex-valued expressions
    :widths: 50 50
    :header-rows: 0

    *   - :py:func:`imaginary_i() <pyoomph.expressions.imaginary_i>`
        - The imaginary unit :math:`i`. All fields themselves are considered real-valued, i.e. only an explicit :math:`i` makes a term imaginary
    *   - :py:func:`real_part(x) <pyoomph.expressions.real_part>`
        - Real part :math:`\Re(x)`
    *   - :py:func:`imag_part(x) <pyoomph.expressions.imag_part>`
        - Imaginary part :math:`\Im(x)`

Further functions can be implemented using the :py:class:`~pyoomph.expressions.cb.CustomMathExpression` class from the module :py:mod:`pyoomph.expressions.cb`, see :numref:`sectemporalcustommath`.
