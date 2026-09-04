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
    *   - :py:func:`piecewise_gt0(cond,iftrue,iffalse) <pyoomph.expressions.piecewise_gt0>`
        - The strict counterpart, i.e. ``iftrue`` only if :math:`\text{cond}> 0`. Both differ only for :math:`\text{cond}=0`
    *   - :py:func:`conditional(cond,iftrue,iffalse) <pyoomph.expressions.conditional>`
        - ``iftrue`` if the comparison ``cond`` holds, else ``iffalse``, with the same unit rule for the branches

For :py:func:`~pyoomph.expressions.maximum`, :py:func:`~pyoomph.expressions.minimum`, :py:func:`~pyoomph.expressions.piecewise_geq0` and :py:func:`~pyoomph.expressions.piecewise_gt0`, a plain ``0`` is accepted as an argument irrespective of the unit of the other one, exactly as it is in a sum.

Instead of comparing against zero by hand, one can write the condition of a :py:func:`~pyoomph.expressions.conditional` as an ordinary comparison with ``<``, ``<=``, ``>`` or ``>=``. Both sides must have the same unit, but they may depend on fields, on the time or on global parameters, i.e. the comparison is kept symbolically and is only evaluated in the generated code:

.. code:: python

   expression = conditional(var("time") < 2*second, 1*meter, 2*meter)

Such an expression can also be evaluated directly by substituting the relevant quantities, e.g. ``expression(time=4*second)`` gives ``2*meter`` here. Note that Python's own ternary ``iftrue if cond else iffalse`` cannot be used for this, since Python insists on casting the condition to a plain ``bool``, which would select one branch immediately. A comparison that *is* numerically decidable, on the other hand, still converts to a ``bool``, i.e. it may be used in an ``if`` statement as usual.

Conditions can be combined, but again not with Python's ``not``, ``and`` and ``or``, which cast their operands to a ``bool`` as well. The bitwise operators ``~``, ``&``, ``|`` and ``^`` take their place:

.. code:: python

   inside = conditional((var("time") > 1*second) & (var("time") < 2*second), 1*meter, 2*meter)

Mind that ``~``, ``&``, ``|`` and ``^`` bind *tighter* than the comparisons, i.e. every operand needs its own parentheses. The named :py:func:`~pyoomph.expressions.logical_not`, :py:func:`~pyoomph.expressions.logical_and`, :py:func:`~pyoomph.expressions.logical_or` and :py:func:`~pyoomph.expressions.logical_xor` are free of that pitfall and take any number of operands. A chained comparison like ``a < b < c`` cannot be used either, since Python expands it into ``(a<b) and (b<c)``; write ``(a<b) & (b<c)`` instead. All of these mistakes end up casting a condition to a ``bool`` and are therefore reported rather than silently mis-evaluated.

Internally, each comparison of a combined condition is evaluated to an indicator that is :math:`1` where it holds and :math:`0` elsewhere, and those are combined arithmetically (:math:`1-p`, :math:`pq`, :math:`p+q-pq` and :math:`p+q-2pq`), so that the whole condition remains a single sign test in the generated code. Only these cheap comparisons are combined that way - the two branch values stay a proper ternary and are never both evaluated, i.e. a branch that is invalid on the other side of the condition, such as a square root of a negative number, is still never reached.

Only these four inequalities are symbolic. ``==`` and ``!=`` are left alone deliberately, since Python requires them to return a plain ``bool`` for expressions to stay usable in dictionaries, sets and ``in`` tests. To compare two expressions, use ``a.is_equal(b)``, which tests whether both are structurally identical, or ``(a-b).is_zero()``, which simplifies the difference first.

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
