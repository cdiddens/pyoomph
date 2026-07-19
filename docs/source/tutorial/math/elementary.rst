Elementary functions
--------------------

The following elementary mathematical functions are implemented and work on scalar expressions and numbers.

.. list-table:: Elementary mathematical functions
    :widths: 50 50
    :header-rows: 0

    *   - ``square_root(x,[d=2])``
        - :math:`d`-th root :math:`\sqrt[d]{x}`
    *   - ``exp(x)``
        - Exponential function :math:`\exp(x)`
    *   - ``log(x)``
        - Natural logarithm :math:`\log(x)`
    *   - ``sin(x)``
        - Sine :math:`\sin(x)`
    *   - ``cos(x)``
        - Cosine :math:`\cos(x)`
    *   - ``tan(x)``
        - Tangent :math:`\tan(x)`          
    *   - ``asin(x)``
        - Inverse sine :math:`\operatorname{asin}(x)`
    *   - ``acos(x)``
        - Inverse cosine :math:`\operatorname{acos}(x)` 
    *   - ``atan(x)``
        - Inverse tangent :math:`\operatorname{atan}(x)`
    *   - ``atan2(y,x)``
        - Inverse tangent with case distinguishment :math:`\operatorname{atan2}(y,x)`        
    *   - ``sinh(x)``
        - Hyperbolic sine :math:`\sinh(x)`
    *   - ``cosh(x)``
        - Hyperbolic cosine :math:`\cosh(x)`
    *   - ``tanh(x)``        
        - Hyperbolic tangent :math:`\tanh(x)`

Further functions can be implemented using the :py:class:`~pyoomph.expressions.cb.CustomMathExpression` class from the module :py:mod:`pyoomph.expressions.cb`, see :numref:`sectemporalcustommath`.
