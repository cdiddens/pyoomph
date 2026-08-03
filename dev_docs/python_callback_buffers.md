# Buffers handed to Python callbacks

A `CustomMultiReturnExpression` implemented in Python only - no `generate_c_code()` - returned
**zero** for every result and every derivative, in every problem, silently. The generated element
code called it, Python computed the right numbers, and none of them came back.

## What happened

The generated code calls such a callback through `my_func_table->invoke_multi_ret`, which lands in
`CustomMultiReturnExpression::_call` (`src/nanobind/expressions.cpp`). That copies the arguments into
a `std::vector<double>` it owns, wraps its argument, result and derivative vectors as numpy arrays,
calls the Python `eval()`, and copies the result vector back out to the caller's buffer.

The wrapping was

```cpp
nb::ndarray<nb::numpy, double> resview(resbuffer.data(), {resbuffer.size()});
```

which has no owner. nanobind casts a trampoline's arguments under `rv_policy::automatic_reference`,
and `ndarray_export()` (`nanobind/src/nb_ndarray.cpp`) decides

```cpp
case rv_policy::automatic:
case rv_policy::automatic_reference:
    copy = th->owner == nullptr && th->self == nullptr;
```

so Python was handed a **copy**. `eval()` filled the copy; `resbuffer` stayed as it was; `_call`
copied those zeros into the caller's array. For the argument buffer a copy is harmless, which is why
the callback saw correct inputs and the failure looked like a broken implementation rather than
broken marshalling.

Every other `nb::ndarray` in the binding layer already passes an owner (`src/nanobind/mesh.cpp` -
`wown`, `pown`, `owner`, `howner`, ...); only these three did not. The fix is
`callback_buffer_view()`, which attaches a capsule with an empty deleter - the buffer belongs to the
C++ object, so the capsule exists only to make `th->owner` non-null.

## How it presented

It was found from a wrong drag coefficient in the confined-cylinder tutorial, which is worth
recording because the route from cause to symptom is long and every step of it was silent:

1. The tutorial reads the polymer stress on the cylinder as `var("polymer_stress", domain="..")`, a
   field defined by substitution.
2. In the log-conformation formulation that substitution contains `SymmetricMatrixExponential`, a
   multi-return callback. The instance embedded in the substitution is created during
   `define_fields`, before any residual is assembled, and is not among the ones the bulk residual
   registers - so the interface code has no C implementation for it and dispatches to Python.
3. Python returned zeros, i.e. **C = 0**, so the polymer stress evaluated to
   `(eta_p/lambda)*(0 - I)`: a constant isotropic tensor.
4. A constant isotropic stress contributes `-2*c*n_x` to the traction, and the integral of `n_x` over
   the half cylinder is exactly zero. The polymer term therefore contributed **exactly** `0.0000000`
   instead of `16.3775348356`, and the drag came out 113.99 rather than 130.36 - low by a clean
   12.6%, with no NaN, no warning and a perfectly smooth flow field.

Everything else in that script reads the polymer stress from the bulk, where the same callback is
inlined as C, and was correct throughout. Nothing about the failure pointed at the binding layer.

## Tests

`tests/test_custom_callbacks.py`. The same function is implemented twice, once in Python only and
once with `generate_c_code()`, and the two are required to agree:

- integrals of `2x` and `x^2` over the unit square, which a lost result buffer turns into zero;
- a Newton solve of `u^2 = 4` written through the callback, which a lost derivative buffer turns
  into an exactly singular Jacobian - it fails with a Pardiso zero pivot rather than converging
  slowly.

With the fix reverted, 2 of the 3 fail; the C-code control passes throughout.

## For the next binding

`nb::ndarray` over memory the C++ side owns needs an owner whenever Python may write to it. Without
one the array is a copy under the default policy, and a copy is indistinguishable from a view until
somebody reads the buffer back.
