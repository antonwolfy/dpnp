# cython: language_level=3
# distutils: language = c++
# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2016-2024, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
# *****************************************************************************

"""
Interface of the Logic part of the DPNP

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

# pylint: disable=protected-access
# pylint: disable=c-extension-no-member
# pylint: disable=duplicate-code
# pylint: disable=no-name-in-module


import dpctl.tensor as dpt
import dpctl.tensor._tensor_elementwise_impl as tei
import numpy

import dpnp
from dpnp.dpnp_algo import dpnp_allclose
from dpnp.dpnp_algo.dpnp_elementwise_common import DPNPBinaryFunc, DPNPUnaryFunc
from dpnp.dpnp_array import dpnp_array
from dpnp.dpnp_utils import call_origin

__all__ = [
    "all",
    "allclose",
    "any",
    "equal",
    "greater",
    "greater_equal",
    "isclose",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "less",
    "less_equal",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "not_equal",
]


def all(a, /, axis=None, out=None, keepdims=False, *, where=True):
    """
    Test whether all array elements along a given axis evaluate to True.

    For full documentation refer to :obj:`numpy.all`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which a logical AND reduction is performed.
        The default is to perform a logical AND over all the dimensions
        of the input array.`axis` may be negative, in which case it counts
        from the last to the first axis.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the returned
        values) will be cast if necessary.
        Default: ``None``.
    keepdims : bool, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        An array with a data type of `bool`
        containing the results of the logical AND reduction is returned
        unless `out` is specified. Otherwise, a reference to `out` is returned.
        The result has the same shape as `a` if `axis` is not ``None``
        or `a` is a 0-d array.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.ndarray.all` : equivalent method
    :obj:`dpnp.any` : Test whether any element along a given axis evaluates
                      to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity
    evaluate to ``True`` because these are not equal to zero.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[True, False], [True, True]])
    >>> np.all(x)
    array(False)

    >>> np.all(x, axis=0)
    array([ True, False])

    >>> x2 = np.array([-1, 4, 5])
    >>> np.all(x2)
    array(True)

    >>> x3 = np.array([1.0, np.nan])
    >>> np.all(x3)
    array(True)

    >>> o = np.array(False)
    >>> z = np.all(x2, out=o)
    >>> z, o
    (array(True), array(True))
    >>> # Check now that `z` is a reference to `o`
    >>> z is o
    True
    >>> id(z), id(o) # identity of `z` and `o`
    (139884456208480, 139884456208480) # may vary

    """

    dpnp.check_limitations(where=where)

    dpt_array = dpnp.get_usm_ndarray(a)
    result = dpnp_array._create_from_usm_ndarray(
        dpt.all(dpt_array, axis=axis, keepdims=keepdims)
    )
    # TODO: temporary solution until dpt.all supports out parameter
    result = dpnp.get_result_array(result, out)
    return result


def allclose(a, b, rtol=1.0e-5, atol=1.0e-8, **kwargs):
    """
    Returns ``True`` if two arrays are element-wise equal within a tolerance.

    The tolerance values are positive, typically very small numbers. The
    relative difference (`rtol` * abs(`b`)) and the absolute difference `atol`
    are added together to compare against the absolute difference between `a`
    and `b`.

    If either array contains one or more ``NaNs``, ``False`` is returned.
    ``Infs`` are treated as equal if they are in the same place and of the same
    sign in both arrays.

    For full documentation refer to :obj:`numpy.allclose`.

    Returns
    -------
    out : dpnp.ndarray
        A 0-dim array with ``True`` value if the two arrays are equal within
        the given tolerance; with ``False`` otherwise.

    Limitations
    -----------
    Parameters `a` and `b` are supported either as :class:`dpnp.ndarray`,
    :class:`dpctl.tensor.usm_ndarray` or scalars, but both `a` and `b`
    can not be scalars at the same time.
    Keyword argument `kwargs` is currently unsupported.
    Otherwise the functions will be executed sequentially on CPU.
    Parameters `rtol` and `atol` are supported as scalars. Otherwise
    ``TypeError`` exception will be raised.
    Input array data types are limited by supported integer and
    floating DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.isclose` : Test whether two arrays are element-wise equal.
    :obj:`dpnp.all` : Test whether all elements evaluate to True.
    :obj:`dpnp.any` : Test whether any element evaluates to True.
    :obj:`dpnp.equal` : Return (x1 == x2) element-wise.

    Examples
    --------
    >>> import dpnp as np
    >>> a = np.array([1e10, 1e-7])
    >>> b = np.array([1.00001e10, 1e-8])
    >>> np.allclose(a, b)
    array([False])

    >>> a = np.array([1.0, np.nan])
    >>> b = np.array([1.0, np.nan])
    >>> np.allclose(a, b)
    array([False])

    >>> a = np.array([1.0, np.inf])
    >>> b = np.array([1.0, np.inf])
    >>> np.allclose(a, b)
    array([ True])

    """

    if dpnp.isscalar(a) and dpnp.isscalar(b):
        # at least one of inputs has to be an array
        pass
    elif not (
        dpnp.is_supported_array_or_scalar(a)
        and dpnp.is_supported_array_or_scalar(b)
    ):
        pass
    elif kwargs:
        pass
    else:
        if not dpnp.isscalar(rtol):
            raise TypeError(
                f"An argument `rtol` must be a scalar, but got {rtol}"
            )
        if not dpnp.isscalar(atol):
            raise TypeError(
                f"An argument `atol` must be a scalar, but got {atol}"
            )

        if dpnp.isscalar(a):
            a = dpnp.full_like(b, fill_value=a)
        elif dpnp.isscalar(b):
            b = dpnp.full_like(a, fill_value=b)
        elif a.shape != b.shape:
            a, b = dpt.broadcast_arrays(a.get_array(), b.get_array())

        a_desc = dpnp.get_dpnp_descriptor(a, copy_when_nondefault_queue=False)
        b_desc = dpnp.get_dpnp_descriptor(b, copy_when_nondefault_queue=False)
        if a_desc and b_desc:
            return dpnp_allclose(a_desc, b_desc, rtol, atol).get_pyobj()

    return call_origin(numpy.allclose, a, b, rtol=rtol, atol=atol, **kwargs)


def any(a, /, axis=None, out=None, keepdims=False, *, where=True):
    """
    Test whether any array element along a given axis evaluates to True.

    For full documentation refer to :obj:`numpy.any`.

    Parameters
    ----------
    a : {dpnp.ndarray, usm_ndarray}
        Input array.
    axis : {None, int, tuple of ints}, optional
        Axis or axes along which a logical OR reduction is performed.
        The default is to perform a logical OR over all the dimensions
        of the input array.`axis` may be negative, in which case it counts
        from the last to the first axis.
        Default: ``None``.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the returned
        values) will be cast if necessary.
        Default: ``None``.
    keepdims : bool, optional
        If ``True``, the reduced axes (dimensions) are included in the result
        as singleton dimensions, so that the returned array remains
        compatible with the input array according to Array Broadcasting
        rules. Otherwise, if ``False``, the reduced axes are not included in
        the returned array.
        Default: ``False``.

    Returns
    -------
    out : dpnp.ndarray
        An array with a data type of `bool`
        containing the results of the logical OR reduction is returned
        unless `out` is specified. Otherwise, a reference to `out` is returned.
        The result has the same shape as `a` if `axis` is not ``None``
        or `a` is a 0-d array.

    Limitations
    -----------
    Parameters `where` is only supported with its default value.
    Otherwise ``NotImplementedError`` exception will be raised.

    See Also
    --------
    :obj:`dpnp.ndarray.any` : equivalent method
    :obj:`dpnp.all` : Test whether all elements along a given axis evaluate
                      to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity evaluate
    to ``True`` because these are not equal to zero.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array([[True, False], [True, True]])
    >>> np.any(x)
    array(True)

    >>> np.any(x, axis=0)
    array([ True,  True])

    >>> x2 = np.array([-1, 0, 5])
    >>> np.any(x2)
    array(True)

    >>> x3 = np.array([1.0, np.nan])
    >>> np.any(x3)
    array(True)

    >>> o = np.array(False)
    >>> z = np.any(x2, out=o)
    >>> z, o
    (array(True), array(True))
    >>> # Check now that `z` is a reference to `o`
    >>> z is o
    True
    >>> id(z), id(o) # identity of `z` and `o`
    >>> (140053638309840, 140053638309840) # may vary

    """

    dpnp.check_limitations(where=where)

    dpt_array = dpnp.get_usm_ndarray(a)
    result = dpnp_array._create_from_usm_ndarray(
        dpt.any(dpt_array, axis=axis, keepdims=keepdims)
    )
    # TODO: temporary solution until dpt.any supports out parameter
    result = dpnp.get_result_array(result, out)
    return result


_EQUAL_DOCSTRING = """
Calculates equality test results for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.equal`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array have the correct shape and the expected data type.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the result of element-wise equality comparison.
    The returned array has a data type of `bool`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.
:obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
:obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
:obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
:obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([0, 1, 3])
>>> x2 = np.arange(3)
>>> np.equal(x1, x2)
array([ True,  True, False])

What is compared are values, not types. So an int (1) and an array of
length one can evaluate as True:

>>> np.equal(1, np.ones(1))
array([ True])

The ``==`` operator can be used as a shorthand for ``equal`` on
:class:`dpnp.ndarray`.

>>> a = np.array([2, 4, 6])
>>> b = np.array([2, 4, 2])
>>> a == b
array([ True,  True, False])
"""

equal = DPNPBinaryFunc(
    "equal",
    tei._equal_result_type,
    tei._equal,
    _EQUAL_DOCSTRING,
)


_GREATER_DOCSTRING = """
Computes the greater-than test results for each element `x1_i` of
the input array `x1` with the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.greater`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the result of element-wise greater-than comparison.
    The returned array has a data type of `bool`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
:obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
:obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
:obj:`dpnp.equal` : Return (x1 == x2) element-wise.
:obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([4, 2])
>>> x2 = np.array([2, 2])
>>> np.greater(x1, x2)
array([ True, False])

The ``>`` operator can be used as a shorthand for ``greater`` on
:class:`dpnp.ndarray`.

>>> a = np.array([4, 2])
>>> b = np.array([2, 2])
>>> a > b
array([ True, False])
"""

greater = DPNPBinaryFunc(
    "greater",
    tei._greater_result_type,
    tei._greater,
    _GREATER_DOCSTRING,
)


_GREATER_EQUAL_DOCSTRING = """
Computes the greater-than or equal-to test results for each element `x1_i` of
the input array `x1` with the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.greater_equal`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the result of element-wise greater-than or equal-to
    comparison.
    The returned array has a data type of `bool`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
:obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
:obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
:obj:`dpnp.equal` : Return (x1 == x2) element-wise.
:obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([4, 2, 1])
>>> x2 = np.array([2, 2, 2])
>>> np.greater_equal(x1, x2)
array([ True,  True, False])

The ``>=`` operator can be used as a shorthand for ``greater_equal`` on
:class:`dpnp.ndarray`.

>>> a = np.array([4, 2, 1])
>>> b = np.array([2, 2, 2])
>>> a >= b
array([ True,  True, False])
"""

greater_equal = DPNPBinaryFunc(
    "greater",
    tei._greater_equal_result_type,
    tei._greater_equal,
    _GREATER_EQUAL_DOCSTRING,
)


def isclose(x1, x2, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns a boolean array where two arrays are element-wise equal within
    a tolerance.

    For full documentation refer to :obj:`numpy.isclose`.

    Limitations
    -----------
    `x2` is supported to be integer if `x1` is :class:`dpnp.ndarray` or
    at least either `x1` or `x2` should be as :class:`dpnp.ndarray`.
    Otherwise the function will be executed sequentially on CPU.
    Input array data types are limited by supported DPNP :ref:`Data types`.

    See Also
    --------
    :obj:`dpnp.allclose` : Returns True if two arrays are element-wise equal
                           within a tolerance.

    Examples
    --------
    >>> import dpnp as np
    >>> x1 = np.array([1e10,1e-7])
    >>> x2 = np.array([1.00001e10,1e-8])
    >>> out = np.isclose(x1, x2)
    >>> [i for i in out]
    [True, False]

    """

    # x1_desc = dpnp.get_dpnp_descriptor(x1)
    # x2_desc = dpnp.get_dpnp_descriptor(x2)
    # if x1_desc and x2_desc:
    #     result_obj = dpnp_isclose(
    #         x1_desc, x2_desc, rtol, atol, equal_nan
    #     ).get_pyobj()
    #     return result_obj

    return call_origin(
        numpy.isclose, x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


_ISFINITE_DOCSTRING = """
Test if each element of input array is a finite number.

For full documentation refer to :obj:`numpy.isfinite`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array which is True where `x` is not positive infinity,
    negative infinity, or ``NaN``, False otherwise.
    The data type of the returned array is `bool`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.isinf` : Test element-wise for positive or negative infinity.
:obj:`dpnp.isneginf` : Test element-wise for negative infinity,
                        return result as bool array.
:obj:`dpnp.isposinf` : Test element-wise for positive infinity,
                        return result as bool array.
:obj:`dpnp.isnan` : Test element-wise for NaN and
                    return result as a boolean array.

Notes
-----
Not a Number, positive infinity and negative infinity are considered
to be non-finite.

Examples
--------
>>> import dpnp as np
>>> x = np.array([-np.inf, 0., np.inf])
>>> np.isfinite(x)
array([False,  True, False])
"""

isfinite = DPNPUnaryFunc(
    "isfinite",
    tei._isfinite_result_type,
    tei._isfinite,
    _ISFINITE_DOCSTRING,
)


_ISINF_DOCSTRING = """
Test if each element of input array is an infinity.

For full documentation refer to :obj:`numpy.isinf`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array which is True where `x` is positive or negative infinity,
    False otherwise. The data type of the returned array is `bool`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.isneginf` : Test element-wise for negative infinity,
                        return result as bool array.
:obj:`dpnp.isposinf` : Test element-wise for positive infinity,
                        return result as bool array.
:obj:`dpnp.isnan` : Test element-wise for NaN and
                    return result as a boolean array.
:obj:`dpnp.isfinite` : Test element-wise for finiteness.

Examples
--------
>>> import dpnp as np
>>> x = np.array([-np.inf, 0., np.inf])
>>> np.isinf(x)
array([ True, False,  True])
"""

isinf = DPNPUnaryFunc(
    "isinf",
    tei._isinf_result_type,
    tei._isinf,
    _ISINF_DOCSTRING,
)


_ISNAN_DOCSTRING = """
Test if each element of an input array is a NaN.

For full documentation refer to :obj:`numpy.isnan`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array, expected to have numeric data type.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array which is True where `x` is ``NaN``, False otherwise.
    The data type of the returned array is `bool`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.isinf` : Test element-wise for positive or negative infinity.
:obj:`dpnp.isneginf` : Test element-wise for negative infinity,
                        return result as bool array.
:obj:`dpnp.isposinf` : Test element-wise for positive infinity,
                        return result as bool array.
:obj:`dpnp.isfinite` : Test element-wise for finiteness.
:obj:`dpnp.isnat` : Test element-wise for NaT (not a time)
                    and return result as a boolean array.

Examples
--------
>>> import dpnp as np
>>> x = np.array([np.inf, 0., np.nan])
>>> np.isnan(x)
array([False, False,  True])
"""

isnan = DPNPUnaryFunc(
    "isnan",
    tei._isnan_result_type,
    tei._isnan,
    _ISNAN_DOCSTRING,
)


def isneginf(x, out=None):
    """
    Test element-wise for negative infinity, return result as bool array.

    For full documentation refer to :obj:`numpy.isneginf`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to and a boolean data type.
        If not provided or ``None``, a freshly-allocated boolean array
        is returned.
        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Boolean array of same shape as ``x``.

    See Also
    --------
    :obj:`dpnp.isinf` : Test element-wise for positive or negative infinity.
    :obj:`dpnp.isposinf` : Test element-wise for positive infinity,
                            return result as bool array.
    :obj:`dpnp.isnan` : Test element-wise for NaN and
                    return result as a boolean array.
    :obj:`dpnp.isfinite` : Test element-wise for finiteness.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array(np.inf)
    >>> np.isneginf(-x)
    array(True)
    >>> np.isneginf(x)
    array(False)

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> np.isneginf(x)
    array([ True, False, False])

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.zeros(x.shape, dtype='bool')
    >>> np.isneginf(x, y)
    array([ True, False, False])
    >>> y
    array([ True, False, False])

    """

    dpnp.check_supported_arrays_type(x)

    if out is not None:
        dpnp.check_supported_arrays_type(out)

    x_dtype = x.dtype
    if dpnp.issubdtype(x_dtype, dpnp.complexfloating):
        raise TypeError(
            f"This operation is not supported for {x_dtype} values "
            "because it would be ambiguous."
        )

    is_inf = dpnp.isinf(x)
    signbit = dpnp.signbit(x)

    # TODO: support different out dtype #1717(dpctl)
    return dpnp.logical_and(is_inf, signbit, out=out)


def isposinf(x, out=None):
    """
    Test element-wise for positive infinity, return result as bool array.

    For full documentation refer to :obj:`numpy.isposinf`.

    Parameters
    ----------
    x : {dpnp.ndarray, usm_ndarray}
        Input array.
    out : {None, dpnp.ndarray, usm_ndarray}, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to and a boolean data type.
        If not provided or ``None``, a freshly-allocated boolean array
        is returned.
        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray
        Boolean array of same shape as ``x``.

    See Also
    --------
    :obj:`dpnp.isinf` : Test element-wise for positive or negative infinity.
    :obj:`dpnp.isneginf` : Test element-wise for negative infinity,
                            return result as bool array.
    :obj:`dpnp.isnan` : Test element-wise for NaN and
                    return result as a boolean array.
    :obj:`dpnp.isfinite` : Test element-wise for finiteness.

    Examples
    --------
    >>> import dpnp as np
    >>> x = np.array(np.inf)
    >>> np.isposinf(x)
    array(True)
    >>> np.isposinf(-x)
    array(False)

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> np.isposinf(x)
    array([False, False,  True])

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.zeros(x.shape, dtype='bool')
    >>> np.isposinf(x, y)
    array([False, False,  True])
    >>> y
    array([False, False,  True])

    """

    dpnp.check_supported_arrays_type(x)

    if out is not None:
        dpnp.check_supported_arrays_type(out)

    x_dtype = x.dtype
    if dpnp.issubdtype(x_dtype, dpnp.complexfloating):
        raise TypeError(
            f"This operation is not supported for {x_dtype} values "
            "because it would be ambiguous."
        )

    is_inf = dpnp.isinf(x)
    signbit = ~dpnp.signbit(x)

    # TODO: support different out dtype #1717(dpctl)
    return dpnp.logical_and(is_inf, signbit, out=out)


_LESS_DOCSTRING = """
Computes the less-than test results for each element `x1_i` of
the input array `x1` with the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.less`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the result of element-wise less-than comparison.
    The returned array has a data type of `bool`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
:obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.
:obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
:obj:`dpnp.equal` : Return (x1 == x2) element-wise.
:obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([1, 2])
>>> x2 = np.array([2, 2])
>>> np.less(x1, x2)
array([ True, False])

The ``<`` operator can be used as a shorthand for ``less`` on
:class:`dpnp.ndarray`.

>>> a = np.array([1, 2])
>>> b = np.array([2, 2])
>>> a < b
array([ True, False])
"""

less = DPNPBinaryFunc(
    "less",
    tei._less_result_type,
    tei._less,
    _LESS_DOCSTRING,
)


_LESS_EQUAL_DOCSTRING = """
Computes the less-than or equal-to test results for each element `x1_i` of
the input array `x1` with the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.less_equal`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the result of element-wise less-than or equal-to
    comparison. The returned array has a data type of `bool`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
:obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
:obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
:obj:`dpnp.equal` : Return (x1 == x2) element-wise.
:obj:`dpnp.not_equal` : Return (x1 != x2) element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([4, 2, 1])
>>> x2 = np.array([2, 2, 2]
>>> np.less_equal(x1, x2)
array([False,  True,  True])

The ``<=`` operator can be used as a shorthand for ``less_equal`` on
:class:`dpnp.ndarray`.

>>> a = np.array([4, 2, 1])
>>> b = np.array([2, 2, 2])
>>> a <= b
array([False,  True,  True])
"""

less_equal = DPNPBinaryFunc(
    "less_equal",
    tei._less_equal_result_type,
    tei._less_equal,
    _LESS_EQUAL_DOCSTRING,
)


_LOGICAL_AND_DOCSTRING = """
Computes the logical AND for each element `x1_i` of the input array `x1` with
the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.logical_and`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise logical AND results.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logical_or` : Compute the truth value of x1 OR x2 element-wise.
:obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.
:obj:`dpnp.logical_xor` : Compute the truth value of x1 XOR x2, element-wise.
:obj:`dpnp.bitwise_and` : Compute the bit-wise AND of two arrays element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([True, False])
>>> x2 = np.array([False, False])
>>> np.logical_and(x1, x2)
array([False, False])

>>> x = np.arange(5)
>>> np.logical_and(x > 1, x < 4)
array([False, False,  True,  True, False])

The ``&`` operator can be used as a shorthand for ``logical_and`` on
boolean :class:`dpnp.ndarray`.

>>> a = np.array([True, False])
>>> b = np.array([False, False])
>>> a & b
array([False, False])
"""

logical_and = DPNPBinaryFunc(
    "logical_and",
    tei._logical_and_result_type,
    tei._logical_and,
    _LOGICAL_AND_DOCSTRING,
)


_LOGICAL_NOT_DOCSTRING = """
Computes the logical NOT for each element `x_i` of input array `x`.

For full documentation refer to :obj:`numpy.logical_not`.

Parameters
----------
x : {dpnp.ndarray, usm_ndarray}
    Input array.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise logical NOT results.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logical_and` : Compute the truth value of x1 AND x2 element-wise.
:obj:`dpnp.logical_or` : Compute the truth value of x1 OR x2 element-wise.
:obj:`dpnp.logical_xor` : Compute the truth value of x1 XOR x2, element-wise.

Examples
--------
>>> import dpnp as np
>>> x = np.array([True, False, 0, 1])
>>> np.logical_not(x)
array([False,  True,  True, False])

>>> x = np.arange(5)
>>> np.logical_not(x < 3)
array([False, False, False,  True,  True])
"""

logical_not = DPNPUnaryFunc(
    "logical_not",
    tei._logical_not_result_type,
    tei._logical_not,
    _LOGICAL_NOT_DOCSTRING,
)


_LOGICAL_OR_DOCSTRING = """
Computes the logical OR for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.logical_or`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise logical OR results.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logical_and` : Compute the truth value of x1 AND x2 element-wise.
:obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.
:obj:`dpnp.logical_xor` : Compute the truth value of x1 XOR x2, element-wise.
:obj:`dpnp.bitwise_or` : Compute the bit-wise OR of two arrays element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([True, False])
>>> x2 = np.array([False, False])
>>> np.logical_or(x1, x2)
array([ True, False])

>>> x = np.arange(5)
>>> np.logical_or(x < 1, x > 3)
array([ True, False, False, False,  True])

The ``|`` operator can be used as a shorthand for ``logical_or`` on
boolean :class:`dpnp.ndarray`.

>>> a = np.array([True, False])
>>> b = np.array([False, False])
>>> a | b
array([ True, False])
"""

logical_or = DPNPBinaryFunc(
    "logical_or",
    tei._logical_or_result_type,
    tei._logical_or,
    _LOGICAL_OR_DOCSTRING,
)


_LOGICAL_XOR_DOCSTRING = """
Computes the logical XOR for each element `x1_i` of the input array `x1`
with the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.logical_xor`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the element-wise logical XOR results.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.logical_and` : Compute the truth value of x1 AND x2 element-wise.
:obj:`dpnp.logical_or` : Compute the truth value of x1 OR x2 element-wise.
:obj:`dpnp.logical_not` : Compute the truth value of NOT x element-wise.
:obj:`dpnp.bitwise_xor` : Compute the bit-wise XOR of two arrays element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([True, True, False, False])
>>> x2 = np.array([True, False, True, False])
>>> np.logical_xor(x1, x2)
array([False,  True,  True, False])

>>> x = np.arange(5)
>>> np.logical_xor(x < 1, x > 3)
array([ True, False, False, False,  True])

Simple example showing support of broadcasting

>>> np.logical_xor(0, np.eye(2))
array([[ True, False],
       [False,  True]])
"""

logical_xor = DPNPBinaryFunc(
    "logical_xor",
    tei._logical_xor_result_type,
    tei._logical_xor,
    _LOGICAL_XOR_DOCSTRING,
)


_NOT_EQUAL_DOCSTRING = """
Calculates inequality test results for each element `x1_i` of the
input array `x1` with the respective element `x2_i` of the input array `x2`.

For full documentation refer to :obj:`numpy.not_equal`.

Parameters
----------
x1 : {dpnp.ndarray, usm_ndarray, scalar}
    First input array, expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
x2 : {dpnp.ndarray, usm_ndarray, scalar}
    Second input array, also expected to have numeric data type.
    Both inputs `x1` and `x2` can not be scalars at the same time.
out : {None, dpnp.ndarray, usm_ndarray}, optional
    Output array to populate.
    Array must have the correct shape and the expected data type.
    Default: ``None``.
order : {"C", "F", "A", "K"}, optional
    Memory layout of the newly output array, if parameter `out` is ``None``.
    Default: ``"K"``.

Returns
-------
out : dpnp.ndarray
    An array containing the result of element-wise inequality comparison.
    The returned array has a data type of `bool`.

Limitations
-----------
Parameters `where` and `subok` are supported with their default values.
Otherwise ``NotImplementedError`` exception will be raised.

See Also
--------
:obj:`dpnp.equal` : Return (x1 == x2) element-wise.
:obj:`dpnp.greater` : Return the truth value of (x1 > x2) element-wise.
:obj:`dpnp.greater_equal` : Return the truth value of (x1 >= x2) element-wise.
:obj:`dpnp.less` : Return the truth value of (x1 < x2) element-wise.
:obj:`dpnp.less_equal` : Return the truth value of (x1 =< x2) element-wise.

Examples
--------
>>> import dpnp as np
>>> x1 = np.array([1., 2.])
>>> x2 = np.arange(1., 3.)
>>> np.not_equal(x1, x2)
array([False, False])

The ``!=`` operator can be used as a shorthand for ``not_equal`` on
:class:`dpnp.ndarray`.

>>> a = np.array([1., 2.])
>>> b = np.array([1., 3.])
>>> a != b
array([False,  True])
"""

not_equal = DPNPBinaryFunc(
    "not_equal",
    tei._not_equal_result_type,
    tei._not_equal,
    _NOT_EQUAL_DOCSTRING,
)
