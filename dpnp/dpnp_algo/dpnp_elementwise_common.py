# -*- coding: utf-8 -*-
# *****************************************************************************
# Copyright (c) 2023-2025, Intel Corporation
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

import dpctl.tensor as dpt
import dpctl.tensor._tensor_impl as dti
import dpctl.tensor._type_utils as dtu
import numpy
from dpctl.tensor._elementwise_common import (
    BinaryElementwiseFunc,
    UnaryElementwiseFunc,
)

import dpnp
import dpnp.backend.extensions.vm._vm_impl as vmi
from dpnp.dpnp_array import dpnp_array

__all__ = [
    "DPNPI0",
    "DPNPAngle",
    "DPNPBinaryFunc",
    "DPNPFix",
    "DPNPImag",
    "DPNPReal",
    "DPNPRound",
    "DPNPSinc",
    "DPNPUnaryFunc",
    "acceptance_fn_gcd_lcm",
    "acceptance_fn_negative",
    "acceptance_fn_positive",
    "acceptance_fn_sign",
    "acceptance_fn_subtract",
    "resolve_weak_types_2nd_arg_int",
]


class DPNPUnaryFunc(UnaryElementwiseFunc):
    """
    Class that implements unary element-wise functions.

    Parameters
    ----------
    name : {str}
        Name of the unary function
    result_type_resovler_fn : {callable}
        Function that takes dtype of the input and returns the dtype of
        the result if the implementation functions supports it, or
        returns `None` otherwise.
    unary_dp_impl_fn : {callable}
        Data-parallel implementation function with signature
        `impl_fn(src: usm_ndarray, dst: usm_ndarray,
            sycl_queue: SyclQueue, depends: Optional[List[SyclEvent]])`
        where the `src` is the argument array, `dst` is the
        array to be populated with function values, effectively
        evaluating `dst = func(src)`.
        The `impl_fn` is expected to return a 2-tuple of `SyclEvent`s.
        The first event corresponds to data-management host tasks,
        including lifetime management of argument Python objects to ensure
        that their associated USM allocation is not freed before offloaded
        computational tasks complete execution, while the second event
        corresponds to computational tasks associated with function evaluation.
    docs : {str}
        Documentation string for the unary function.
    mkl_fn_to_call : {callable}
        Check input arguments to answer if function from OneMKL VM library
        can be used.
    mkl_impl_fn : {callable}
        Function from OneMKL VM library to call.
    acceptance_fn : {callable}, optional
        Function to influence type promotion behavior of this unary
        function. The function takes 4 arguments:
            arg_dtype - Data type of the first argument
            buf_dtype - Data type the argument would be cast to
            res_dtype - Data type of the output array with function values
            sycl_dev - The :class:`dpctl.SyclDevice` where the function
                evaluation is carried out.
        The function is invoked when the argument of the unary function
        requires casting, e.g. the argument of `dpctl.tensor.log` is an
        array with integral data type.
    """

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
        acceptance_fn=None,
    ):
        def _call_func(src, dst, sycl_queue, depends=None):
            """
            A callback to register in UnaryElementwiseFunc class of
            dpctl.tensor
            """

            if depends is None:
                depends = []

            if vmi._is_available() and mkl_fn_to_call is not None:
                if getattr(vmi, mkl_fn_to_call)(sycl_queue, src, dst):
                    # call pybind11 extension for unary function from OneMKL VM
                    return getattr(vmi, mkl_impl_fn)(
                        sycl_queue, src, dst, depends
                    )
            return unary_dp_impl_fn(src, dst, sycl_queue, depends)

        super().__init__(
            name,
            result_type_resolver_fn,
            _call_func,
            docs,
            acceptance_fn=acceptance_fn,
        )
        self.__name__ = "DPNPUnaryFunc"

    def __call__(
        self,
        x,
        out=None,
        where=True,
        order="K",
        dtype=None,
        subok=True,
        **kwargs,
    ):
        if kwargs:
            raise NotImplementedError(
                f"Requested function={self.name_} with kwargs={kwargs} "
                "isn't currently supported."
            )
        elif where is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with where={where} "
                "isn't currently supported."
            )
        elif subok is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with subok={subok} "
                "isn't currently supported."
            )
        elif not dpnp.is_supported_array_type(x):
            raise TypeError(
                "Input array must be any of supported type, "
                f"but got {type(x)}"
            )
        elif dtype is not None and out is not None:
            raise TypeError(
                f"Requested function={self.name_} only takes `out` or `dtype`"
                "as an argument, but both were provided."
            )

        if order is None:
            order = "K"
        elif order in "afkcAFKC":
            order = order.upper()
        else:
            raise ValueError(
                "order must be one of 'C', 'F', 'A', or 'K' " f"(got '{order}')"
            )

        x_usm = dpnp.get_usm_ndarray(x)
        if dtype is not None:
            x_usm = dpt.astype(x_usm, dtype, copy=False)

        out_usm = None if out is None else dpnp.get_usm_ndarray(out)
        res_usm = super().__call__(x_usm, out=out_usm, order=order)

        if out is not None and isinstance(out, dpnp_array):
            return out
        return dpnp_array._create_from_usm_ndarray(res_usm)


class DPNPBinaryFunc(BinaryElementwiseFunc):
    """
    Class that implements binary element-wise functions.

    Args:
    name : {str}
        Name of the unary function
    result_type_resovle_fn : {callable}
        Function that takes dtype of the input and returns the dtype of
        the result if the implementation functions supports it, or
        returns `None` otherwise..
    binary_dp_impl_fn : {callable}
        Data-parallel implementation function with signature
        `impl_fn(src1: usm_ndarray, src2: usm_ndarray, dst: usm_ndarray,
            sycl_queue: SyclQueue, depends: Optional[List[SyclEvent]])`
        where the `src1` and `src2` are the argument arrays, `dst` is the
        array to be populated with function values,
        i.e. `dst=func(src1, src2)`.
        The `impl_fn` is expected to return a 2-tuple of `SyclEvent`s.
        The first event corresponds to data-management host tasks,
        including lifetime management of argument Python objects to ensure
        that their associated USM allocation is not freed before offloaded
        computational tasks complete execution, while the second event
        corresponds to computational tasks associated with function
        evaluation.
    docs : {str}
        Documentation string for the unary function.
    mkl_fn_to_call : {callable}
        Check input arguments to answer if function from OneMKL VM library
        can be used.
    mkl_impl_fn : {callable}
        Function from OneMKL VM library to call.
    binary_inplace_fn : {callable}, optional
        Data-parallel implementation function with signature
        `impl_fn(src: usm_ndarray, dst: usm_ndarray,
            sycl_queue: SyclQueue, depends: Optional[List[SyclEvent]])`
        where the `src` is the argument array, `dst` is the
        array to be populated with function values,
        i.e. `dst=func(dst, src)`.
        The `impl_fn` is expected to return a 2-tuple of `SyclEvent`s.
        The first event corresponds to data-management host tasks,
        including async lifetime management of Python arguments,
        while the second event corresponds to computational tasks
        associated with function evaluation.
    acceptance_fn : {callable}, optional
        Function to influence type promotion behavior of this binary
        function. The function takes 6 arguments:
            arg1_dtype - Data type of the first argument
            arg2_dtype - Data type of the second argument
            ret_buf1_dtype - Data type the first argument would be cast to
            ret_buf2_dtype - Data type the second argument would be cast to
            res_dtype - Data type of the output array with function values
            sycl_dev - The :class:`dpctl.SyclDevice` where the function
                evaluation is carried out.
        The function is only called when both arguments of the binary
        function require casting, e.g. both arguments of
        `dpctl.tensor.logaddexp` are arrays with integral data type.
    weak_type_resolver : {callable}, optional
        Function to influence type promotion behavior for Python scalar types
        of this binary function. The function takes 3 arguments:
            o1_dtype - Data type or Python scalar type of the first argument
            o2_dtype - Data type or Python scalar type of of the second argument
            sycl_dev - The :class:`dpctl.SyclDevice` where the function
                evaluation is carried out.
        One of `o1_dtype` and `o2_dtype` must be a ``dtype`` instance.
    """

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        binary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
        binary_inplace_fn=None,
        acceptance_fn=None,
        weak_type_resolver=None,
    ):
        def _call_func(src1, src2, dst, sycl_queue, depends=None):
            """
            A callback to register in UnaryElementwiseFunc class of
            dpctl.tensor
            """

            if depends is None:
                depends = []

            if vmi._is_available() and mkl_fn_to_call is not None:
                if getattr(vmi, mkl_fn_to_call)(sycl_queue, src1, src2, dst):
                    # call pybind11 extension for binary function from OneMKL VM
                    return getattr(vmi, mkl_impl_fn)(
                        sycl_queue, src1, src2, dst, depends
                    )
            return binary_dp_impl_fn(src1, src2, dst, sycl_queue, depends)

        super().__init__(
            name,
            result_type_resolver_fn,
            _call_func,
            docs,
            binary_inplace_fn,
            acceptance_fn=acceptance_fn,
            weak_type_resolver=weak_type_resolver,
        )
        self.__name__ = "DPNPBinaryFunc"

    def __call__(
        self,
        x1,
        x2,
        out=None,
        where=True,
        order="K",
        dtype=None,
        subok=True,
        **kwargs,
    ):
        dpnp.check_supported_arrays_type(
            x1, x2, scalar_type=True, all_scalars=False
        )
        if kwargs:
            raise NotImplementedError(
                f"Requested function={self.name_} with kwargs={kwargs} "
                "isn't currently supported."
            )
        elif where is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with where={where} "
                "isn't currently supported."
            )
        elif subok is not True:
            raise NotImplementedError(
                f"Requested function={self.name_} with subok={subok} "
                "isn't currently supported."
            )
        elif dtype is not None and out is not None:
            raise TypeError(
                f"Requested function={self.name_} only takes `out` or `dtype`"
                "as an argument, but both were provided."
            )

        x1_usm = dpnp.get_usm_ndarray_or_scalar(x1)
        x2_usm = dpnp.get_usm_ndarray_or_scalar(x2)
        out_usm = None if out is None else dpnp.get_usm_ndarray(out)

        if (
            isinstance(x1, dpnp_array)
            and x1 is out
            and order == "K"
            and dtype is None
        ):
            # in-place operation
            super()._inplace_op(x1_usm, x2_usm)
            return x1

        if order is None:
            order = "K"
        elif order in "afkcAFKC":
            order = order.upper()
        else:
            raise ValueError(
                "order must be one of 'C', 'F', 'A', or 'K' (got '{order}')"
            )

        if dtype is not None:
            if dpnp.isscalar(x1):
                x1_usm = dpt.asarray(
                    x1,
                    dtype=dtype,
                    sycl_queue=x2.sycl_queue,
                    usm_type=x2.usm_type,
                )
                x2_usm = dpt.astype(x2_usm, dtype, copy=False)
            elif dpnp.isscalar(x2):
                x1_usm = dpt.astype(x1_usm, dtype, copy=False)
                x2_usm = dpt.asarray(
                    x2,
                    dtype=dtype,
                    sycl_queue=x1.sycl_queue,
                    usm_type=x1.usm_type,
                )
            else:
                x1_usm = dpt.astype(x1_usm, dtype, copy=False)
                x2_usm = dpt.astype(x2_usm, dtype, copy=False)

        res_usm = super().__call__(x1_usm, x2_usm, out=out_usm, order=order)

        if out is not None and isinstance(out, dpnp_array):
            return out
        return dpnp_array._create_from_usm_ndarray(res_usm)

    def outer(
        self,
        x1,
        x2,
        out=None,
        where=True,
        order="K",
        dtype=None,
        subok=True,
        **kwargs,
    ):
        """
        Apply the ufunc op to all pairs (a, b) with a in A and b in B.

        Parameters
        ----------
        x1 : {dpnp.ndarray, usm_ndarray}
            First input array.
        x2 : {dpnp.ndarray, usm_ndarray}
            Second input array.
        out : {None, dpnp.ndarray, usm_ndarray}, optional
            Output array to populate.
            Array must have the correct shape and the expected data type.
        order : {None, "C", "F", "A", "K"}, optional
            Memory layout of the newly output array, Cannot be provided
            together with `out`. Default: ``"K"``.
        dtype : {None, str, dtype object}, optional
            If provided, the destination array will have this dtype. Cannot be
            provided together with `out`. Default: ``None``.

        Returns
        -------
        out : dpnp.ndarray
            Output array. The data type of the returned array is determined by
            the Type Promotion Rules.

        Limitations
        -----------
        Parameters `where` and `subok` are supported with their default values.
        Keyword argument `kwargs` is currently unsupported.
        Otherwise ``NotImplementedError`` exception will be raised.

        See also
        --------
        :obj:`dpnp.outer` : A less powerful version of dpnp.multiply.outer
                            that ravels all inputs to 1D. This exists primarily
                            for compatibility with old code.

        :obj:`dpnp.tensordot` : dpnp.tensordot(a, b, axes=((), ())) and
                                dpnp.multiply.outer(a, b) behave same for all
                                dimensions of a and b.

        Examples
        --------
        >>> import dpnp as np
        >>> A = np.array([1, 2, 3])
        >>> B = np.array([4, 5, 6])
        >>> np.multiply.outer(A, B)
        array([[ 4,  5,  6],
               [ 8, 10, 12],
               [12, 15, 18]])

        A multi-dimensional example:
        >>> A = np.array([[1, 2, 3], [4, 5, 6]])
        >>> A.shape
        (2, 3)
        >>> B = np.array([[1, 2, 3, 4]])
        >>> B.shape
        (1, 4)
        >>> C = np.multiply.outer(A, B)
        >>> C.shape; C
        (2, 3, 1, 4)
        array([[[[ 1,  2,  3,  4]],
                [[ 2,  4,  6,  8]],
                [[ 3,  6,  9, 12]]],
               [[[ 4,  8, 12, 16]],
                [[ 5, 10, 15, 20]],
                [[ 6, 12, 18, 24]]]])

        """

        dpnp.check_supported_arrays_type(
            x1, x2, scalar_type=True, all_scalars=False
        )
        if dpnp.isscalar(x1) or dpnp.isscalar(x2):
            _x1 = x1
            _x2 = x2
        else:
            _x1 = x1[(Ellipsis,) + (None,) * x2.ndim]
            _x2 = x2[(None,) * x1.ndim + (Ellipsis,)]
        return self.__call__(
            _x1,
            _x2,
            out=out,
            where=where,
            order=order,
            dtype=dtype,
            subok=subok,
            **kwargs,
        )


class DPNPAngle(DPNPUnaryFunc):
    """Class that implements dpnp.angle unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
            mkl_fn_to_call=mkl_fn_to_call,
            mkl_impl_fn=mkl_impl_fn,
        )

    def __call__(self, x, deg=False, out=None, order="K"):
        res = super().__call__(x, out=out, order=order)
        if deg is True:
            res *= 180 / dpnp.pi
        return res


class DPNPFix(DPNPUnaryFunc):
    """Class that implements dpnp.fix unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
        )

    def __call__(self, x, out=None, order="K"):
        if not dpnp.is_supported_array_type(x):
            pass  # pass to raise error in main implementation
        elif dpnp.issubdtype(x.dtype, dpnp.inexact):
            pass  # for inexact types, pass to calculate in the backend
        elif out is not None and not dpnp.is_supported_array_type(out):
            pass  # pass to raise error in main implementation
        elif out is not None and out.dtype != x.dtype:
            # passing will raise an error but with incorrect needed dtype
            raise ValueError(
                f"Output array of type {x.dtype} is needed, got {out.dtype}"
            )
        else:
            # for exact types, return the input
            if out is None:
                return dpnp.copy(x, order=order)

            if isinstance(out, dpt.usm_ndarray):
                out = dpnp_array._create_from_usm_ndarray(out)
            out[...] = x
            return out

        return super().__call__(x, out=out, order=order)


class DPNPI0(DPNPUnaryFunc):
    """Class that implements dpnp.i0 unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
            mkl_fn_to_call=mkl_fn_to_call,
            mkl_impl_fn=mkl_impl_fn,
        )

    def __call__(self, x, out=None, order="K"):
        return super().__call__(x, out=out, order=order)


class DPNPImag(DPNPUnaryFunc):
    """Class that implements dpnp.imag unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
        )

    def __call__(self, x, out=None, order="K"):
        return super().__call__(x, out=out, order=order)


class DPNPReal(DPNPUnaryFunc):
    """Class that implements dpnp.real unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
        )

    def __call__(self, x, out=None, order="K"):
        if numpy.iscomplexobj(x):
            return super().__call__(x, out=out, order=order)
        return x


class DPNPRound(DPNPUnaryFunc):
    """Class that implements dpnp.round unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
        mkl_fn_to_call=None,
        mkl_impl_fn=None,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
            mkl_fn_to_call=mkl_fn_to_call,
            mkl_impl_fn=mkl_impl_fn,
        )

    def __call__(self, x, decimals=0, out=None, dtype=None):
        if decimals != 0:
            x_usm = dpnp.get_usm_ndarray(x)
            out_usm = None if out is None else dpnp.get_usm_ndarray(out)

            if dpnp.issubdtype(x_usm.dtype, dpnp.integer):
                if decimals < 0:
                    dtype = x_usm.dtype
                    x_usm = dpt.round(x_usm * 10**decimals, out=out_usm)
                    res_usm = dpt.divide(x_usm, 10**decimals, out=out_usm)
                else:
                    res_usm = dpt.round(x_usm, out=out_usm)
            else:
                x_usm = dpt.round(x_usm * 10**decimals, out=out_usm)
                res_usm = dpt.divide(x_usm, 10**decimals, out=out_usm)

            if dtype is not None:
                res_usm = dpt.astype(res_usm, dtype, copy=False)

            if out is not None and isinstance(out, dpnp_array):
                return out
            return dpnp_array._create_from_usm_ndarray(res_usm)
        else:
            return super().__call__(x, out=out, dtype=dtype)


class DPNPSinc(DPNPUnaryFunc):
    """Class that implements dpnp.sinc unary element-wise functions."""

    def __init__(
        self,
        name,
        result_type_resolver_fn,
        unary_dp_impl_fn,
        docs,
    ):
        super().__init__(
            name,
            result_type_resolver_fn,
            unary_dp_impl_fn,
            docs,
        )

    def __call__(self, x, out=None, order="K"):
        return super().__call__(x, out=out, order=order)


def acceptance_fn_gcd_lcm(
    arg1_dtype, arg2_dtype, buf1_dt, buf2_dt, res_dt, sycl_dev
):
    # gcd/lcm are not defined for boolean data type
    if arg1_dtype.char == "?" and arg2_dtype.char == "?":
        raise ValueError(
            "The function is not supported for inputs of data type bool"
        )
    else:
        return True


def acceptance_fn_negative(arg_dtype, buf_dt, res_dt, sycl_dev):
    # negative is not defined for boolean data type
    if arg_dtype.char == "?":
        raise TypeError(
            "The `negative` function, the `-` operator, is not supported "
            "for inputs of data type bool, use the `~` operator or the "
            "`logical_not` function instead"
        )
    else:
        return True


def acceptance_fn_positive(arg_dtype, buf_dt, res_dt, sycl_dev):
    # positive is not defined for boolean data type
    if arg_dtype.char == "?":
        raise TypeError(
            "The `positive` function is not supported for inputs of data type "
            "bool"
        )
    else:
        return True


def acceptance_fn_sign(arg_dtype, buf_dt, res_dt, sycl_dev):
    # sign is not defined for boolean data type
    if arg_dtype.char == "?":
        raise TypeError(
            "The `sign` function is not supported for inputs of data type bool"
        )
    else:
        return True


def acceptance_fn_subtract(
    arg1_dtype, arg2_dtype, buf1_dt, buf2_dt, res_dt, sycl_dev
):
    # subtract is not defined for boolean data type
    if arg1_dtype.char == "?" and arg2_dtype.char == "?":
        raise TypeError(
            "The `subtract` function, the `-` operator, is not supported "
            "for inputs of data type bool, use the `^` operator,  the "
            "`bitwise_xor`, or the `logical_xor` function instead"
        )
    else:
        return True


def resolve_weak_types_2nd_arg_int(o1_dtype, o2_dtype, sycl_dev):
    """
    The second weak dtype has to be upcasting up to default integer dtype
    for a SYCL device where it is possible.
    For other cases the default weak types resolving will be applied.

    """

    if dtu._is_weak_dtype(o2_dtype):
        o1_kind_num = dtu._strong_dtype_num_kind(o1_dtype)
        o2_kind_num = dtu._weak_type_num_kind(o2_dtype)
        if o2_kind_num < o1_kind_num:
            if isinstance(
                o2_dtype, (dtu.WeakBooleanType, dtu.WeakIntegralType)
            ):
                return o1_dtype, dpt.dtype(
                    dti.default_device_int_type(sycl_dev)
                )
    return dtu._resolve_weak_types(o1_dtype, o2_dtype, sycl_dev)
