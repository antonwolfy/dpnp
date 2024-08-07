# *****************************************************************************
# Copyright (c) 2024, Intel Corporation
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
Helping functions to implement the FFT interface.

These include assertion functions to validate input array and
functions with the main implementation part to fulfill the interface.
The main computational work is performed by enabling FFT functions
available as a pybind11 extension.

"""

# pylint: disable=protected-access
# pylint: disable=no-name-in-module

import dpctl
import dpctl.tensor._tensor_impl as ti
import dpctl.utils as dpu
import numpy
from dpctl.tensor._numpy_helper import normalize_axis_index
from dpctl.utils import ExecutionPlacementError

import dpnp
import dpnp.backend.extensions.fft._fft_impl as fi

from ..dpnp_array import dpnp_array
from ..dpnp_utils import map_dtype_to_device
from ..dpnp_utils.dpnp_utils_linearalgebra import (
    _standardize_strides_to_nonzero,
)

__all__ = [
    "dpnp_fft",
]


def _check_norm(norm):
    if norm not in (None, "ortho", "forward", "backward"):
        raise ValueError(
            f"Invalid norm value {norm}; should be None, "
            '"ortho", "forward", or "backward".'
        )


def _commit_descriptor(a, in_place, c2c, a_strides, index, axes):
    """Commit the FFT descriptor for the input array."""

    a_shape = a.shape
    shape = a_shape[index:]
    strides = (0,) + a_strides[index:]
    if c2c:  # c2c FFT
        if a.dtype == dpnp.complex64:
            dsc = fi.Complex64Descriptor(shape)
        else:
            dsc = fi.Complex128Descriptor(shape)
    else:  # r2c/c2r FFT
        if a.dtype in [dpnp.float32, dpnp.complex64]:
            dsc = fi.Real32Descriptor(shape)
        else:
            dsc = fi.Real64Descriptor(shape)

    dsc.fwd_strides = strides
    dsc.bwd_strides = dsc.fwd_strides
    dsc.transform_in_place = in_place
    if axes is not None:  # batch_fft
        dsc.fwd_distance = a_strides[0]
        dsc.bwd_distance = dsc.fwd_distance
        dsc.number_of_transforms = numpy.prod(a_shape[0])
    dsc.commit(a.sycl_queue)

    return dsc


def _compute_result(dsc, a, out, forward, c2c, a_strides):
    """Compute the result of the FFT."""

    exec_q = a.sycl_queue
    _manager = dpu.SequentialOrderManager[exec_q]
    dep_evs = _manager.submitted_events

    a_usm = dpnp.get_usm_ndarray(a)
    if dsc.transform_in_place:
        # in-place transform
        # TODO: investigate the performance of in-place implementation
        # for r2c/c2r, see SAT-7154
        ht_fft_event, fft_event = fi._fft_in_place(
            dsc, a_usm, forward, depends=dep_evs
        )
        result = a
    else:
        if (
            out is not None
            and out.strides == a_strides
            and not ti._array_overlap(a_usm, dpnp.get_usm_ndarray(out))
        ):
            res_usm = dpnp.get_usm_ndarray(out)
            result = out
        else:
            # Result array that is used in OneMKL must have the exact same
            # stride as input array

            if c2c:  # c2c FFT
                out_shape = a.shape
                out_dtype = a.dtype
            else:
                if forward:  # r2c FFT
                    tmp = a.shape[-1] // 2 + 1
                    out_shape = a.shape[:-1] + (tmp,)
                    out_dtype = (
                        dpnp.complex64
                        if a.dtype == dpnp.float32
                        else dpnp.complex128
                    )
                else:  # c2r FFT
                    out_shape = a.shape  # a is already zero-padded
                    out_dtype = (
                        dpnp.float32
                        if a.dtype == dpnp.complex64
                        else dpnp.float64
                    )
            result = dpnp_array(
                out_shape,
                dtype=out_dtype,
                strides=a_strides,
                usm_type=a.usm_type,
                sycl_queue=exec_q,
            )
            res_usm = result.get_array()
        ht_fft_event, fft_event = fi._fft_out_of_place(
            dsc, a_usm, res_usm, forward, depends=dep_evs
        )
    _manager.add_event_pair(ht_fft_event, fft_event)

    if not isinstance(result, dpnp_array):
        return dpnp_array._create_from_usm_ndarray(result)
    return result


def _copy_array(x, complex_input):
    """
    Creating a C-contiguous copy of input array if input array has a negative
    stride or it does not have a complex data types. In this situation, an
    in-place FFT can be performed.
    """
    dtype = x.dtype
    if numpy.min(x.strides) < 0:
        # negative stride is not allowed in OneMKL FFT
        copy_flag = True
    elif complex_input and not dpnp.issubdtype(dtype, dpnp.complexfloating):
        # c2c/c2r FFT, if input is not complex, convert to complex
        copy_flag = True
        if dtype == dpnp.float32:
            dtype = dpnp.complex64
        else:
            dtype = map_dtype_to_device(dpnp.complex128, x.sycl_device)
    elif not complex_input and dtype not in [dpnp.float32, dpnp.float64]:
        # r2c FFT, if input is integer or float16 dtype, convert to
        # float32 or float64 depending on device capabilities
        copy_flag = True
        dtype = map_dtype_to_device(dpnp.float64, x.sycl_device)
    else:
        copy_flag = False

    if copy_flag:
        x_copy = dpnp.empty_like(x, dtype=dtype, order="C")

        exec_q = x.sycl_queue
        _manager = dpu.SequentialOrderManager[exec_q]
        dep_evs = _manager.submitted_events

        ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
            src=dpnp.get_usm_ndarray(x),
            dst=x_copy.get_array(),
            sycl_queue=exec_q,
            depends=dep_evs,
        )
        _manager.add_event_pair(ht_copy_ev, copy_ev)
        x = x_copy

    # if copying is done, FFT can be in-place (copy_flag = in_place flag)
    return x, copy_flag


def _fft(a, norm, out, forward, in_place, c2c, axes=None):
    """Calculates FFT of the input array along the specified axes."""

    index = 0
    if axes is not None:  # batch_fft
        len_axes = 1 if isinstance(axes, int) else len(axes)
        local_axes = numpy.arange(-len_axes, 0)
        a = dpnp.moveaxis(a, axes, local_axes)
        a_shape_orig = a.shape
        local_shape = (-1,) + a_shape_orig[-len_axes:]
        a = dpnp.reshape(a, local_shape)
        index = 1

    a_strides = _standardize_strides_to_nonzero(a.strides, a.shape)
    dsc = _commit_descriptor(a, in_place, c2c, a_strides, index, axes)
    res = _compute_result(dsc, a, out, forward, c2c, a_strides)
    res = _scale_result(res, a.shape, norm, forward, index)

    if axes is not None:  # batch_fft
        tmp_shape = a_shape_orig[:-1] + (res.shape[-1],)
        res = dpnp.reshape(res, tmp_shape)
        res = dpnp.moveaxis(res, local_axes, axes)

    result = dpnp.get_result_array(res, out=out, casting="same_kind")
    if out is None and not (
        result.flags.c_contiguous or result.flags.f_contiguous
    ):
        result = dpnp.ascontiguousarray(result)

    return result


def _scale_result(res, a_shape, norm, forward, index):
    """Scale the result of the FFT according to `norm`."""
    scale = numpy.prod(a_shape[index:], dtype=res.real.dtype)
    norm_factor = 1
    if norm == "ortho":
        norm_factor = numpy.sqrt(scale)
    elif norm == "forward" and forward:
        norm_factor = scale
    elif norm in [None, "backward"] and not forward:
        norm_factor = scale

    res /= norm_factor
    return res


def _truncate_or_pad(a, shape, axes):
    """Truncating or zero-padding the input array along the specified axes."""

    shape = (shape,) if isinstance(shape, int) else shape
    axes = (axes,) if isinstance(axes, int) else axes

    for s, axis in zip(shape, axes):
        a_shape = list(a.shape)
        index = [slice(None)] * a.ndim
        if s == a_shape[axis]:
            pass
        elif s < a_shape[axis]:
            # truncating
            index[axis] = slice(0, s)
            a = a[tuple(index)]
        else:
            # zero-padding
            exec_q = a.sycl_queue
            _manager = dpu.SequentialOrderManager[exec_q]
            dep_evs = _manager.submitted_events
            index[axis] = slice(0, a_shape[axis])  # orig shape
            a_shape[axis] = s  # modified shape
            order = "F" if a.flags.fnc else "C"
            z = dpnp.zeros(
                a_shape,
                dtype=a.dtype,
                order=order,
                usm_type=a.usm_type,
                sycl_queue=exec_q,
            )
            ht_copy_ev, copy_ev = ti._copy_usm_ndarray_into_usm_ndarray(
                src=dpnp.get_usm_ndarray(a),
                dst=z.get_array()[tuple(index)],
                sycl_queue=exec_q,
                depends=dep_evs,
            )
            _manager.add_event_pair(ht_copy_ev, copy_ev)
            a = z

    return a


def _validate_out_keyword(a, out, axis, c2r, r2c):
    """Validate out keyword argument."""
    if out is not None:
        dpnp.check_supported_arrays_type(out)
        if (
            dpctl.utils.get_execution_queue((a.sycl_queue, out.sycl_queue))
            is None
        ):
            raise ExecutionPlacementError(
                "Input and output allocation queues are not compatible"
            )

        # validate out shape
        expected_shape = a.shape
        if r2c:
            expected_shape = list(a.shape)
            expected_shape[axis] = a.shape[axis] // 2 + 1
            expected_shape = tuple(expected_shape)
        if out.shape != expected_shape:
            raise ValueError(
                "output array has incorrect shape, expected "
                f"{expected_shape}, got {out.shape}."
            )

        # validate out data type
        if c2r:
            if not dpnp.issubdtype(out.dtype, dpnp.floating):
                raise TypeError(
                    "output array should have real floating data type."
                )
        else:  # c2c/r2c FFT
            if not dpnp.issubdtype(out.dtype, dpnp.complexfloating):
                raise TypeError("output array should have complex data type.")


def dpnp_fft(a, forward, real, n=None, axis=-1, norm=None, out=None):
    """Calculates 1-D FFT of the input array along axis"""

    a_ndim = a.ndim
    if a_ndim == 0:
        raise ValueError("Input array must be at least 1D")

    c2c = not real  # complex-to-complex FFT
    r2c = real and forward  # real-to-complex FFT
    c2r = real and not forward  # complex-to-real FFT
    if r2c and dpnp.issubdtype(a.dtype, dpnp.complexfloating):
        raise TypeError("Input array must be real")

    axis = normalize_axis_index(axis, a_ndim)
    if n is None:
        if c2r:
            n = (a.shape[axis] - 1) * 2
        else:
            n = a.shape[axis]
    elif not isinstance(n, int):
        raise TypeError("`n` should be None or an integer")
    if n < 1:
        raise ValueError(f"Invalid number of FFT data points ({n}) specified")

    _check_norm(norm)
    a = _truncate_or_pad(a, n, axis)
    _validate_out_keyword(a, out, axis, c2r, r2c)
    # if input array is copied, in-place FFT can be used
    a, in_place = _copy_array(a, c2c or c2r)
    if not in_place and out is not None:
        # if input is also given for out, in-place FFT can be used
        in_place = dpnp.are_same_logical_tensors(a, out)

    if a.size == 0:
        return dpnp.get_result_array(a, out=out, casting="same_kind")

    # non-batch FFT
    axis = None if a_ndim == 1 else axis

    return _fft(
        a,
        norm=norm,
        out=out,
        forward=forward,
        # TODO: currently in-place is only implemented for c2c, see SAT-7154
        in_place=in_place and c2c,
        c2c=c2c,
        axes=axis,
    )
