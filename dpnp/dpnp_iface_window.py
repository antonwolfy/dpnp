# *****************************************************************************
# Copyright (c) 2025, Intel Corporation
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
Interface of the window functions of dpnp

Notes
-----
This module is a face or public interface file for the library
it contains:
 - Interface functions
 - documentation for the functions
 - The functions parameters check

"""

# pylint: disable=no-name-in-module
# pylint: disable=invalid-name
# pylint: disable=protected-access

import dpctl.utils as dpu

import dpnp
import dpnp.backend.extensions.window._window_impl as wi

__all__ = ["blackman", "hamming", "hanning"]


def _call_window_kernel(
    M, _window_kernel, device=None, usm_type=None, sycl_queue=None
):

    try:
        M = int(M)
    except Exception as e:
        raise TypeError("M must be an integer") from e

    cfd_kwarg = {
        "device": device,
        "usm_type": usm_type,
        "sycl_queue": sycl_queue,
    }

    if M < 1:
        return dpnp.empty(0, **cfd_kwarg)
    if M == 1:
        return dpnp.ones(1, **cfd_kwarg)

    result = dpnp.empty(M, **cfd_kwarg)
    exec_q = result.sycl_queue
    _manager = dpu.SequentialOrderManager[exec_q]

    ht_ev, win_ev = _window_kernel(
        exec_q, dpnp.get_usm_ndarray(result), depends=_manager.submitted_events
    )

    _manager.add_event_pair(ht_ev, win_ev)

    return result


def blackman(M, device=None, usm_type=None, sycl_queue=None):
    r"""
    Return the Blackman window.

    The Blackman window is a taper formed by using the first three terms of a
    summation of cosines. It was designed to have close to the minimal leakage
    possible. It is close to optimal, only slightly worse than a Kaiser window.

    For full documentation refer to :obj:`numpy.blackman`.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty array
        is returned.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the output array is created.
        `device` can be ``None``, a oneAPI filter selector string, an instance
        of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL
        device, an instance of :class:`dpctl.SyclQueue`, or a
        :class:`dpctl.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.

        Default: ``None``.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of shape (M,)
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    :obj:`dpnp.bartlett` : Return the Bartlett window.
    :obj:`dpnp.hamming` : Return the Hamming window.
    :obj:`dpnp.hanning` : Return the Hanning window.
    :obj:`dpnp.kaiser` : Return the Kaiser window.

    Notes
    -----
    The Blackman window is defined as

    .. math::  w(n) = 0.42 - 0.5\cos\left(\frac{2\pi{n}}{M-1}\right)
               + 0.08\cos\left(\frac{4\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    Examples
    --------
    >>> import dpnp as np
    >>> np.blackman(12)
    array([-1.38777878e-17,  3.26064346e-02,  1.59903635e-01,  4.14397981e-01,
            7.36045180e-01,  9.67046769e-01,  9.67046769e-01,  7.36045180e-01,
            4.14397981e-01,  1.59903635e-01,  3.26064346e-02, -1.38777878e-17])

    Creating the output array on a different device or with a
    specified usm_type:

    >>> x = np.blackman(3) # default case
    >>> x, x.device, x.usm_type
    (array([-1.38777878e-17,  1.00000000e+00, -1.38777878e-17]),
     Device(level_zero:gpu:0),
     'device')

    >>> y = np.blackman(3, device="cpu")
    >>> y, y.device, y.usm_type
    (array([-1.38777878e-17,  1.00000000e+00, -1.38777878e-17]),
     Device(opencl:cpu:0),
     'device')

    >>> z = np.blackman(3, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([-1.38777878e-17,  1.00000000e+00, -1.38777878e-17]),
     Device(level_zero:gpu:0),
     'host')

    """

    return _call_window_kernel(
        M, wi._blackman, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )


def hamming(M, device=None, usm_type=None, sycl_queue=None):
    r"""
    Return the Hamming window.

    The Hamming window is a taper formed by using a weighted cosine.

    For full documentation refer to :obj:`numpy.hamming`.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty array
        is returned.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the output array is created.
        `device` can be ``None``, a oneAPI filter selector string, an instance
        of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL
        device, an instance of :class:`dpctl.SyclQueue`, or a
        :class:`dpctl.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.

        Default: ``None``.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of shape (M,)
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    :obj:`dpnp.bartlett` : Return the Bartlett window.
    :obj:`dpnp.blackman` : Return the Blackman window.
    :obj:`dpnp.hanning` : Return the Hanning window.
    :obj:`dpnp.kaiser` : Return the Kaiser window.

    Notes
    -----
    The Hamming window is defined as

    .. math::  w(n) = 0.54 - 0.46\cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    Examples
    --------
    >>> import dpnp as np
    >>> np.hamming(12)
    array([0.08      , 0.15302337, 0.34890909, 0.60546483, 0.84123594,
           0.98136677, 0.98136677, 0.84123594, 0.60546483, 0.34890909,
           0.15302337, 0.08      ])  # may vary

    Creating the output array on a different device or with a
    specified usm_type:

    >>> x = np.hamming(4) # default case
    >>> x, x.device, x.usm_type
    (array([0.08, 0.77, 0.77, 0.08]), Device(level_zero:gpu:0), 'device')

    >>> y = np.hamming(4, device="cpu")
    >>> y, y.device, y.usm_type
    (array([0.08, 0.77, 0.77, 0.08]), Device(opencl:cpu:0), 'device')

    >>> z = np.hamming(4, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([0.08, 0.77, 0.77, 0.08]), Device(level_zero:gpu:0), 'host')

    """

    return _call_window_kernel(
        M, wi._hamming, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )


def hanning(M, device=None, usm_type=None, sycl_queue=None):
    r"""
    Return the Hanning window.

    The Hanning window is a taper formed by using a weighted cosine.

    For full documentation refer to :obj:`numpy.hanning`.

    Parameters
    ----------
    M : int
        Number of points in the output window. If zero or less, an empty array
        is returned.
    device : {None, string, SyclDevice, SyclQueue, Device}, optional
        An array API concept of device where the output array is created.
        `device` can be ``None``, a oneAPI filter selector string, an instance
        of :class:`dpctl.SyclDevice` corresponding to a non-partitioned SYCL
        device, an instance of :class:`dpctl.SyclQueue`, or a
        :class:`dpctl.tensor.Device` object returned by
        :attr:`dpnp.ndarray.device`.

        Default: ``None``.
    usm_type : {None, "device", "shared", "host"}, optional
        The type of SYCL USM allocation for the output array.

        Default: ``None``.
    sycl_queue : {None, SyclQueue}, optional
        A SYCL queue to use for output array allocation and copying. The
        `sycl_queue` can be passed as ``None`` (the default), which means
        to get the SYCL queue from `device` keyword if present or to use
        a default queue.

        Default: ``None``.

    Returns
    -------
    out : dpnp.ndarray of shape (M,)
        The window, with the maximum value normalized to one (the value one
        appears only if the number of samples is odd).

    See Also
    --------
    :obj:`dpnp.bartlett` : Return the Bartlett window.
    :obj:`dpnp.blackman` : Return the Blackman window.
    :obj:`dpnp.hamming` : Return the Hamming window.
    :obj:`dpnp.kaiser` : Return the Kaiser window.

    Notes
    -----
    The Hanning window is defined as

    .. math::  w(n) = 0.5 - 0.5\cos\left(\frac{2\pi{n}}{M-1}\right)
               \qquad 0 \leq n \leq M-1

    Examples
    --------
    >>> import dpnp as np
    >>> np.hanning(12)
    array([0.        , 0.07937323, 0.29229249, 0.57115742, 0.82743037,
           0.97974649, 0.97974649, 0.82743037, 0.57115742, 0.29229249,
           0.07937323, 0.        ])

    Creating the output array on a different device or with a
    specified usm_type:

    >>> x = np.hanning(4) # default case
    >>> x, x.device, x.usm_type
    (array([0.  , 0.75, 0.75, 0.  ]), Device(level_zero:gpu:0), 'device')

    >>> y = np.hanning(4, device="cpu")
    >>> y, y.device, y.usm_type
    (array([0.  , 0.75, 0.75, 0.  ]), Device(opencl:cpu:0), 'device')

    >>> z = np.hanning(4, usm_type="host")
    >>> z, z.device, z.usm_type
    (array([0.  , 0.75, 0.75, 0.  ]), Device(level_zero:gpu:0), 'host')

    """

    return _call_window_kernel(
        M, wi._hanning, device=device, usm_type=usm_type, sycl_queue=sycl_queue
    )
