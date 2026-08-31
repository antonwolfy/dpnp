# *****************************************************************************
# Copyright (c) 2026, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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

import numpy as np
import pytest

import dpnp.tensor as dpt
import dpnp.tensor._copy_utils as cu
from dpnp.tests.third_party.cupy.testing import with_requires

from .helper import get_queue_or_skip, skip_if_dtype_not_supported


def test_copy_utils_empty_like_orderK():
    get_queue_or_skip()
    a = dpt.empty((10, 10), dtype=dpt.int32, order="F")
    X = cu._empty_like_orderK(a, dpt.int32, a.usm_type, a.device)
    assert X.flags["F"]


def test_copy_utils_empty_like_orderK_invalid_args():
    get_queue_or_skip()
    with pytest.raises(TypeError):
        cu._empty_like_orderK([1, 2, 3], dpt.int32, "device", None)
    with pytest.raises(TypeError):
        cu._empty_like_pair_orderK(
            [1, 2, 3],
            (
                1,
                2,
                3,
            ),
            dpt.int32,
            (3,),
            "device",
            None,
        )

    a = dpt.empty(10, dtype=dpt.int32)
    with pytest.raises(TypeError):
        cu._empty_like_pair_orderK(
            a,
            (
                1,
                2,
                3,
            ),
            dpt.int32,
            (10,),
            "device",
            None,
        )


def test_copy_utils_from_numpy_empty_like_orderK():
    q = get_queue_or_skip()

    a = np.empty((10, 10), dtype=np.int32, order="C")
    r0 = cu._from_numpy_empty_like_orderK(a, dpt.int32, "device", q)
    assert r0.flags["C"]

    b = np.empty((10, 10), dtype=np.int32, order="F")
    r1 = cu._from_numpy_empty_like_orderK(b, dpt.int32, "device", q)
    assert r1.flags["F"]

    c = np.empty((2, 3, 4), dtype=np.int32, order="C")
    c = np.transpose(c, (1, 0, 2))
    r2 = cu._from_numpy_empty_like_orderK(c, dpt.int32, "device", q)
    assert not r2.flags["C"] and not r2.flags["F"]


def test_copy_utils_from_numpy_empty_like_orderK_invalid_args():
    with pytest.raises(TypeError):
        cu._from_numpy_empty_like_orderK([1, 2, 3], dpt.int32, "device", None)


@with_requires("numpy>=2.4")
@pytest.mark.parametrize(
    "data, src_dt, dst_dt",
    [
        ([1, 2, 3], dpt.int32, dpt.int8),  # exact integer downcast
        ([2.0, 3.0], dpt.float32, dpt.int64),  # float with integral values
        ([1, 2, 3], dpt.int32, dpt.float32),  # exact int -> float
        ([2, 0, 5], dpt.int32, dpt.bool),  # any numeric maps to bool
    ],
)
def test_astype_same_value_preserved(data, src_dt, dst_dt):
    q = get_queue_or_skip()
    a = np.array(data, dtype=src_dt)
    x = dpt.asarray(a, sycl_queue=q)
    r = dpt.astype(x, dst_dt, casting="same_value")
    assert r.dtype == dst_dt
    assert (dpt.asnumpy(r) == a.astype(dst_dt)).all()


@with_requires("numpy>=2.4")
@pytest.mark.parametrize(
    "data, src_dt, dst_dt",
    [
        ([1000], dpt.int32, dpt.int8),  # integer overflow
        ([1.0, 2.5], dpt.float32, dpt.int64),  # rounding of floats
        ([1e30], dpt.float32, dpt.int64),  # out-of-range float -> int
    ],
)
def test_astype_same_value_raises(data, src_dt, dst_dt):
    q = get_queue_or_skip()
    x = dpt.asarray(np.array(data, dtype=src_dt), sycl_queue=q)
    with pytest.raises(ValueError):
        dpt.astype(x, dst_dt, casting="same_value")


@with_requires("numpy>=2.4")
def test_astype_same_value_nan_inf_preserved():
    q = get_queue_or_skip()
    skip_if_dtype_not_supported(dpt.float64, q)
    a = np.array([np.nan, np.inf, -np.inf], dtype=np.float64)
    x = dpt.asarray(a, sycl_queue=q)
    r = dpt.astype(x, dpt.float32, casting="same_value")
    assert np.array_equal(dpt.asnumpy(r), a.astype(np.float32), equal_nan=True)


@with_requires("numpy>=2.4")
def test_astype_same_value_non_numeric_target_raises():
    q = get_queue_or_skip()
    x = dpt.asarray([1, 2, 3], sycl_queue=q)
    with pytest.raises(ValueError):
        dpt.astype(x, "U4", casting="same_value")


@with_requires("numpy>=2.4")
def test_astype_same_value_copy_false_same_dtype():
    q = get_queue_or_skip()
    x = dpt.asarray([1, 2, 3], dtype=dpt.int32, sycl_queue=q)
    r = dpt.astype(x, dpt.int32, casting="same_value", copy=False)
    assert r is x


def test_gh_2055():
    """
    Test that `dpt.asarray` works on contiguous NumPy arrays with `order="K"`
    when dimensions are permuted.

    See: https://github.com/IntelPython/dpctl/issues/2055
    """
    get_queue_or_skip()

    a = np.ones((2, 3, 4), dtype=dpt.int32)
    a_t = np.transpose(a, (2, 0, 1))
    r = dpt.asarray(a_t)
    assert not r.flags["C"] and not r.flags["F"]
