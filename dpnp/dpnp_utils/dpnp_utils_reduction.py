# *****************************************************************************
# Copyright (c) 2024-2025, Intel Corporation
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


import dpnp

__all__ = ["dpnp_wrap_reduction_call"]


def dpnp_wrap_reduction_call(usm_a, out, _reduction_fn, res_dt, **kwargs):
    """Wrap a reduction call from dpctl.tensor interface."""

    input_out = out
    if out is None:
        usm_out = None
    else:
        dpnp.check_supported_arrays_type(out)

        # dpctl requires strict data type matching of out array with the result
        if out.dtype != res_dt:
            out = dpnp.astype(out, res_dt, copy=False)

        usm_out = dpnp.get_usm_ndarray(out)

    kwargs["out"] = usm_out
    res_usm = _reduction_fn(usm_a, **kwargs)
    return dpnp.get_result_array(res_usm, input_out, casting="unsafe")
