import pytest

import dpnp as dp


list_of_usm_types = [
    "device",
    "shared",
    "host"
]


@pytest.mark.parametrize("usm_type", list_of_usm_types, ids=list_of_usm_types)
def test_coerced_usm_types_sum(usm_type):
    x = dp.arange(10, usm_type = "device")
    y = dp.arange(10, usm_type = usm_type)

    z = x + y
    
    assert z.usm_type == x.usm_type
    assert z.usm_type == "device"
    assert y.usm_type == usm_type
