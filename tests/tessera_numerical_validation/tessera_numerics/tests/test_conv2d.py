import numpy as np
import pytest
from tessera_numerics.validators import check_allclose
from tessera_numerics import tessera_adapter as A

def ref_conv2d_nhwc(x, w, stride=(1,1), padding=(0,0)):
    # Use adapter's numpy path (same math as test target but float64 accum)
    return A.conv2d_nhwc(x.astype(np.float64), w.astype(np.float64), stride=stride, padding=padding, dtype="float64")

@pytest.mark.parametrize("case", [0,1])
@pytest.mark.parametrize("dtype", ["float32","float16","bfloat16"])
def test_conv2d(cfg, rec, case, dtype):
    cases = cfg["conv2d_nhwc"]["shapes"]
    xshape = cases[case]["x"]
    wshape = cases[case]["w"]
    stride = tuple(cases[case]["stride"])
    padding = tuple(cases[case]["padding"])
    rng = A.rng(123)
    x = (rng.standard_normal(xshape)).astype(np.float64)*0.1
    w = (rng.standard_normal(wshape)).astype(np.float64)*0.1

    got = A.conv2d_nhwc(x, w, stride=stride, padding=padding, dtype=dtype)
    ref = ref_conv2d_nhwc(x, w, stride=stride, padding=padding)
    check_allclose("conv2d_nhwc", got, ref.astype(got.dtype), dtype=dtype)
    rec("conv2d_nhwc", {"x": xshape, "w": wshape, "dtype": dtype})
