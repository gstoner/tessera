"""Public resident source-active tracing must preserve device roots."""
import numpy as np
import ml_dtypes
import pytest
import tessera as ts
from tessera import runtime as rt
from benchmarks.nvidia.active_source_jvp_reference import product, product_reversed


class MetadataRoot:
    def __init__(self, shape, dtype, ordinal):
        self.dtype = np.dtype(dtype)
        self.interface = {
            "version": 3, "shape": tuple(shape), "typestr": self.dtype.str,
            "strides": None, "data": (0x100000 + ordinal * 0x10000, True),
            "stream": 0x100 + ordinal,
        }

    @property
    def __cuda_array_interface__(self):
        return dict(self.interface)

    def __array__(self, *args, **kwargs):
        raise AssertionError("resident source-active frontend read a device root")


@pytest.mark.parametrize("dtype", [np.float16, ml_dtypes.bfloat16])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("keywords", [False, True])
def test_source_active_resident_frontend_uses_typed_trace_without_host_reads(
    dtype, permuted, keywords, monkeypatch,
):
    import tessera.compiler.prepared_nvidia_active_source as prepared

    a, b, seed = [MetadataRoot(shape, dtype, i)
                  for i, shape in enumerate(((17, 35), (35, 19), (17, 35)))]
    roots = (b, a) if permuted else (a, b)
    observed = []

    class Owner:
        def __init__(self, source):
            observed.append(source)
            self._public_runtime_artifact = object()
            self.receipt = {"native_call_binding": "prepared_cpp_active_source_program"}

        def __call__(self, values):
            assert all(actual is expected for actual, expected
                       in zip(values, (*roots, seed), strict=True))
            return np.zeros((17, 19), np.float32), np.zeros((17, 19), np.float32)

        def close(self):
            pass

    monkeypatch.setattr(prepared, "PreparedActiveSourceJVP", Owner)
    monkeypatch.setattr(rt, "_load_nvidia_ptx_launch",
                        lambda: pytest.fail("metadata test accessed CUDA provider"))
    fn = ts.jit(target="nvidia_sm120", autodiff="forward", wrt=("a",))(
        product_reversed if permuted else product)
    try:
        for _ in range(2):
            outputs = (fn.native_jvp(**dict(zip(fn.arg_names, roots, strict=True)),
                                     tangents=(seed,)) if keywords
                       else fn.native_jvp(*roots, tangents=(seed,)))
            assert all(output.shape == (17, 19) for output in outputs)
        assert len(observed) == 1
        assert 'tessera.rmsnorm' in observed[0]
        assert 'tessera.relu' in observed[0] and 'tessera.matmul' in observed[0]
        assert f"tensor<17x35x{'bf16' if np.dtype(dtype).name == 'bfloat16' else 'f16'}>" in observed[0]
        assert fn.last_frontend_differential is not None
    finally:
        fn.close_native_storage()
