"""The NVFP4 row-bound request never reaches the cached frontend trace.

GFX1201-NVFP4-BOUND-TRACE-COPY-20261028. ``_trace_frontend_capture`` returns
the cached specialization itself; a bound attached to it would leak into every
later call and compile for the same signature.
"""
import pytest

from tests.device.rocm.test_nvfp4_resident_jit import make_function
from tests.unit.test_rocm_nvfp4_resident import inputs_and_oracle


def _cached_bounds(fn):
    return [m.module_attrs.get("tessera.native.nvfp4_m_bound")
            for m in fn._traced_frontend_specializations.values()]


def test_bound_request_never_reaches_the_cached_trace(rocm_image_toolchain):
    fn = make_function(32, 64)
    args = inputs_and_oracle(17, 32, 64)[0]
    with pytest.raises(RuntimeError, match="exceed their checked storage capacity"):
        fn.compile_native_nvfp4_program(*args, m_bound=16)
    assert _cached_bounds(fn) == [None]
    assert getattr(fn, "_rocm_nvfp4_bounded_program", None) is None
    bounded = fn.compile_native_nvfp4_program(*args, m_bound=257)
    assert bounded.native.consumer.m == 257 and _cached_bounds(fn) == [None]
    static = fn.compile_native_nvfp4_program(*args)
    assert static.native.consumer.m == 17


@pytest.mark.parametrize("bound", [0, -1, 2.0, True, "8"])
def test_malformed_bound_is_refused_before_tracing(bound, monkeypatch):
    fn = make_function(32, 64)
    monkeypatch.setattr(fn, "_trace_frontend_capture",
                        lambda *a, **k: pytest.fail("traced a malformed bound"))
    with pytest.raises(ValueError, match="positive integer"):
        fn.compile_native_nvfp4_program(*inputs_and_oracle(17, 32, 64)[0], m_bound=bound)
