"""Bounded transport rejects malformed row/launch metadata without hardware."""
import copy

import pytest

from tessera.compiler.prepared_native_tensor_frame import validate_frame
from tests.unit.test_native_tensor_frame_transport import frame


def bounded_frame():
    value = frame()
    value.update(schema="tessera.native.tensor_program_frame.v2",
                 row_bound=17, row_root=0, row_buffers=[1, 0, 1])
    for index in (0, 2):
        spec = value["buffers"][index]
        spec["shape"][0] = 17
        spec["bytes"] *= 17
    value["stages"][0].update(
        scalars=[17, 1, 1], geometry=[1, 2, 1, 32, 1, 1],
        scalar_row_multipliers=[1, 0, 0], grid_row_multipliers=[0, 1, 0],
        grid_row_divisors=[1, 16, 1])
    return value


def test_bounded_metadata_validates_without_allocating():
    value = bounded_frame()
    before = copy.deepcopy(value)
    validate_frame(value)
    assert value == before


@pytest.mark.parametrize("key,value", [
    ("row_bound", 0), ("row_bound", -1), ("row_bound", 2**31), ("row_bound", True),
    ("row_root", 2), ("row_root", -1), ("row_root", True),
    ("row_buffers", [1, 0]), ("row_buffers", [0, 0, 1]),
    ("row_buffers", [1, 2, 1]), ("row_buffers", [True, 0, 1]),
    ("row_buffers", [1, 1, 1]),
])
def test_bounded_metadata_rejects_invalid_capacity_bindings(key, value):
    packet = bounded_frame()
    packet[key] = value
    with pytest.raises(ValueError):
        validate_frame(packet)


@pytest.mark.parametrize("key,value", [
    ("scalar_row_multipliers", [1]),
    ("scalar_row_multipliers", [2, 0, 0]),
    ("scalar_row_multipliers", [-1, 0, 0]),
    ("scalar_row_multipliers", [2**62, 0, 0]),
    ("scalar_row_multipliers", [True, 0, 0]),
    ("grid_row_multipliers", [0, 2, 0]),
    ("grid_row_multipliers", [0, 2**62, 0]),
    ("grid_row_divisors", [1, 0, 1]),
    ("grid_row_divisors", [2, 16, 1]),
    ("grid_row_divisors", [1, -16, 1]),
])
def test_bounded_metadata_rejects_inconsistent_or_overflowing_launch_expressions(key, value):
    packet = bounded_frame()
    packet["stages"][0][key] = value
    with pytest.raises(ValueError):
        validate_frame(packet)


def test_static_schema_does_not_accept_bounded_extension_fields():
    packet = bounded_frame()
    packet["schema"] = "tessera.native.tensor_program_frame.v1"
    with pytest.raises(ValueError):
        validate_frame(packet)


@pytest.mark.parametrize("row_bound", [None, 65])
@pytest.mark.parametrize("provider_version", ["missing", "empty", "v1", "v2"])
def test_jit_checks_required_prepare_symbol_before_tracing(row_bound, provider_version, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    import tessera as ts
    from tessera import runtime
    from tessera.compiler.jit import _JIT_FALLBACK
    import tessera.compiler.native_tensor_program as packages
    import tessera.compiler.scheduled_matmul as tools
    from tests.unit.test_native_tensor_binary_producers import lhs_add
    import numpy as np

    exports = {}
    if provider_version in {"v1", "v2"}:
        exports["tessera_nvidia_native_frame_prepare_v1"] = object()
    if provider_version == "v2":
        exports["tessera_nvidia_native_frame_prepare_v2"] = object()
    provider = None if provider_version == "missing" else SimpleNamespace(**exports)
    monkeypatch.setattr(runtime, "_load_nvidia_ptx_launch", lambda: provider)
    monkeypatch.setattr(tools, "find_tessera_opt", lambda: Path("/configured/compiler"))
    fn = ts.jit(target="nvidia_sm120")(lhs_add)
    roots = (np.zeros((17, 32), np.float16), np.zeros((17, 32), np.float16),
             np.zeros((32, 11), np.float16))
    admitted = provider_version == "v2" or (provider_version == "v1" and row_bound is None)

    class TraceReached(Exception):
        pass

    def trace(*args, **kwargs):
        if not admitted:
            pytest.fail("missing preparation ABI was admitted before tracing")
        raise TraceReached

    def forbidden_compile(*args, **kwargs):
        pytest.fail("prerequisite check reached compilation")

    monkeypatch.setattr(fn, "_traced_autodiff_module", trace)
    monkeypatch.setattr(packages, "compile_native_tensor_program", forbidden_compile)
    try:
        if admitted:
            with pytest.raises(TraceReached):
                fn._try_native_tensor_program_call(roots, {}, row_bound=row_bound)
        else:
            assert fn._try_native_tensor_program_call(roots, {}, row_bound=row_bound) is _JIT_FALLBACK
        assert not fn._native_tensor_program_cache
        assert not fn._native_tensor_calls
    finally:
        fn.close_native_storage()
