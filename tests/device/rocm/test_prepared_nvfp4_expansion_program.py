"""Owning-device prepared source-preserving expansion/product programs."""
import ctypes as C
from dataclasses import replace
import json
import os

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.rocm_nvfp4_expansion_program import package_native_nvfp4_expansion_program
from tests.unit.test_native_nvfp4_expansion_program import source_graph

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires owning gfx1201 device")


def inputs_and_oracle(m, n, k, roles, output):
    rng = np.random.default_rng(m + k)
    codes = rng.integers(0, 256, (n, k // 2), np.uint8)
    scales = rng.choice([.5, 1., 2., 4.], (n, k // 16)).astype(ml_dtypes.float8_e4m3fn)
    globals_ = np.array([.25, .125], np.float64)
    a = rng.integers(-3, 4, (m, k)).astype(ml_dtypes.float8_e4m3fn)
    sa = rng.choice([.5, 1., 2.], (m, k // 16)).astype(np.float32)
    roots = [None] * 5
    for role, slot in enumerate(roles):
        roots[slot] = [codes, scales, globals_, a, sa][role]
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)
    weights = np.empty((n, k), np.float64)
    weights[:, 0::2], weights[:, 1::2] = levels[codes & 15], levels[codes >> 4]
    product = (scales.astype(np.float64) * np.where(np.arange(n) < 3, globals_[0], globals_[1])[:, None]).astype(np.float32)
    weights *= np.repeat(product.astype(np.float64), 16, axis=1)
    activation = a.astype(np.float64) * np.repeat(sa.astype(np.float64), 16, axis=1)
    dtype = np.float32 if output == "f32" else ml_dtypes.bfloat16
    return roots, (activation @ weights.T).astype(dtype)


@pytest.mark.parametrize("m,n,k", [(1, 7, 16), (19, 17, 48), (33, 23, 80)])
@pytest.mark.parametrize("output", ["f32", "bf16"])
@pytest.mark.parametrize("roles", [(0, 1, 2, 3, 4), (2, 0, 4, 1, 3)])
def test_prepared_native_expansion_reuses_owner(m, n, k, output, roles):
    assert rt._rocm_live_arch() == "gfx1201"
    source = source_graph(m, n, k, roles, output=output)
    package = package_native_nvfp4_expansion_program(source)
    roots, expected = inputs_and_oracle(m, n, k, roles, output)
    frozen = [v.copy() for v in roots]
    with package.prepare(roots, runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as prepared:
        with pytest.raises(RuntimeError, match="status 10"):
            prepared.invoke_consumer()
        imported, _ = prepared.import_weights()
        with pytest.raises(RuntimeError):
            prepared.read(imported)
        first, _ = prepared.invoke_consumer()
        actual, = prepared.read(first)
        np.testing.assert_array_equal(actual, expected)
        retained = actual.copy()
        second, _ = prepared.invoke(repeats=3)
        assert second > first
        np.testing.assert_array_equal(prepared.read(second)[0], expected)
        # Only the copied host activation changes; full update remains explicit.
        roots[roles[3]] = -roots[roles[3]]
        prepared.update_activations(roots[roles[3]], roots[roles[4]])
        partial, _ = prepared.invoke_consumer()
        np.testing.assert_array_equal(prepared.read(partial)[0], -expected)
        prepared.update(roots)
        with pytest.raises(RuntimeError, match="status 10"):
            prepared.invoke_consumer()
        with pytest.raises(RuntimeError):
            prepared.read(second)
        third, _ = prepared.invoke()
        np.testing.assert_array_equal(prepared.read(third)[0], -expected)
        np.testing.assert_array_equal(actual, retained)
        # A malformed import must fail before its native prepare call.
        invalid = list(roots)
        invalid[roles[2]] = np.array([0., .125], np.float64)
        with pytest.raises(ValueError):
            package.prepare(invalid, runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"])
    assert prepared.handle.value == 0
    for i, value in enumerate(frozen):
        if i != roles[3]:
            np.testing.assert_array_equal(roots[i], value)


def test_prepared_rejects_modified_program_before_loading_library(monkeypatch):
    assert rt._rocm_live_arch() == "gfx1201"
    package = package_native_nvfp4_expansion_program(source_graph())
    plan = json.loads(package.program_json)
    plan["steps"][1]["inputs"][1] = 6
    modified = replace(package, program_json=json.dumps(plan))
    monkeypatch.setattr(C, "CDLL", lambda *_: pytest.fail("invalid program reached native library"))
    roots, _ = inputs_and_oracle(19, 7, 48, (0, 1, 2, 3, 4), "f32")
    with pytest.raises(ValueError, match="binding differs"):
        modified.prepare(roots, runtime_library="/must/not/load")


@pytest.mark.parametrize("bad", ["duplicate", "private_slot", "wrong_bytes", "zero_count", "too_many"])
def test_partial_update_rejection_is_transactional(bad):
    package = package_native_nvfp4_expansion_program(source_graph())
    roots, expected = inputs_and_oracle(19, 7, 48, (0, 1, 2, 3, 4), "f32")
    with package.prepare(roots, runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as prepared:
        prepared.import_weights()
        generation, _ = prepared.invoke_consumer()
        # Bind the additive API through an ordinary valid update first.
        prepared.update_activations(roots[3], roots[4])
        generation, _ = prepared.invoke_consumer()
        slots = (C.c_uint32 * 2)(3, 4)
        pointers = (C.c_void_p * 2)(roots[3].ctypes.data, roots[4].ctypes.data)
        sizes = (C.c_uint64 * 2)(roots[3].nbytes, roots[4].nbytes)
        count = 2
        if bad == "duplicate":
            slots[1] = 3
        elif bad == "private_slot":
            slots[0] = 5
        elif bad == "wrong_bytes":
            sizes[1] += 1
        elif bad == "zero_count":
            count = 0
        else:
            count = 6
        assert prepared.lib.tessera_rocm_program_update_slots(
            prepared.handle, count, slots, pointers, sizes) == 1
        np.testing.assert_array_equal(prepared.read(generation)[0], expected)


def test_partial_weight_update_invalidates_resident_product():
    package = package_native_nvfp4_expansion_program(source_graph())
    roots, expected = inputs_and_oracle(19, 7, 48, (0, 1, 2, 3, 4), "f32")
    with package.prepare(roots, runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as prepared:
        prepared.import_weights()
        generation, _ = prepared.invoke_consumer()
        prepared.update_activations(roots[3], roots[4])
        update = prepared.lib.tessera_rocm_program_update_slots
        changed_codes = np.zeros_like(roots[0])
        assert update(prepared.handle, 1, (C.c_uint32 * 1)(0),
                      (C.c_void_p * 1)(changed_codes.ctypes.data),
                      (C.c_uint64 * 1)(changed_codes.nbytes)) == 0
        with pytest.raises(RuntimeError, match="status 10"):
            prepared.invoke_consumer()
        prepared.import_weights()
        generation, _ = prepared.invoke_consumer()
        np.testing.assert_array_equal(prepared.read(generation)[0], np.zeros_like(expected))


def frontend_product(codes, scales, globals_, activation, activation_scales):
    import tessera as ts
    from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy
    weight, weight_scales = ts.ops.nvfp4_expand(
        codes, scales, globals_, row_offsets=[0, 3, 7],
        numeric_policy=nvfp4_expansion_policy())
    return ts.ops.scaled_matmul(
        activation, weight, activation_scales, weight_scales, transposeB=True,
        numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
        scale_layout={"granularity": "block", "block": [1, 16], "format": "fp32"})


def test_normal_frontend_capture_to_prepared_native(monkeypatch):
    from tessera.compiler import nvfp4_expansion, reference_typed_scaled_matmul
    from tessera.compiler.trace import trace, to_graph_ir_module
    def forbidden(*a, **kw):
        pytest.fail("native composed program called an eager numerical oracle")
    monkeypatch.setattr(nvfp4_expansion, "reference_nvfp4_expand", forbidden)
    monkeypatch.setattr(reference_typed_scaled_matmul, "reference_typed_scaled_matmul", forbidden)
    roots, expected = inputs_and_oracle(19, 7, 48, (0, 1, 2, 3, 4), "f32")
    module = to_graph_ir_module(trace(frontend_product, *roots), target="rocm_gfx1201")
    package = package_native_nvfp4_expansion_program(module)
    with package.prepare(roots, runtime_library=os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]) as prepared:
        prepared.import_weights()
        generation, _ = prepared.invoke_consumer()
        np.testing.assert_array_equal(prepared.read(generation)[0], expected)


def test_cached_allocation_cannot_reuse_previous_session_values(monkeypatch):
    monkeypatch.setenv("TESSERA_ROCM_PROGRAM_CACHE", "1")
    package = package_native_nvfp4_expansion_program(source_graph())
    roots, expected = inputs_and_oracle(19, 7, 48, (0, 1, 2, 3, 4), "f32")
    provider = os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]
    with package.prepare(roots, runtime_library=provider) as first:
        first.import_weights()
        generation, _ = first.invoke_consumer()
        np.testing.assert_array_equal(first.read(generation)[0], expected)
        lib = first.lib
        stats = lib.tessera_rocm_program_cache_stats
        stats.argtypes = [C.POINTER(C.c_uint64)] * 4
        counts = [C.c_uint64() for _ in range(4)]
        assert stats(*[C.byref(x) for x in counts]) == 0
        previous_hits = counts[0].value
    # Exact same image/geometry frame must acquire the cached allocation.
    with package.prepare(roots, runtime_library=provider) as second:
        assert stats(*[C.byref(x) for x in counts]) == 0
        assert counts[0].value == previous_hits + 1
        with pytest.raises(RuntimeError, match="status 10"):
            second.invoke_consumer()
        second.import_weights()
        generation, _ = second.invoke_consumer()
        np.testing.assert_array_equal(second.read(generation)[0], expected)


def test_prepared_checks_descriptor_geometry_against_native_target():
    package = package_native_nvfp4_expansion_program(source_graph())
    provenance = dict(package.consumer.descriptor.provenance)
    provenance["workgroup"] = [64, 1, 1]
    consumer = replace(package.consumer,
                       descriptor=replace(package.consumer.descriptor, provenance=provenance))
    modified = replace(package, consumer=consumer)
    # Self-consistent metadata digest is not a substitute for native ABI checks.
    modified = replace(modified, binding_digest=modified._identity())
    with pytest.raises(ValueError, match="geometry/ABI"):
        modified.validate()
