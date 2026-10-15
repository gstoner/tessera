"""Diagnostic ABI proof for a native stage with two genuine SSA outputs.

This tests native ownership, not a compiled producer-to-matmul program.
"""
import ctypes as C
import os

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler.native_scaled_program import _Buffer
from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy, reference_nvfp4_expand
from tessera.compiler.rocm_nvfp4_expand_native import (
    build_nvfp4_expansion_graph, package_nvfp4_expansion,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1",
    reason="requires the gfx1201 owning device")


class MultiOutputStep(C.Structure):
    _fields_ = [
        ("image", C.c_void_p), ("image_bytes", C.c_uint64), ("entry", C.c_char_p),
        ("input_count", C.c_uint32), ("scalar_count", C.c_uint32),
        ("inputs", C.c_uint32*6), ("output_count", C.c_uint32),
        ("outputs", C.c_uint32*7), ("geometry", C.c_uint32*6),
        ("scalars", C.c_int64*8),
    ]


@pytest.fixture(scope="module")
def lib():
    assert rt._rocm_live_arch() == "gfx1201"
    path = os.environ.get("TESSERA_ROCM_MULTI_OUTPUT_LIB")
    if not path:
        pytest.fail("owning-device multi-output provider is required")
    library = C.CDLL(path)
    library.tessera_rocm_program_prepare_multi_output.argtypes = [
        C.c_char_p, C.c_uint32, C.c_uint32, C.POINTER(_Buffer), C.c_uint32,
        C.POINTER(MultiOutputStep), C.POINTER(C.c_void_p),
        C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
    library.tessera_rocm_program_invoke.argtypes = [
        C.c_uint64, C.c_uint32, C.POINTER(C.c_uint64), C.POINTER(C.c_float)]
    library.tessera_rocm_program_read_many.argtypes = [
        C.c_uint64, C.c_uint64, C.c_uint32, C.POINTER(C.c_uint32),
        C.POINTER(C.c_void_p), C.POINTER(C.c_uint64)]
    library.tessera_rocm_program_close.argtypes = [C.c_uint64]
    for name in ("prepare_multi_output", "invoke", "read_many", "close"):
        getattr(library, "tessera_rocm_program_" + name).restype = C.c_int
    return library


@pytest.fixture(scope="module")
def package():
    return package_nvfp4_expansion(build_nvfp4_expansion_graph(
        7, 48, [0, 3, 7], numeric_policy=nvfp4_expansion_policy()))


def frame(package):
    codes = np.resize(np.arange(256, dtype=np.uint8), (7, 24))
    scales = np.resize(np.arange(127, dtype=np.uint8), (7, 3)).view(ml_dtypes.float8_e4m3fn)
    scales[0, 0] = -0.
    globals_ = np.array([.000123456789, .007654321], np.float64)
    roots = [codes, scales, globals_]
    expected = reference_nvfp4_expand(*roots, row_offsets=[0, 3, 7],
                                      numeric_policy=nvfp4_expansion_policy())
    arrays = roots + [np.full_like(expected[0], -1), np.full_like(expected[1], -1)]
    buffers = (_Buffer*5)(*[
        _Buffer(a.nbytes, a.size, -1 if i < 3 else 0, 0 if i < 3 else 1,
                0 if i < 3 else 2, 0) for i, a in enumerate(arrays)])
    blob = C.create_string_buffer(package.native.image.payload)
    entry = package.native.descriptor.entry_symbol.encode()
    step = MultiOutputStep()
    step.image = C.cast(blob, C.c_void_p)
    step.image_bytes = len(package.native.image.payload)
    step.entry = entry
    step.input_count, step.output_count = 3, 2
    step.inputs[:3], step.outputs[:2] = (0, 1, 2), (3, 4)
    geometry = package.native.descriptor.geometry
    step.geometry[:] = (*geometry.grid, *geometry.workgroup)
    pointers = (C.c_void_p*3)(*[a.ctypes.data for a in roots])
    sizes = (C.c_uint64*3)(*[a.nbytes for a in roots])
    # Keep every borrowed host/image object alive through native admission.
    return arrays, expected, buffers, step, pointers, sizes, blob, entry


def prepare(lib, frame_):
    _, _, buffers, step, pointers, sizes, _, _ = frame_
    handle = C.c_uint64()
    status = lib.tessera_rocm_program_prepare_multi_output(
        b"gfx1201", 3, 5, buffers, 1, C.byref(step), pointers, sizes, C.byref(handle))
    return status, handle


def test_native_two_output_storage_and_transactional_read(lib, package):
    data = frame(package)
    arrays, expected = data[:2]
    status, handle = prepare(lib, data)
    try:
        assert status == 0 and handle.value
        generation = C.c_uint64()
        assert lib.tessera_rocm_program_invoke(handle, 1, C.byref(generation), None) == 0
        outputs = (C.c_void_p*2)(*[a.ctypes.data for a in arrays[3:]])
        sizes = (C.c_uint64*2)(*[a.nbytes for a in arrays[3:]])
        slots = (C.c_uint32*2)(3, 4)
        assert lib.tessera_rocm_program_read_many(
            handle, generation, 2, slots, outputs, sizes) == 0
        np.testing.assert_array_equal(arrays[3].view(np.uint8), expected[0].view(np.uint8))
        np.testing.assert_array_equal(arrays[4].view(np.uint32), expected[1].view(np.uint32))
        # A failed batched read must not partially publish either host output.
        arrays[3].view(np.uint8).fill(31)
        arrays[4].view(np.uint32).fill(31)
        duplicate = (C.c_uint32*2)(3, 3)
        assert lib.tessera_rocm_program_read_many(
            handle, generation, 2, duplicate, outputs, sizes) != 0
        assert np.all(arrays[3].view(np.uint8) == 31)
        assert np.all(arrays[4].view(np.uint32) == 31)
    finally:
        if handle.value:
            assert lib.tessera_rocm_program_close(handle) == 0


@pytest.mark.parametrize("bad", [
    "zero_results", "too_many_results", "duplicate_result", "reordered_results",
    "omitted_result", "future_input", "wrong_write", "wrong_last_read", "too_many_scalars",
])
def test_multi_output_ssa_admission(lib, package, bad):
    data = frame(package)
    buffers, step = data[2:4]
    if bad == "zero_results":
        step.output_count = 0
    elif bad == "too_many_results":
        step.output_count = 7
    elif bad == "duplicate_result":
        step.outputs[1] = 3
    elif bad == "reordered_results":
        step.outputs[:2] = (4, 3)
    elif bad == "omitted_result":
        step.output_count = 1
    elif bad == "future_input":
        step.inputs[0] = 3
    elif bad == "wrong_write":
        buffers[4].first_write = 1
    elif bad == "wrong_last_read":
        buffers[4].last_read = 0
    else:
        step.scalar_count = 9
    status, handle = prepare(lib, data)
    try:
        assert status == 1
        assert handle.value == 0
    finally:
        if handle.value:
            assert lib.tessera_rocm_program_close(handle) == 0


@pytest.mark.parametrize("mnk", [(1, 7, 16), (19, 17, 48), (33, 23, 80)])
def test_native_expansion_product_private_edges(lib, mnk):
    """Execute the native compiler's two-stage SSA plan without intermediate IO."""
    from tessera.compiler.native_nvfp4_program import export_native_nvfp4_program
    from tessera.compiler.rocm_fp8_blockscale import (
        BlockScaleProgram, BlockScaleShape, package_blockscale,
    )
    from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
    from tests.unit.test_native_nvfp4_expansion_program import source_graph

    m, n, k = mnk
    program = export_native_nvfp4_program(source_graph(m, n, k))
    manifest = program.manifest
    producer = package_nvfp4_expansion(program.project_member(0)).native
    consumer_graph = program.project_member(1)
    tool = find_tessera_opt()
    schedule = run_tessera_opt(tool, consumer_graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool, schedule, "--tessera-schedule-to-tile")
    consumer = package_blockscale(BlockScaleProgram(
        BlockScaleShape(m, n, k, 16, 1, "nk", "f32"),
        "w8a8_blockscale", consumer_graph, schedule, tile))
    rng = np.random.default_rng(m + k)
    codes = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    scales = rng.choice([.5, 1., 2., 4.], (n, k // 16)).astype(ml_dtypes.float8_e4m3fn)
    globals_ = np.array([.25, .125], np.float64)
    a = rng.integers(-3, 4, (m, k)).astype(ml_dtypes.float8_e4m3fn)
    sa = rng.choice([.5, 1., 2.], (m, k // 16)).astype(np.float32)
    roots = [codes, scales, globals_, a, sa]
    # Independent nibble decode and mathematical product; no producer oracle.
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)
    weights = np.empty((n, k), np.float64)
    weights[:, 0::2] = levels[codes & 15]
    weights[:, 1::2] = levels[codes >> 4]
    global_rows = np.where(np.arange(n) < 3, globals_[0], globals_[1])
    plane = (scales.astype(np.float64) * global_rows[:, None]).astype(np.float32)
    weights *= np.repeat(plane.astype(np.float64), 16, axis=1)
    activation = a.astype(np.float64) * np.repeat(sa.astype(np.float64), 16, axis=1)
    expected = activation @ weights.T
    contracts = (_Buffer * len(manifest["buffers"]))()
    for i, b in enumerate(manifest["buffers"]):
        elements = int(np.prod(b["shape"]))
        contracts[i] = _Buffer(b["bytes"], elements, b["first_write"], b["last_read"],
                               {"readonly_input": 0, "private_scratch": 1, "returned_output": 2}[b["ownership"]], 0)
    steps = (MultiOutputStep * 2)()
    blobs, entries = [], []
    for index, package_ in enumerate((producer, consumer)):
        descriptor = package_.descriptor
        blobs.append(C.create_string_buffer(package_.image.payload))
        entries.append(descriptor.entry_symbol.encode())
        step = steps[index]
        step.image, step.image_bytes = C.cast(blobs[-1], C.c_void_p), len(package_.image.payload)
        step.entry = entries[-1]
        inputs, outputs = manifest["steps"][index]["inputs"], manifest["steps"][index]["outputs"]
        step.input_count, step.output_count = len(inputs), len(outputs)
        step.inputs[:len(inputs)], step.outputs[:len(outputs)] = inputs, outputs
        if index == 0:
            step.geometry[:] = (*descriptor.geometry.grid, *descriptor.geometry.workgroup)
        else:
            bm, bn = descriptor.provenance["macro_tile"]
            step.geometry[:] = ((n + bn - 1) // bn, (m + bm - 1) // bm, 1,
                                *descriptor.provenance["workgroup"])
            step.scalar_count = 3
            step.scalars[:3] = (m, n, k)
    pointers = (C.c_void_p * 5)(*[v.ctypes.data for v in roots])
    sizes = (C.c_uint64 * 5)(*[v.nbytes for v in roots])
    handle, generation = C.c_uint64(), C.c_uint64()
    status = lib.tessera_rocm_program_prepare_multi_output(
        b"gfx1201", 5, len(contracts), contracts, 2, steps, pointers, sizes, C.byref(handle))
    try:
        assert status == 0 and handle.value
        assert lib.tessera_rocm_program_invoke(handle, 1, C.byref(generation), None) == 0
        output = np.full((m, n), np.nan, np.float32)
        slots = (C.c_uint32 * 1)(manifest["output"])
        destinations = (C.c_void_p * 1)(output.ctypes.data)
        bytes_ = (C.c_uint64 * 1)(output.nbytes)
        assert lib.tessera_rocm_program_read_many(
            handle, generation, 1, slots, destinations, bytes_) == 0
        np.testing.assert_allclose(output.astype(np.float64), expected, rtol=1e-6, atol=1e-6)

        evidence_dir = os.environ.get("TESSERA_MULTI_OUTPUT_EVIDENCE_DIR")
        if evidence_dir:
            import hashlib
            import json
            from pathlib import Path
            import statistics
            import time

            lib.tessera_rocm_program_profile_members.argtypes = [
                C.c_uint64, C.c_uint32, C.c_uint32, C.POINTER(C.c_uint64), C.POINTER(C.c_float)]
            lib.tessera_rocm_program_profile_members.restype = C.c_int
            samples = []
            host_ms = []
            repeats = 4096
            for _ in range(3):
                times = (C.c_float * 2)()
                assert lib.tessera_rocm_program_profile_members(
                    handle, repeats, 2, C.byref(generation), times) == 0
                assert all(np.isfinite(x) and x > 0 for x in times)
                samples.append(list(times))
                start = time.perf_counter_ns()
                assert lib.tessera_rocm_program_invoke(
                    handle, 1, C.byref(generation), None) == 0
                host_ms.append((time.perf_counter_ns() - start) / 1e6)
            assert lib.tessera_rocm_program_read_many(
                handle, generation, 1, slots, destinations, bytes_) == 0
            np.testing.assert_allclose(output.astype(np.float64), expected, rtol=1e-6, atol=1e-6)
            hip = rt._load_hip_for_launch()
            ordinal, device_name = C.c_int(), C.create_string_buffer(256)
            assert hip.hipGetDevice(C.byref(ordinal)) == 0
            assert hip.hipDeviceGetName(device_name, len(device_name), ordinal.value) == 0
            sha = lambda data: hashlib.sha256(data).hexdigest()
            packet = {
                "schema": "tessera.rocm.native_multi_output_owner.diagnostic.v1",
                "architecture": rt._rocm_live_arch(), "device": device_name.value.decode(),
                "shape_mnk": list(mnk), "correctness": "independent_nibble_oracle_before_and_after_timing",
                "max_abs_error": float(np.max(np.abs(output.astype(np.float64) - expected))),
                "no_intermediate_host_transfer": True,
                "public_jit_composition": False, "import_once_execution": False,
                "compiler_sha256": sha(tool.read_bytes()),
                "provider_sha256": sha(Path(os.environ["TESSERA_ROCM_MULTI_OUTPUT_LIB"]).read_bytes()),
                "plan_sha256": sha(program.plan_json.encode()),
                "source_graph_sha256": sha(program.source_ir.encode()),
                "image_payload_sha256": [p.image.payload_digest for p in (producer, consumer)],
                "member_event_graph_dispatch_samples_ms": samples,
                "member_event_graph_dispatch_median_ms": [
                    statistics.median(x[i] for x in samples) for i in range(2)],
                "member_repeats": repeats,
                "completed_native_invoke_samples_ms": host_ms,
                "completed_native_invoke_median_ms": statistics.median(host_ms),
                "timing_scope": "per-member captured HIP graphs; native invoke excludes prepare/uploads/readback",
            }
            destination = Path(evidence_dir)
            destination.mkdir(parents=True, exist_ok=True)
            (destination / ("x".join(map(str, mnk)) + ".json")).write_text(json.dumps(packet, indent=2) + "\n")

        # The producer's two outputs are native private storage, never returned.
        private_slots = (C.c_uint32 * 1)(5)
        assert lib.tessera_rocm_program_read_many(
            handle, generation, 1, private_slots, destinations, bytes_) != 0
    finally:
        if handle.value:
            assert lib.tessera_rocm_program_close(handle) == 0
