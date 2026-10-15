"""Original-checkpoint public JIT/replay/prepared NVFP4 composition proof."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import ml_dtypes
import numpy as np
import tessera as ts
from tessera import runtime as rt
from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy
from benchmarks.rocm import benchmark_rocm_nvfp4_checkpoint as checkpoint
from benchmarks.rocm import benchmark_gfx1201_three_formats as formats
from benchmarks.rocm.checkpoint_activations import load_capture


def run(args):
    pool, activation_source = load_capture(args.activations, [args.m])
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("requires owning gfx1201")
    loaded = [checkpoint._load_projection("model.layers.0.mlp." + role + ".weight")
              for role in ("gate_proj", "up_proj")]
    projections = [row["projection"] for row in loaded]
    sources = [row["source"] for row in loaded]
    base = np.concatenate([np.asarray(row["bf16"], np.float32) for row in loaded])
    codes = np.ascontiguousarray(np.concatenate([p.packed_codes for p in projections]))
    scales = np.ascontiguousarray(np.concatenate([p.e4m3_scales for p in projections]))
    globals_ = np.array([p.global_scale for p in projections], np.float64)
    offsets = [0, projections[0].packed_codes.shape[0], codes.shape[0]]
    n, packed_k = codes.shape
    k = packed_k * 2
    a = np.ascontiguousarray(pool[:args.m], dtype=np.float32)
    qa, _, encoded, da, _ = formats.quantize_fp8(a, np.zeros((k, 1), np.float32), 16, 1)
    sa = encoded[0]
    roots = [codes, scales, globals_, qa, sa]
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float64)
    weights = np.empty((n, k), np.float64)
    weights[:, 0::2], weights[:, 1::2] = levels[codes & 15], levels[codes >> 4]
    row_globals = np.where(np.arange(n) < offsets[1], globals_[0], globals_[1])
    scale64 = scales.astype(np.float64) * row_globals[:, None]
    represented = weights * np.repeat(scale64.astype(np.float32).astype(np.float64), 16, axis=1)
    shipped = weights * np.repeat(scale64, 16, axis=1)
    expected = da @ represented.T
    magnitude = np.abs(da) @ np.abs(represented).T
    original = a.astype(np.float64) @ base.astype(np.float64).T
    shipped_output = a.astype(np.float64) @ shipped.T
    gamma = (k + 8 * (k // 16) + 16) * 2.**-24
    gamma /= 1 - gamma
    bound = gamma * magnitude + 2.**-126
    del loaded, projections, base, represented, shipped, weights

    from tessera.compiler import nvfp4_expansion, reference_typed_scaled_matmul
    def forbidden(*a, **kw):
        raise AssertionError("native checkpoint execution called an eager numerical oracle")
    nvfp4_expansion.reference_nvfp4_expand = forbidden
    reference_typed_scaled_matmul.reference_typed_scaled_matmul = forbidden

    @ts.jit(target="rocm_gfx1201")
    def product(codes, scales, globals_, activation, activation_scales):
        b, sb = ts.ops.nvfp4_expand(codes, scales, globals_,
            row_offsets=offsets, numeric_policy=nvfp4_expansion_policy())
        return ts.ops.scaled_matmul(activation, b, activation_scales, sb,
            transposeB=True, numeric_policy={"accum": "fp32", "execution_mode": "exact_per_block"},
            scale_layout={"granularity": "block", "block": [1, 16], "format": "fp32"})

    def verify(value):
        value = np.asarray(value)
        if (value.shape != expected.shape or value.dtype != np.float32
                or not np.isfinite(value).all()
                or np.any(np.abs(value.astype(np.float64) - expected) > bound)):
            raise AssertionError("native checkpoint product exceeds independent FP32 arithmetic bound")
        return value

    started = time.perf_counter_ns()
    first = verify(product(*roots))
    cold_ms = (time.perf_counter_ns() - started) / 1e6
    assert product.execution_kind == "native_gpu"
    artifact = product.runtime_artifact()
    serialized = artifact.to_json()
    restored = rt.RuntimeArtifact.from_json(serialized)
    program = product._rocm_nvfp4_last_program
    package = program.native
    provider = rt._load_rocm_native_movement_runtime()
    if provider is None:
        raise RuntimeError("missing native provider")
    host = {"public_jit": [], "serialized_replay": [], "prepared_activation_consumer": []}
    cache_samples = {key: [] for key in host}
    cache_stats = provider.tessera_rocm_program_cache_stats
    cache_stats.argtypes = [C.POINTER(C.c_uint64)] * 4
    def cache_snapshot():
        values = [C.c_uint64() for _ in range(4)]
        if cache_stats(*[C.byref(v) for v in values]):
            raise RuntimeError("native allocation-cache query failed")
        return dict(zip(("hits", "misses", "idle_entries", "idle_bytes"),
                        (v.value for v in values)))
    digests = []
    sha = lambda x: hashlib.sha256(x).hexdigest()
    prep_started = time.perf_counter_ns()
    with package.prepare(roots, runtime_library=provider._name) as prepared:
        prepare_ms = (time.perf_counter_ns() - prep_started) / 1e6
        _, import_ms = prepared.import_weights(timed=True)
        generation, _ = prepared.invoke_consumer()
        verify(prepared.read(generation)[0])
        _, pilot = prepared.profile_members(repeats=1)
        repeats = min(65536, max(64, math.ceil(30. / min(pilot))))
        for _ in range(3):
            _, calibration = prepared.profile_members(repeats=repeats)
            if min(calibration) * repeats >= 25.:
                break
            repeats = min(65536, math.ceil(30. / min(calibration)))
        else:
            raise RuntimeError("member timing windows failed calibration")
        stage_samples = []
        for _ in range(3):
            generation, timing = prepared.profile_members(repeats=repeats)
            if min(timing) * repeats < 20.:
                raise RuntimeError("member timing window shorter than 20 ms")
            stage_samples.append(list(timing))
            verify(prepared.read(generation)[0])
        for window in range(3):
            order = list(host) if window % 2 == 0 else list(reversed(host))
            for mode in order:
                cache_before = cache_snapshot()
                started = time.perf_counter_ns()
                if mode == "public_jit":
                    value = product(*roots)
                elif mode == "serialized_replay":
                    result = rt.launch(restored, roots)
                    if not result.get("ok") or result["execution_kind"] != "native_gpu":
                        raise RuntimeError(result)
                    value = result["output"]
                else:
                    prepared.update_activations(qa, sa)
                    generation, _ = prepared.invoke_consumer()
                    value, = prepared.read(generation)
                host[mode].append((time.perf_counter_ns() - started) / 1e6)
                cache_after = cache_snapshot()
                cache_samples[mode].append({
                    key: cache_after[key] - cache_before[key] for key in ("hits", "misses")})
                verify(value)
                np.testing.assert_array_equal(value, first)
                digests.append(sha(value.tobytes()))
    hip = rt._load_hip_for_launch()
    device, name = C.c_int(), C.create_string_buffer(256)
    if hip.hipGetDevice(C.byref(device)) or hip.hipDeviceGetName(name, len(name), device.value):
        raise RuntimeError("missing exact-device identity")
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), root / "python/tessera/compiler/rocm_nvfp4_source_frontend.py",
             root / "python/tessera/compiler/rocm_nvfp4_expansion_program.py",
             root / "python/tessera/compiler/rocm_nvfp4_program.py",
             root / "python/tessera/compiler/jit.py", root / "python/tessera/runtime.py"]
    record = {
        "schema": "tessera.gfx1201.original_nvfp4_composed.v1",
        "architecture": rt._rocm_live_arch(), "device": name.value.decode(),
        "shape_mnk": [args.m, n, k], "checkpoint_sources": sources,
        "activation_source": activation_source, "projection_row_offsets": offsets,
        "projection_globals": globals_.tolist(), "output_storage": "fp32",
        "public_cold_call_ms": cold_ms, "prepare_upload_ms": prepare_ms,
        "initial_import_native_event_ms": import_ms,
        "completed_call_samples_ms": host, "allocation_cache_delta_samples": cache_samples,
        "native_device_buffer_bytes": sum(b["bytes"] for b in json.loads(package.program_json)["buffers"]),
        "expanded_weight_bytes": sum(b["bytes"] for b in json.loads(package.program_json)["buffers"][5:7]),
        "per_call_upload_bytes": {
            "public_jit": sum(v.nbytes for v in roots),
            "serialized_replay": sum(v.nbytes for v in roots),
            "prepared_activation_consumer": qa.nbytes + sa.nbytes},
        "completed_call_median_ms": {key: statistics.median(v) for key, v in host.items()},
        "member_graph_dispatch_samples_ms": stage_samples, "member_repeats": repeats,
        "native_vs_bf16_projection": formats.errors(first, original),
        "shipped_weights_vs_bf16_projection": formats.errors(shipped_output, original),
        "native_vs_decoded_fp8_projection": formats.errors(first, expected),
        "arithmetic_bound_violations": int(np.count_nonzero(np.abs(first.astype(np.float64) - expected) > bound)),
        "all_mode_output_sha256": sorted(set(digests)), "artifact_bytes": len(serialized.encode()),
        "binding_digest": package.binding_digest,
        "compile_report": product.compile_report().as_dict(),
        "compiler_sha256": sha(Path(os.environ["TESSERA_OPT"]).read_bytes()),
        "compiler_version": subprocess.check_output([os.environ["TESSERA_OPT"], "--version"], text=True).strip(),
        "provider_sha256": sha(Path(provider._name).read_bytes()),
        "component_images": [p.image.image_digest for p in product.native_nvfp4_packages()],
        "source_sha256": {str(p.relative_to(root)): sha(p.read_bytes()) for p in paths},
        "full_model_quality_acceptance": False, "selector_promotion": False,
        "timing_scope": "host excludes oracle verification; graph includes device dispatch; startup and steady-state separate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"timings": record["completed_call_median_ms"], "quality": record["native_vs_bf16_projection"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
