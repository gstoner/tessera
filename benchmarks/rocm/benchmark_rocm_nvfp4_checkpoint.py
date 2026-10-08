#!/usr/bin/env python3
"""Exact gfx1201 NVFP4 checkpoint ingest and scheduled-consumer proof.

Fetches only the named Safetensors tensor ranges from pinned public model
revisions. No checkpoint shard is stored on disk.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time
import urllib.request

import ml_dtypes
import numpy as np

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler import rocm_nvfp4_ingest as ingest
from tessera.compiler import rocm_nvfp4_ingest_native as native_ingest
from benchmarks.rocm import benchmark_rocm_nvfp4_ingest_schedule as scheduled

NVFP4_REPO = "nvidia/Qwen3-8B-NVFP4"
NVFP4_REVISION = "ccd10a893cbca613259517c3efe08e151ddf2b8e"
BF16_REPO = "Qwen/Qwen3-8B"
BF16_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TENSOR = "model.layers.0.self_attn.q_proj.weight"


def _url(repo: str, revision: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{name}"


def _fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "tessera-nvfp4-ingest-validation"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"checkpoint server ignored byte range: HTTP {response.status}")
        data = response.read()
        expected = end - start + 1
        if len(data) != expected:
            raise RuntimeError(f"checkpoint range length {len(data)} != {expected}")
        return data


def _get_json(url: str) -> tuple[dict[str, object], bytes]:
    request = urllib.request.Request(url, headers={"User-Agent": "tessera-nvfp4-ingest-validation"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    return json.loads(payload), payload


def _safetensors_header(repo: str, revision: str, shard: str):
    url = _url(repo, revision, shard)
    prefix = _fetch_range(url, 0, 1_048_575)
    header_size = int.from_bytes(prefix[:8], "little")
    required = 8 + header_size
    if required > len(prefix):
        prefix = _fetch_range(url, 0, required - 1)
    return url, json.loads(prefix[8:required]), required


def _read_tensor(repo: str, revision: str, tensor: str, weight_map: dict[str, str], headers):
    shard = weight_map[tensor]
    url, header, data_start = headers[shard]
    spec = header[tensor]
    start, stop = map(int, spec["data_offsets"])
    raw = _fetch_range(url, data_start + start, data_start + stop - 1)
    dtypes = {
        "U8": np.dtype(np.uint8),
        "F8_E4M3": np.dtype(ml_dtypes.float8_e4m3fn),
        "F32": np.dtype(np.float32),
        "BF16": np.dtype(ml_dtypes.bfloat16),
    }
    try:
        dtype = dtypes[spec["dtype"]]
    except KeyError as exc:
        raise ValueError(f"unsupported source Safetensors dtype {spec['dtype']!r}") from exc
    expected = int(np.prod(spec["shape"], dtype=np.int64)) * dtype.itemsize
    if expected != len(raw):
        raise RuntimeError(f"tensor byte count mismatch for {tensor}: {len(raw)} != {expected}")
    values = np.frombuffer(raw, dtype=dtype).reshape(tuple(spec["shape"])).copy()
    return values, raw, spec, shard


def _relative_rms(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError("quality comparison requires matching shapes")
    # Bound temporary f64 storage for real merged gate/up matrices.
    ref_flat, got_flat = reference.reshape(-1), candidate.reshape(-1)
    error, signal, max_abs = 0.0, 0.0, 0.0
    for start in range(0, ref_flat.size, 1_048_576):
        ref = np.asarray(ref_flat[start:start + 1_048_576], dtype=np.float64)
        diff = ref - np.asarray(got_flat[start:start + 1_048_576], dtype=np.float64)
        error += float(np.square(diff).sum())
        signal += float(np.square(ref).sum())
        max_abs = max(max_abs, float(np.max(np.abs(diff), initial=0.0)))
    return {
        "relative_rms_error": float(np.sqrt(error / signal)) if signal else 0.0,
        "sqnr_db": float(10.0 * np.log10(signal / error)) if signal and error else float("inf"),
        "max_abs_error": max_abs,
    }


def _load_projection(tensor: str = TENSOR):
    allowed = {
        TENSOR,
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    }
    if tensor not in allowed:
        raise ValueError("checkpoint recorder supports layer-0 q_proj or gate/up only")
    started = time.perf_counter()
    q_index, q_index_bytes = _get_json(_url(NVFP4_REPO, NVFP4_REVISION, "model.safetensors.index.json"))
    bf_index, bf_index_bytes = _get_json(_url(BF16_REPO, BF16_REVISION, "model.safetensors.index.json"))
    q_map = q_index["weight_map"]
    bf_map = bf_index["weight_map"]
    prefix = tensor.removesuffix(".weight")
    q_names = (tensor, prefix + ".weight_scale", prefix + ".weight_scale_2")
    q_shards = sorted({q_map[name] for name in q_names})
    bf_shards = sorted({bf_map[tensor]})
    q_headers = {name: _safetensors_header(NVFP4_REPO, NVFP4_REVISION, name) for name in q_shards}
    bf_headers = {name: _safetensors_header(BF16_REPO, BF16_REVISION, name) for name in bf_shards}
    q_weight, q_weight_raw, q_weight_spec, q_shard = _read_tensor(NVFP4_REPO, NVFP4_REVISION, tensor, q_map, q_headers)
    q_scale, q_scale_raw, q_scale_spec, _ = _read_tensor(NVFP4_REPO, NVFP4_REVISION, prefix + ".weight_scale", q_map, q_headers)
    q_global, q_global_raw, q_global_spec, _ = _read_tensor(NVFP4_REPO, NVFP4_REVISION, prefix + ".weight_scale_2", q_map, q_headers)
    bf_weight, bf_weight_raw, bf_spec, bf_shard = _read_tensor(BF16_REPO, BF16_REVISION, tensor, bf_map, bf_headers)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if q_weight_spec["dtype"] != "U8" or q_scale_spec["dtype"] != "F8_E4M3" or q_global_spec["dtype"] != "F32":
        raise ValueError("the selected source tensors do not have the expected NVFP4 storage contract")
    if bf_spec["dtype"] != "BF16" or bf_weight.ndim != 2:
        raise ValueError("the selected BF16 source must be a rank-2 BF16 matrix")
    n, k = bf_weight.shape
    if n <= 0 or k <= 0 or k % 32 or q_weight.shape != (n, k // 2) or q_scale.shape != (n, k // 16):
        raise ValueError("NVFP4 packed weights/scales do not match the BF16 source shape")
    if q_global.size != 1:
        raise ValueError("each NVFP4 projection requires one independent global scale")
    global_scale = float(q_global.reshape(()))
    projection = ingest.NVFP4Projection(prefix.rsplit(".", 1)[-1], q_weight, q_scale, global_scale)
    return {
        "projection": projection,
        "bf16": bf_weight,
        "download_ms": elapsed_ms,
        "source": {
            "nvfp4_repo": NVFP4_REPO,
            "nvfp4_revision": NVFP4_REVISION,
            "bf16_repo": BF16_REPO,
            "bf16_revision": BF16_REVISION,
            "tensor": tensor,
            "nvfp4_shard": q_shard,
            "bf16_shard": bf_shard,
            "nvfp4_index_sha256": hashlib.sha256(q_index_bytes).hexdigest(),
            "bf16_index_sha256": hashlib.sha256(bf_index_bytes).hexdigest(),
            "nvfp4_tensor_sha256": hashlib.sha256(q_weight_raw).hexdigest(),
            "nvfp4_scale_sha256": hashlib.sha256(q_scale_raw).hexdigest(),
            "nvfp4_global_scale_sha256": hashlib.sha256(q_global_raw).hexdigest(),
            "bf16_tensor_sha256": hashlib.sha256(bf_weight_raw).hexdigest(),
            "nvfp4_storage": {"weight": q_weight_spec, "scale": q_scale_spec, "global_scale": q_global_spec},
            "bf16_storage": bf_spec,
            "nvfp4_global_scale": global_scale,
            "downloaded_tensor_bytes": sum(map(len, (q_weight_raw, q_scale_raw, q_global_raw, bf_weight_raw))),
            "download_note": "Only the selected tensor byte ranges and small indexes/headers were fetched; no model shard was saved.",
        },
    }



def _direct_bf16_to_mxfp4(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Directly quantize BF16 values into the same K32 E2M1/E8M0 contract.

    This diagnostic uses the shared bounded joint code/scale search, seeded
    from max(abs(block))/6, and processes rows in bounded batches.
    """
    n, k = values.shape
    if k % 32:
        raise ValueError("direct BF16 baseline requires K divisible by 32")
    groups = k // 32
    codes = np.empty((n, k), dtype=np.uint8)
    exponents = np.zeros((groups, n), dtype=np.uint8)
    for row_start in range(0, n, 64):
        row_end = min(n, row_start + 64)
        blocks = np.asarray(values[row_start:row_end], dtype=np.float64).reshape(row_end - row_start, groups, 32)
        maximum = np.max(np.abs(blocks), axis=-1)
        nonzero = maximum > 0.0
        seed = np.floor(np.log2(np.divide(maximum, 6.0, out=np.ones_like(maximum), where=nonzero)))
        seed = np.clip(seed, -126, 127).astype(np.int64)
        block_codes, block_exponents = ingest._requantize_e2m1_blocks(blocks, seed)
        codes[row_start:row_end] = block_codes.reshape(row_end - row_start, k)
        exponents[:, row_start:row_end] = np.where(
            nonzero, block_exponents + 127, 0
        ).T.astype(np.uint8)
    return codes, exponents, mx.exact_weights(codes, exponents)

def _native_checkpoint_ingest(projections, reference):
    """Run the native Graph ingest and certify its outputs before timing."""
    n, k = reference.shape
    codes = np.ascontiguousarray(np.concatenate([p.packed_codes for p in projections]))
    scales = np.ascontiguousarray(np.concatenate([p.e4m3_scales for p in projections]))
    globals_ = np.asarray([p.global_scale for p in projections], dtype=np.float64)
    graph = native_ingest.build_nvfp4_ingest_graph(
        n, k, reference.row_offsets, numeric_policy=ingest.nvfp4_requantization_policy())
    start = time.perf_counter()
    package = native_ingest.package_nvfp4_ingest_graph(graph)
    compile_ms = (time.perf_counter() - start) * 1000.0

    def validate(outputs):
        packed, exponents, stats = outputs
        np.testing.assert_array_equal(packed, reference.packed_codes)
        np.testing.assert_array_equal(exponents, reference.scale_exponents)
        assert stats.shape == (n, k // 32, 2)
        assert stats.dtype == np.float64
        assert np.isfinite(stats).all() and np.all(stats >= 0)
        # Independent decoded-weight reductions, bounded to 64 rows.
        for projection, first, last in zip(
                projections, reference.row_offsets[:-1], reference.row_offsets[1:], strict=True):
            for start_row in range(first, last, 64):
                stop = min(last, start_row + 64)
                local = slice(start_row - first, stop - first)
                source = ingest._E2M1[mx.unpack_e2m1_codes(
                    projection.packed_codes[local])].astype(np.float64)
                source *= (projection.e4m3_scales[local].astype(np.float64)
                           * projection.global_scale).repeat(16, axis=1)
                decoded = mx.exact_weights(
                    mx.unpack_e2m1_codes(packed[start_row:stop]),
                    exponents[:, start_row:stop]).astype(np.float64)
                expected = np.stack((
                    np.square(source).reshape(stop - start_row, k // 32, 32).sum(axis=-1),
                    np.square(source - decoded).reshape(stop - start_row, k // 32, 32).sum(axis=-1),
                ), axis=-1)
                np.testing.assert_allclose(stats[start_row:stop], expected,
                                           rtol=1e-12, atol=0)
        return packed, exponents

    start = time.perf_counter()
    validate(native_ingest.execute_nvfp4_ingest(package, codes, scales, globals_))
    checked_host_ms = (time.perf_counter() - start) * 1000.0
    samples = []
    outputs = native_ingest.execute_nvfp4_ingest(
        package, codes, scales, globals_, event_samples=samples)
    packed, exponents = validate(outputs)
    if len(samples) != 3 or not all(np.isfinite(x) and x > 0 for x in samples):
        raise RuntimeError("native ingest did not return three positive HIP event samples")
    witness = {
        "route": "GraphIR->ScheduleIR->TileIR->ROCm Target IR->LLVM->HSACO",
        "correctness": "codes/scales bitwise and independent f64 block statistics before and after timing",
        "compiler_package_ms": compile_ms,
        "checked_host_execution_ms": checked_host_ms,
        "resident_event_ms_samples": samples,
        "resident_event_ms_median": statistics.median(samples),
        "event_launches_per_sample": 10,
        "timing_domains": "Host includes launch/module/copies and independent correctness checking; HIP event covers ten resident native launches including dispatch/enqueue gaps.",
        "entry": package.native.descriptor.entry_symbol,
        "abi": package.native.descriptor.abi_id,
        "image_sha256": hashlib.sha256(package.native.image.payload).hexdigest(),
        "graph_ir_sha256": hashlib.sha256(package.graph_ir.encode()).hexdigest(),
        "schedule_ir_sha256": hashlib.sha256(package.schedule_ir.encode()).hexdigest(),
        "tile_ir_sha256": hashlib.sha256(package.native.tile_ir.encode()).hexdigest(),
        "target_ir_sha256": hashlib.sha256(package.native.target_ir.encode()).hexdigest(),
    }
    return replace(reference, packed_codes=packed, scale_exponents=exponents), witness


def measure(m: int = 16, repeats: int = 7, iterations: int = 20, *,
            projection_group: str = "q_proj") -> dict[str, object]:
    if m <= 0 or repeats <= 0 or iterations <= 0:
        raise ValueError("M, repeats and iterations must be positive")
    if projection_group not in {"q_proj", "gate_up"}:
        raise ValueError("projection_group must be q_proj or gate_up")
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("checkpoint ingest benchmark must run on Tajasaurus gfx1201")
    tensors = ([TENSOR] if projection_group == "q_proj" else [
        "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight",
    ])
    loaded = [_load_projection(tensor) for tensor in tensors]
    projections = [item["projection"] for item in loaded]
    base = np.concatenate([np.asarray(item["bf16"], dtype=np.float32) for item in loaded])
    ingest_start = time.perf_counter()
    weights = ingest.ingest_nvfp4_projections(projections)
    ingest_ms = (time.perf_counter() - ingest_start) * 1000.0
    weights, native_conversion = _native_checkpoint_ingest(projections, weights)
    source_values = []
    for source in projections:
        source_codes = mx.unpack_e2m1_codes(source.packed_codes)
        _, _, _, source_scales, _ = ingest._validate_projection(source)
        source_values.append(ingest._E2M1[source_codes] * source_scales.repeat(16, axis=1))
    source_nvfp4 = np.concatenate(source_values)
    del source_values
    mxfp4 = mx.exact_weights(mx.unpack_e2m1_codes(weights.packed_codes), weights.scale_exponents)
    direct_codes, direct_exponents, direct_mxfp4 = _direct_bf16_to_mxfp4(base)
    quality = {
        "nvfp4_dequantized_vs_bf16": _relative_rms(base, source_nvfp4),
        "mxfp4_ingested_vs_nvfp4_source": _relative_rms(source_nvfp4, mxfp4),
        "mxfp4_ingested_vs_bf16": _relative_rms(base, mxfp4),
        "mxfp4_direct_from_bf16_bounded_seed_vs_bf16": _relative_rms(base, direct_mxfp4),
        "mxfp4_ingested_vs_direct_bf16_mxfp4": _relative_rms(direct_mxfp4, mxfp4),
        "per_projection": [
            {
                "name": projection.name,
                "source_global_scale": projection.global_scale,
                "nvfp4_vs_bf16": _relative_rms(base[start:stop], source_nvfp4[start:stop]),
                "ingested_vs_nvfp4": _relative_rms(source_nvfp4[start:stop], mxfp4[start:stop]),
                "ingested_vs_bf16": _relative_rms(base[start:stop], mxfp4[start:stop]),
                "direct_mxfp4_vs_bf16": _relative_rms(base[start:stop], direct_mxfp4[start:stop]),
            }
            for projection, start, stop in zip(
                projections, weights.row_offsets[:-1], weights.row_offsets[1:], strict=True
            )
        ],
        "direct_mxfp4_method": "same bounded joint E2M1/E8M0 candidate search, seeded by floor(log2(max(abs(K32))/6)); row-batched by 64",
    }

    # The matrix package is the same checked Graph->Schedule->Tile->gfx1201
    # route as the synthetic recorder, with independently ingested real projections.
    n, k = weights.shape
    shape = (m, n, k)
    seed = 0x1201_8B
    rng = np.random.default_rng(seed)
    a_values = rng.integers(-4, 5, size=(m, k)).astype(np.float32)
    a_f8 = a_values.astype(ml_dtypes.float8_e4m3fn)
    a_scale = np.ones((m,), dtype=np.float32)
    a = np.ascontiguousarray(a_f8.view(np.uint8))
    output = np.zeros((m, n), dtype=ml_dtypes.bfloat16)
    buffers = {
        "a": a,
        "b_packed": np.ascontiguousarray(weights.packed_codes.T),
        "a_scale": a_scale,
        "b_scale": weights.scale_exponents,
        "output": output,
    }
    expected = ((a_f8.astype(np.float32) * a_scale[:, None]) @ mxfp4.T).astype(ml_dtypes.bfloat16)
    package_start = time.perf_counter()
    package, schedule_ir, tile_ir, target_ir = scheduled._compile_package(m, n, k)
    package_ms = (time.perf_counter() - package_start) * 1000.0
    resident, kernel_samples, grid, block = scheduled._device_resident_run(
        package, buffers, shape, expected=expected, repeats=repeats, iterations=iterations,
    )
    np.testing.assert_allclose(resident, expected, rtol=2e-2, atol=2e-2)

    artifact = rt.RuntimeArtifact(
        metadata={"target": package.image.target}, native_image=package.image,
        launch_descriptor=package.descriptor, tile_ir=tile_ir, target_ir=target_ir,
    )
    direct_output = np.zeros((m, n), dtype=ml_dtypes.bfloat16)
    direct_expected = ((a_f8.astype(np.float32) * a_scale[:, None]) @ direct_mxfp4.T).astype(ml_dtypes.bfloat16)
    direct_args = {
        "buffers": {
            "a": a,
            "b_packed": np.ascontiguousarray(mx.pack_e2m1_codes(direct_codes).T),
            "a_scale": a_scale,
            "b_scale": direct_exponents,
            "output": direct_output,
        },
        "scalars": dict(zip(("M", "N", "K"), shape)),
    }
    direct_result = rt.launch(artifact, direct_args)
    if direct_result.get("ok") is not True or direct_result.get("execution_kind") != "native_gpu":
        raise RuntimeError(f"direct BF16-to-MXFP4 native consumer did not execute: {direct_result}")
    np.testing.assert_allclose(direct_output, direct_expected, rtol=2e-2, atol=2e-2)
    direct_output_error = float(np.max(np.abs(np.asarray(direct_output, dtype=np.float32) - np.asarray(direct_expected, dtype=np.float32))))

    e2e_warmup = 5
    for _ in range(e2e_warmup):
        buffers["output"].fill(0)
        result = rt.launch(artifact, {"buffers": buffers, "scalars": dict(zip(("M", "N", "K"), shape))})
        if result.get("ok") is not True or result.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"native checkpoint consumer warmup did not execute: {result}")
        np.testing.assert_allclose(buffers["output"], expected, rtol=2e-2, atol=2e-2)
    e2e_samples = []
    for _ in range(7):
        buffers["output"].fill(0)
        start = time.perf_counter()
        result = rt.launch(artifact, {"buffers": buffers, "scalars": dict(zip(("M", "N", "K"), shape))})
        e2e_samples.append((time.perf_counter() - start) * 1000.0)
        if result.get("ok") is not True or result.get("execution_kind") != "native_gpu":
            raise RuntimeError(f"native checkpoint consumer did not execute: {result}")
        np.testing.assert_allclose(buffers["output"], expected, rtol=2e-2, atol=2e-2)
    root = Path(__file__).resolve().parents[2]
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True).stdout
    return {
        "schema": "tessera.rocm.nvfp4_checkpoint_ingest.v2",
        "work_item": "ROCM-NVFP4-INGEST-1",
        "sync_key": ("ROCM-NVFP4-INGEST-1-QWEN3-GATE-UP-2026-10-03"
                     if projection_group == "gate_up" else
                     "ROCM-NVFP4-INGEST-1-QWEN3-QPROJ-2026-10-02"),
        "revision": os.environ.get("TESSERA_SOURCE_REVISION", "source_snapshot"),
        "source_dirty": bool(dirty.strip()),
        "source_code_sha256": {
            "native_ingest": hashlib.sha256((root / "python/tessera/compiler/rocm_nvfp4_ingest_native.py").read_bytes()).hexdigest(),
            "ingest": hashlib.sha256((root / "python/tessera/compiler/rocm_nvfp4_ingest.py").read_bytes()).hexdigest(),
            "schedule_benchmark": hashlib.sha256((root / "benchmarks/rocm/benchmark_rocm_nvfp4_ingest_schedule.py").read_bytes()).hexdigest(),
            "checkpoint_benchmark": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "compiler_sha256": hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "device": {"host": socket.gethostname(), "target": "rocm_gfx1201", "architecture": rt._rocm_live_arch()},
        "projection_group": projection_group,
        "source": loaded[0]["source"] if projection_group == "q_proj" else {
            "projections": [item["source"] for item in loaded],
            "merge_order": list(weights.projection_names),
            "row_offsets": list(weights.row_offsets),
        },
        "ingest_numeric_policy": weights.numeric_policy(),
        "shape_mnk": list(shape),
        "source_range_fetch_ms": sum(item["download_ms"] for item in loaded),
        "host_reference_ingest_ms": ingest_ms,
        "native_ingest": native_conversion,
        "compiler_package_ms": package_ms,
        "quality": quality,
        "output_correctness": "passed_before_timing_and_after_every_end_to_end_launch",
        "direct_bf16_mxfp4_output_correctness": "passed_on_same_native_package",
        "direct_bf16_mxfp4_maximum_output_abs_error": direct_output_error,
        "maximum_output_abs_error": float(np.max(np.abs(np.asarray(resident, dtype=np.float32) - np.asarray(expected, dtype=np.float32)))),
        "package": {
            "entry": package.descriptor.entry_symbol,
            "abi": package.descriptor.abi_id,
            "image_sha256": hashlib.sha256(package.image.payload).hexdigest(),
            "schedule_ir_sha256": hashlib.sha256(schedule_ir.encode()).hexdigest(),
            "tile_ir_sha256": hashlib.sha256(tile_ir.encode()).hexdigest(),
            "target_ir_sha256": hashlib.sha256(target_ir.encode()).hexdigest(),
            "grid": list(grid),
            "workgroup": list(block),
        },
        "timing": {
            "kernel_event_ms_samples": [value / 1000.0 for value in kernel_samples],
            "kernel_event_ms_median": statistics.median(kernel_samples) / 1000.0,
            "kernel_event_repeats_per_sample": iterations,
            "end_to_end_warmup_count": e2e_warmup,
            "end_to_end_ms_samples": e2e_samples,
            "end_to_end_ms_median": statistics.median(e2e_samples),
            "end_to_end_cv_pct": statistics.pstdev(e2e_samples) / statistics.mean(e2e_samples) * 100.0,
            "domains": "HIP event covers resident native consumer launch; E2E includes runtime.launch module load, allocations, copies, validation, and dispatch. Checkpoint range download, host reference conversion and native ingest are separate.",
        },
        "limitations": [
            "One layer-0 projection group only; this does not establish whole-model quality or all ModelOpt packing variants.",
            "Native Graph ingest outputs are copied back for the separately packaged consumer; this recorder does not prove a fused resident ingest-to-consumer edge.",
            "BF16 baseline is the pinned Qwen3-8B source revision named by the NVIDIA checkpoint model lineage.",
            "Activation input is deterministic synthetic FP8; only the weight input is real checkpoint data.",
            "No selector promotion follows from this packet.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-group", choices=("q_proj", "gate_up"), default="q_proj")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    packet = json.dumps(measure(args.m, args.repeats, args.iterations, projection_group=args.projection_group), indent=2, sort_keys=True)
    if args.output is None:
        print(packet)
    else:
        args.output.write_text(packet + "\n")
        print(f"wrote {args.output}")
