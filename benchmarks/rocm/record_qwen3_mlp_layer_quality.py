"""Stage 1 of the quantized-MLP model-quality gate: every Qwen3-8B layer, gfx1201.

For each decoder layer, the captured teacher-forced BF16 MLP inputs run through
the public seven-member native quantized MLP (activation E4M3 quantization,
source-preserving NVFP4 expansion, exact per-block FP32 products, BF16
stores). Each layer's first call passes the staged tolerance certificate
(``assert_compiled_chain``) before anything is scored. Two FP64 references
come from ``benchmark_checkpoint_quantized_mlp.layer_reference``. Every output
must be finite; the certificate runs on each layer's first chunk only (see
``measure_layer``) and the record names which chunks carried it.

* ``bf16_weights``: the original BF16 weights, no quantization;
* ``shipped_nvfp4_weights``: the decoded NVFP4 checkpoint weights, no
  activation quantization.

Three relative-RMS errors are aggregated per layer over all captured rows:
native against each reference, and the shipped-NVFP4 reference against the
BF16 one (the format's own loss with no Tessera arithmetic involved). This is
a per-layer characterization; it is not model accuracy, and no acceptance
threshold applies to it (``model_quality_acceptance`` stays false).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

import ml_dtypes
import numpy as np

from benchmarks.rocm import capture_qwen3_all_layer_mlp_inputs as capture
from benchmarks.rocm import qwen3_local_checkpoint as local
from benchmarks.rocm.benchmark_checkpoint_quantized_mlp import decode_projection, layer_reference

SCHEMA = "tessera.quality.qwen3_mlp_layer.v1"


class _Accumulator:
    """Streaming squared-error and signal sums for one comparison."""

    def __init__(self) -> None:
        self.error = 0.0
        self.signal = 0.0
        self.max_abs = 0.0

    def add(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        if reference.shape != candidate.shape:
            raise ValueError("quality comparison requires matching shapes")
        ref = np.asarray(reference, np.float64)
        diff = ref - np.asarray(candidate, np.float64)
        if not (np.isfinite(ref).all() and np.isfinite(diff).all()):
            raise ValueError("non-finite quality comparison")
        self.error += float(np.square(diff).sum())
        self.signal += float(np.square(ref).sum())
        self.max_abs = max(self.max_abs, float(np.abs(diff).max(initial=0.0)))

    def result(self) -> dict[str, float]:
        if self.signal <= 0.0:
            raise ValueError("quality comparison has no signal")
        rel = float(np.sqrt(self.error / self.signal))
        return {"relative_rms_error": rel,
                "sqnr_db": float(10.0 * np.log10(self.signal / self.error)) if self.error else float("inf"),
                "max_abs_error": self.max_abs}


def _device_identity(rt) -> tuple[str, str]:
    if rt._rocm_live_arch() != "gfx1201":
        raise RuntimeError("stage 1 records on the owning gfx1201 device only")
    info = subprocess.check_output(["/opt/rocm/bin/rocminfo"], text=True)
    match = re.search(r"Name:\s+gfx1201\s+Uuid:\s+(GPU-\S+)\s+Marketing Name:\s+([^\n]+)", info)
    if match is None:
        raise RuntimeError("owning device identity unavailable")
    return match[1], match[2].strip()


def measure_layer(fn, rt, lib, layer: int, cache_root: Path, capture_dir: Path, metadata: dict,
                  rows: int, chunk: int, snapshots) -> dict:
    from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain

    pool = capture.load_layer(capture_dir, layer, metadata)
    if rows > pool.shape[0] or rows % chunk:
        raise ValueError(f"rows {rows} must divide into {chunk}-row chunks within {pool.shape[0]} captured rows")
    loaded = local.load_layer_mlp(cache_root, layer, bf16=snapshots[0], nvfp4=snapshots[1])
    gate, up, down = loaded["projections"]
    fixed = [np.ascontiguousarray(np.concatenate([gate.packed_codes, up.packed_codes])),
             np.ascontiguousarray(np.concatenate([gate.e4m3_scales, up.e4m3_scales])),
             np.array([gate.global_scale, up.global_scale], np.float64),
             down.packed_codes, down.e4m3_scales, np.array([down.global_scale], np.float64)]
    decoded = [decode_projection(p) for p in loaded["projections"]]
    arms = {"native_vs_bf16_weights": _Accumulator(), "native_vs_shipped_nvfp4": _Accumulator(),
            "shipped_nvfp4_vs_bf16_weights": _Accumulator()}
    certificates, started = [], time.perf_counter()
    for start in range(0, rows, chunk):
        x = np.ascontiguousarray(pool[start:start + chunk])
        roots = [x, *fixed]
        out = fn(*roots)
        artifact = fn.runtime_artifact()
        if fn.execution_kind != "native_gpu" or artifact.metadata["compiler_path"] != "canonical_rocm_quantized_mlp_program":
            raise RuntimeError("public call did not execute the seven-member native program")
        if not np.isfinite(np.asarray(out, np.float32)).all():
            raise RuntimeError(f"layer {layer} rows {start}..{start + chunk} produced non-finite output")
        if start == 0:
            # The certificate costs ~80 s per 128-row chunk (unoptimized FP64
            # einsum). Within one layer every chunk runs the same image on the
            # same weights, so the first chunk carries it; every chunk is
            # still scored against both FP64 references.
            certificates.append(
                assert_compiled_chain(fn._rocm_quantized_product_last_program.native, roots, out, lib._name))
        ref_bf16 = layer_reference(x, *loaded["bf16"])
        ref_nvfp4 = layer_reference(x, *decoded)
        arms["native_vs_bf16_weights"].add(ref_bf16, out)
        arms["native_vs_shipped_nvfp4"].add(ref_nvfp4, out)
        arms["shipped_nvfp4_vs_bf16_weights"].add(ref_bf16, ref_nvfp4)
    return {"layer": layer, "rows": rows, "chunk_rows": chunk,
            "quality": {name: acc.result() for name, acc in arms.items()},
            "certified_chunk_starts": [0], "certificate": certificates[0],
            "scored_uncertified_chunks": rows // chunk - 1,
            "checkpoint_source": loaded["source"], "elapsed_s": round(time.perf_counter() - started, 1)}


def resume_mismatches(existing: dict, identity: dict) -> list[str]:
    """Name every measurement setting a resumed record does not share."""
    changed = [key for key, value in identity.items() if existing.get(key) != value]
    changed += [f"layer {item['layer']} rows/chunk" for item in existing.get("layers", [])
                if item.get("rows") != identity["rows_per_layer"] or item.get("chunk_rows") != identity["shape_mhkn"][0]]
    return changed


def record(cache_root: Path, capture_dir: Path, output: Path, layers: list[int], rows: int, chunk: int) -> dict:
    import tessera as ts
    from tessera import runtime as rt
    from benchmarks.rocm.benchmark_public_quantized_mlp import make_mlp

    metadata = json.loads((capture_dir / "capture.json").read_text())
    uuid, name = _device_identity(rt)
    lib = rt._load_rocm_native_movement_runtime()
    if lib is None:
        raise RuntimeError("native ROCm movement runtime unavailable")
    snapshots = (local.LocalSnapshot.open(cache_root, local.BF16_REPO),
                 local.LocalSnapshot.open(cache_root, local.NVFP4_REPO))
    first = local.load_layer_mlp(cache_root, layers[0], bf16=snapshots[0], nvfp4=snapshots[1])
    gate, _, down = first["projections"]
    h, k, n = gate.packed_codes.shape[0], gate.packed_codes.shape[1] * 2, down.packed_codes.shape[0]
    fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(chunk, h, k, n))
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": SCHEMA, "architecture": "gfx1201", "device_uuid": uuid,
        "recorder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compiler_sha256": hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
        "provider_sha256": hashlib.sha256(Path(lib._name).read_bytes()).hexdigest(),
        "capture_sha256": hashlib.sha256((capture_dir / "capture.json").read_bytes()).hexdigest(),
        "shape_mhkn": [chunk, h, k, n], "rows_per_layer": rows}
    result = json.loads(output.read_text()) if output.exists() else {
        **identity, "device_name": name, "host": socket.gethostname(),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "corpus": metadata["corpus"],
        "route": "authored Graph SSA -> seven native partitions -> Schedule -> Tile -> ROCm Target -> LLVM/HSACO -> one checked multi-output HIP owner",
        "reference_policy": "FP64 products; FP32 then BF16 parent/down stores; FP32 SiLU and BF16 activation; no activation quantization",
        "model_quality_acceptance": False, "layers": []}
    changed = resume_mismatches(result, identity)
    if changed:
        raise RuntimeError(f"resumed record was measured under different settings: {', '.join(changed)}; "
                           "use a new output path")
    done = {item["layer"] for item in result["layers"]}
    for layer in layers:
        if layer in done:
            continue
        item = measure_layer(fn, rt, lib, layer, cache_root, capture_dir, metadata, rows, chunk, snapshots)
        result["layers"].append(item)
        result["layers"].sort(key=lambda entry: entry["layer"])
        output.write_text(json.dumps(result, indent=2) + "\n")
        q = item["quality"]
        print(json.dumps({"layer": layer, "elapsed_s": item["elapsed_s"],
                          **{arm: round(v["relative_rms_error"], 5) for arm, v in q.items()}}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path.home() / ".cache/huggingface/hub")
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(36)))
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=128)
    args = parser.parse_args()
    record(args.cache_root, args.capture, args.output, args.layers, args.rows, args.chunk)
