"""Record cubin/SASS resources for native TF32/FP8 transformer candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_low_precision_native_resources.json"


def record() -> dict[str, object]:
    from benchmarks.nvidia.record_tile_fragment_resources import (
        _CudaOccupancy, parse_resource_usage, parse_sass_instruction_families)
    from tessera.compiler.emit.nvidia_cuda import (
        _MMA_ATTN_ENTRY, _MMA_FUSED_ENTRY, _MMA_GATED_ENTRY,
        _synthesize_mma_attn_cuda, _synthesize_mma_fused_cuda,
        _synthesize_mma_gated_cuda)

    nvcc = "/usr/local/cuda/bin/nvcc"
    cuobjdump = "/usr/local/cuda/bin/cuobjdump"
    rows: list[dict[str, object]] = []
    occupancy = _CudaOccupancy()
    try:
        with tempfile.TemporaryDirectory(prefix="tessera-lowp-resources-") as tmp:
            work = Path(tmp)
            for storage in ("f32", "fp8_e4m3", "fp8_e5m2"):
                sources = {
                    "fused": (_MMA_FUSED_ENTRY,
                              _synthesize_mma_fused_cuda(True, "gelu", storage)),
                    "attention": (_MMA_ATTN_ENTRY,
                                  _synthesize_mma_attn_cuda(storage)),
                    "gated": (_MMA_GATED_ENTRY,
                              _synthesize_mma_gated_cuda(storage, "silu")),
                }
                suffix = "tf32" if storage == "f32" else storage
                for kind, (entry, source) in sources.items():
                    src = work / f"{kind}-{suffix}.cu"
                    cubin = work / f"{kind}-{suffix}.cubin"
                    src.write_text(source)
                    subprocess.run(
                        [nvcc, "-std=c++17", "-O3", "-arch=sm_120a", "-cubin",
                         str(src), "-o", str(cubin)], check=True,
                        capture_output=True, text=True)
                    usage_text = subprocess.run(
                        [cuobjdump, "--dump-resource-usage", str(cubin)],
                        check=True, capture_output=True, text=True).stdout
                    sass_text = subprocess.run(
                        [cuobjdump, "--dump-sass", str(cubin)], check=True,
                        capture_output=True, text=True).stdout
                    resources = parse_resource_usage(usage_text)
                    sass = parse_sass_instruction_families(sass_text)
                    logical_kernel = f"{entry}_kernel"
                    matches = [name for name in resources
                               if logical_kernel in name]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"missing unique {logical_kernel} resource record; found "
                            f"{sorted(resources)}")
                    kernel = matches[0]
                    dynamic_sizes = (8192, 32768) if kind == "attention" else (0,)
                    for dynamic in dynamic_sizes:
                        row: dict[str, object] = {
                            "candidate": (f"nvidia_mma_attn_{suffix}"
                                          if kind == "attention"
                                          else f"nvidia_mma_{kind}_{suffix}"),
                            "storage_dtype": storage,
                            "kernel": kernel,
                            "dynamic_shared_memory_bytes": dynamic,
                            **resources[kernel],
                            **occupancy.row(cubin, kernel, 32, dynamic),
                            "sass_instruction_families": sass.get(kernel, []),
                        }
                        row["spill_evidence_complete"] = True
                        row["spills_detected"] = bool(
                            row["stack_bytes"] or row["local_bytes"])
                        canonical = json.dumps(
                            row, sort_keys=True, separators=(",", ":"))
                        row["resource_fingerprint"] = "sha256:" + hashlib.sha256(
                            canonical.encode()).hexdigest()
                        row["cubin_sha256"] = hashlib.sha256(
                            cubin.read_bytes()).hexdigest()
                        rows.append(row)
    finally:
        occupancy.close()
    return {
        "schema": "tessera.nvidia.lowp-native-resources.v1",
        "device": "nvidia:sm_120", "arch": "sm_120a",
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    payload = record()
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
