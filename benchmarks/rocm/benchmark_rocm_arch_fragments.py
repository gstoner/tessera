#!/usr/bin/env python3
"""Repeated-median compiler ratchet for ROCM-5 fragment families.

This benchmark measures the portable Tile fixture through architecture-owned
fragment materialization and exact-target object serialization.  It records
stable code-object resources separately from compiler latency.  These are
cross-assembly results, not device-performance claims; only an installed exact
GPU can provide kernel latency or measured occupancy.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TESSERA_OPT = ROOT / "build/tools/tessera-opt/tessera-opt"
MLIR_OPT = Path("/usr/lib/llvm-22/bin/mlir-opt")
FIXTURE = (
    ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/test/rocm"
    / "architecture_tile_fragment_store.mlir"
)


@dataclass(frozen=True)
class Case:
    arch: str
    family: str
    dtype: str
    serialize: bool = True


CASES = (
    Case("gfx1100", "rdna3_wmma", "f16"),
    Case("gfx1151", "rdna3_wmma", "f16"),
    Case("gfx1200", "rdna4_wmma", "f16"),
    Case("gfx1201", "rdna4_wmma", "f16"),
    Case("gfx1201", "rdna4_wmma", "bf16"),
    Case("gfx1201", "rdna4_wmma", "e4m3"),
    Case("gfx1201", "rdna4_wmma", "e5m2"),
    Case("gfx1201", "rdna4_wmma", "int8"),
    Case("gfx1201", "rdna4_wmma", "int4"),
    Case("gfx1250", "gfx125x_wmma_v2", "f16"),
    Case("gfx1250", "gfx125x_wmma_v2", "bf16"),
    Case("gfx1251", "gfx125x_wmma_v2", "f16"),
    Case("gfx1251", "gfx125x_wmma_v2", "bf16"),
    Case("gfx90a", "cdna2_mfma", "f16"),
    Case("gfx90a", "cdna2_mfma", "bf16"),
    # The installed Debian LLVM 22 serializer does not recognize gfx940.
    # Lowering is still measured; gfx942 serializes the same CDNA3 family.
    Case("gfx940", "cdna3_mfma", "f16", serialize=False),
    Case("gfx942", "cdna3_mfma", "f16"),
    Case("gfx942", "cdna3_mfma", "bf16"),
    Case("gfx950", "cdna4_mfma", "f16"),
    Case("gfx950", "cdna4_mfma", "bf16"),
)


def _source(case: Case) -> str:
    source = FIXTURE.read_text()
    if case.dtype == "bf16":
        source = source.replace("f16", "bf16")
    elif case.dtype in ("e4m3", "e5m2"):
        mlir_type = "f8E4M3FN" if case.dtype == "e4m3" else "f8E5M2"
        source = source.replace("memref<256xf16>", f"memref<256x{mlir_type}>")
        source = source.replace(
            'a = "f16", b = "f16"',
            f'a = "{case.dtype}", b = "{case.dtype}"',
        )
    elif case.dtype in ("int8", "int4"):
        source = source.replace("memref<256xf16>", "memref<256xi8>")
        source = source.replace("memref<256xf32>", "memref<256xi32>")
        source = source.replace(
            'a = "f16", b = "f16", acc = "f32"',
            f'a = "{case.dtype}", b = "{case.dtype}", acc = "i32"',
        )
    if case.family == "gfx125x_wmma_v2" or case.dtype == "int4":
        input_type = {
            "f16": "f16", "bf16": "bf16", "int4": "i8",
        }[case.dtype]
        source = source.replace(
            f"memref<256x{input_type}>", f"memref<512x{input_type}>")
        source = source.replace("k = 16", "k = 32")
        old_layout = "shard = [16, 16] : [16, 1]"
        source = source.replace(
            old_layout, "shard = [16, 32] : [32, 1]", 1)
        source = source.replace(
            old_layout, "shard = [32, 16] : [16, 1]", 1)
        source = source.replace("leading_dim = 16", "leading_dim = 32", 2)
    return source


def _run(command: list[str], source: str) -> tuple[str, float]:
    start = time.perf_counter_ns()
    result = subprocess.run(command, input=source, capture_output=True, text=True)
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout, elapsed_ms


def _lower(case: Case, source: str) -> tuple[str, float]:
    return _run([
        str(TESSERA_OPT), "-", "--allow-unregistered-dialect",
        "--pass-pipeline=builtin.module("
        f"lower-tile-to-rocm{{arch={case.arch}}},"
        "lower-tessera-target-to-rocdl)",
    ], source)


def _serialize(case: Case, lowered: str) -> tuple[str, float]:
    pipeline = (
        "builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts),"
        f"rocdl-attach-target{{chip={case.arch}}},gpu-module-to-binary)"
    )
    return _run([str(MLIR_OPT), f"--pass-pipeline={pipeline}"], lowered)


def _median_mad(samples: list[float]) -> tuple[float, float]:
    median = statistics.median(samples)
    return median, statistics.median(abs(sample - median) for sample in samples)


def _resources(binary: str) -> dict[str, int | None]:
    fields = (
        "vgpr_count", "sgpr_count", "agpr_count",
        "group_segment_fixed_size", "private_segment_fixed_size",
        "vgpr_spill_count", "sgpr_spill_count", "wavefront_size",
        "max_flat_workgroup_size",
    )
    result: dict[str, int | None] = {}
    for field in fields:
        match = re.search(fr"{field} = (\d+) : i64", binary)
        result[field] = int(match.group(1)) if match else None
    # The AMD metadata encoder uses UINT32_MAX for "AGPR count unavailable"
    # on architectures without a separately reported accumulator register file.
    if result["agpr_count"] == 0xFFFFFFFF:
        result["agpr_count"] = None
    result["lds_bytes"] = result.pop("group_segment_fixed_size")
    result["scratch_bytes"] = result.pop("private_segment_fixed_size")
    return result


def measure(case: Case, repetitions: int) -> dict[str, Any]:
    source = _source(case)
    lower_samples: list[float] = []
    lowered = ""
    for _ in range(repetitions):
        lowered, elapsed = _lower(case, source)
        lower_samples.append(elapsed)
    lower_median, lower_mad = _median_mad(lower_samples)
    row: dict[str, Any] = {
        "arch": case.arch,
        "family": case.family,
        "dtype": case.dtype,
        "lower_median_ms": round(lower_median, 4),
        "lower_mad_ms": round(lower_mad, 4),
        "real_intrinsic": "rocdl.wmma." in lowered or "rocdl.mfma." in lowered,
        "evidence": "compiler_lowering",
        "device_latency_ms": None,
        "measured_occupancy": None,
    }
    if not case.serialize:
        row["serialization_status"] = "toolchain_gated_gfx940"
        return row
    serialize_samples: list[float] = []
    binary = ""
    for _ in range(repetitions):
        binary, elapsed = _serialize(case, lowered)
        serialize_samples.append(elapsed)
    serialize_median, serialize_mad = _median_mad(serialize_samples)
    row.update({
        "serialize_median_ms": round(serialize_median, 4),
        "serialize_mad_ms": round(serialize_mad, 4),
        "serialization_status": "assembled",
        "resources": _resources(binary),
        "evidence": "exact_target_cross_assembly",
    })
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("--repetitions must be at least 3 for median/MAD evidence")
    if not TESSERA_OPT.is_file() or not MLIR_OPT.is_file():
        raise SystemExit("build tessera-opt and install MLIR 22 first")
    payload = {
        "schema": "tessera.rocm.arch-fragment-benchmark.v1",
        "repetitions": args.repetitions,
        "fixture": str(FIXTURE.relative_to(ROOT)),
        "scope": "compiler/cross-assembly; no remote-device performance claims",
        "rows": [measure(case, args.repetitions) for case in CASES],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output:
        args.output.write_text(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
