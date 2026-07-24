#!/usr/bin/env python3
"""Sprint G-8 (2026-05-11) — `nvcc -ptx` compile-only validator.

Compiles the explicit PTX instruction-probe catalog below. These probes validate
CUDA-toolchain acceptance only; emitted-IR coverage belongs to the NVIDIA MLIR
backend suite and must not be inferred from a handwritten CUDA stub.

Hardware-free: `nvcc -ptx` produces PTX text without requiring a GPU.

Usage:
    python scripts/validate_nvcc_compile.py --nvcc /opt/cuda-13.3/bin/nvcc

Exit codes:
    0 — every fixture compiles cleanly (or nvcc absent → skipped)
    1 — at least one fixture failed to compile
    2 — bad arguments
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Minimum nvcc version the validator accepts (matches the Python pin).
MIN_NVCC_VERSION = (13, 3, 0)


# Explicit CUDA-toolchain probe catalog. Each PTX pattern maps to a minimal
# CUDA C++ stub using the intrinsic / inline-asm form. This catalog is
# deliberately independent of MLIR fixtures: compiling a handwritten stub does
# not prove that Tessera emitted the instruction.
#
# This is NOT a runtime test — it's a compile-only well-formedness gate.
PTX_PATTERN_STUBS: dict[str, str] = {
    "wgmma.mma_async.sync.aligned.m64n256k16": r'''
extern "C" __global__ void k_wgmma_m64n256k16(__nv_bfloat16 *a, __nv_bfloat16 *b, float *c) {
    asm volatile(
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3}, %4, %5, p, 1, 1, 0, 0;"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "l"(a), "l"(b));
}
''',
    "wgmma.mma_async.sync.aligned.m64n128k16": r'''
extern "C" __global__ void k_wgmma_m64n128k16(__nv_bfloat16 *a, __nv_bfloat16 *b, float *c) {
    asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3}, %4, %5, p, 1, 1, 0, 0;"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "l"(a), "l"(b));
}
''',
    "wgmma.mma_async.sync.aligned.m32n32k16": r'''
extern "C" __global__ void k_wgmma_m32n32k16(__nv_bfloat16 *a, __nv_bfloat16 *b, float *c) {
    asm volatile(
      "wgmma.mma_async.sync.aligned.m32n32k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3}, %4, %5, p, 1, 1, 0, 0;"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "l"(a), "l"(b));
}
''',
    "mbarrier.arrive.expect_tx": r'''
extern "C" __global__ void k_mbarrier(uint64_t *bar) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 1024;"
                 :: "l"(bar));
}
''',
    "cp.async.bulk.tensor": r'''
extern "C" __global__ void k_cp_async_bulk_tensor(void *desc, void *smem) {
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {0, 0}], [%2];"
      :: "l"(smem), "l"(desc), "l"(smem));
}
''',
    "tcgen05.mma": r'''
extern "C" __global__ void k_tcgen05_mma(void *desc_a, void *desc_b, uint32_t tmem_d) {
    asm volatile("tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, p;"
                 :: "r"(tmem_d), "l"(desc_a), "l"(desc_b));
}
''',
    "tcgen05.alloc": r'''
extern "C" __global__ void k_tcgen05_alloc(uint32_t *out) {
    asm volatile("tcgen05.alloc.cta_group::1.b32 %0, 128;"
                 : "=r"(*out));
}
''',
    "shfl.sync.bfly": r'''
extern "C" __global__ void k_shfl_bfly(float *v) {
    float x = v[threadIdx.x];
    asm volatile("shfl.sync.bfly.b32 %0, %1, 16, 0x1f, 0xffffffff;"
                 : "=f"(x) : "f"(x));
    v[threadIdx.x] = x;
}
''',
}


def _check_nvcc_version(nvcc: Path) -> tuple[int, int, int]:
    """Return the (major, minor, patch) of `nvcc --version`."""
    try:
        out = subprocess.check_output([str(nvcc), "--version"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: failed to invoke {nvcc}: {e}", file=sys.stderr)
        return (0, 0, 0)
    # Example line: "release 13.3, V13.3.123"
    match = re.search(r"release (\d+)\.(\d+)(?:\.(\d+))?", out)
    if not match:
        return (0, 0, 0)
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3) or "0")
    return (major, minor, patch)


def _compile_one(nvcc: Path, source: str, arch: str = "sm_90a") -> tuple[bool, str]:
    """Run `nvcc -ptx --gpu-architecture=sm_90a` over `source`."""
    with tempfile.NamedTemporaryFile(
        suffix=".cu", mode="w", delete=False
    ) as tmp:
        # Need the right header for __nv_bfloat16 + uint64_t / uint32_t.
        tmp.write("#include <cuda_bf16.h>\n")
        tmp.write("#include <cstdint>\n")
        tmp.write(source)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [
                str(nvcc),
                "-ptx",
                "--gpu-architecture=" + arch,
                "-o", "/dev/null",
                tmp_path,
            ],
            capture_output=True,
            text=True,
        )
        ok = proc.returncode == 0
        return ok, proc.stderr or proc.stdout
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nvcc", type=Path, default=None,
                    help="Path to nvcc executable (auto-detected if absent)")
    ap.add_argument("--arch", default="sm_90a",
                    help="CUDA arch for compile (default sm_90a)")
    args = ap.parse_args()

    nvcc = args.nvcc or shutil.which("nvcc")
    if nvcc is None:
        print("nvcc not found on PATH and --nvcc not supplied — skipping.")
        print("This is the expected state on dev boxes without CUDA installed.")
        return 0
    nvcc = Path(nvcc)

    version = _check_nvcc_version(nvcc)
    if version < MIN_NVCC_VERSION:
        print(
            f"ERROR: nvcc version {version} < required {MIN_NVCC_VERSION}.  "
            f"Tessera pins CUDA 13.3.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: nvcc {'.'.join(map(str, version))} at {nvcc}")

    failures: list[tuple[str, str]] = []
    for pattern, stub in PTX_PATTERN_STUBS.items():
        ok, err = _compile_one(nvcc, stub, arch=args.arch)
        if ok:
            print(f"  ✓ {pattern}")
        else:
            print(f"  ✗ {pattern}")
            print(f"    nvcc stderr: {err.strip()[:300]}")
            failures.append(("toolchain_probe", pattern))

    if failures:
        print()
        print(f"FAILED: {len(failures)} PTX patterns rejected by nvcc {version}:")
        for name, pat in failures:
            print(f"  {name}: {pat}")
        return 1

    print()
    print(f"OK: all {len(PTX_PATTERN_STUBS)} PTX probes accepted by nvcc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
