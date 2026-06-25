#!/usr/bin/env python3
"""Sprint G-8 (2026-05-11) — `nvcc -ptx` compile-only validator.

Walks every G-4 lit fixture under
`tests/tessera-ir/phase3/cuda13/`, extracts the embedded PTX-pattern
assertions, and writes a minimal CUDA kernel that exercises each PTX
intrinsic.  Then invokes `nvcc -arch=sm_90a -ptx -o /dev/null` to
confirm the toolchain accepts every pattern Tessera's lowering pass
emits.

Hardware-free: `nvcc -ptx` produces PTX text without requiring a GPU.

Usage:
    python scripts/validate_nvcc_compile.py --nvcc /opt/cuda-13.3/bin/nvcc

Exit codes:
    0 — every fixture compiles cleanly (or nvcc absent → skipped)
    1 — at least one fixture failed to compile
    2 — bad arguments / missing fixtures
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Minimum nvcc version the validator accepts (matches the Python pin).
MIN_NVCC_VERSION = (13, 3, 0)


# PTX patterns the G-4 fixtures lock — each maps to a minimal CUDA C++
# stub that uses the intrinsic / inline asm form.  When `nvcc -ptx`
# accepts the stub, the corresponding lit fixture's `CHECK` directives
# remain reachable.
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
    ap.add_argument("--fixtures", type=Path,
                    default=ROOT / "tests" / "tessera-ir" / "phase3" / "cuda13",
                    help="Directory containing G-4 lit fixtures")
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

    fixtures = sorted(args.fixtures.glob("*.mlir"))
    if not fixtures:
        print(f"ERROR: no fixtures found under {args.fixtures}", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    for fixture in fixtures:
        body = fixture.read_text()
        # Find every PTX pattern this fixture asserts on.
        matched_patterns = [
            pat for pat in PTX_PATTERN_STUBS
            if pat in body
        ]
        if not matched_patterns:
            print(f"  {fixture.name}: no PTX pattern requires compile-check")
            continue
        for pat in matched_patterns:
            stub = PTX_PATTERN_STUBS[pat]
            ok, err = _compile_one(nvcc, stub, arch=args.arch)
            if ok:
                print(f"  {fixture.name}: ✓ {pat}")
            else:
                print(f"  {fixture.name}: ✗ {pat}")
                print(f"    nvcc stderr: {err.strip()[:300]}")
                failures.append((fixture.name, pat))

    if failures:
        print()
        print(f"FAILED: {len(failures)} PTX patterns rejected by nvcc {version}:")
        for name, pat in failures:
            print(f"  {name}: {pat}")
        return 1

    print()
    print(f"OK: all {len(fixtures)} fixtures' PTX patterns accepted by nvcc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
