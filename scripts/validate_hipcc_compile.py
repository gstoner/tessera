#!/usr/bin/env python3
"""Sprint H-8 (2026-05-11) — `hipcc -S` compile-only validator.

Walks every H-4 lit fixture under
`tests/tessera-ir/phase8/rocm_7_2/`, extracts the embedded AMDGCN
intrinsic assertions, and writes a minimal HIP kernel that exercises
each builtin.  Then invokes `hipcc -S --offload-arch=gfx942 -o /dev/null`
to confirm the toolchain accepts every intrinsic Tessera's lowering
pass emits.

Hardware-free: `hipcc -S` produces AMDGCN assembly without requiring a GPU.

Usage:
    python scripts/validate_hipcc_compile.py --hipcc /opt/rocm-7.2.3/bin/hipcc

Exit codes:
    0 — every fixture compiles cleanly (or hipcc absent → skipped)
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

# Minimum hipcc version the validator accepts (matches the Python pin).
MIN_HIP_VERSION = (7, 2, 3)


# AMDGCN intrinsic patterns the H-4 fixtures lock — each maps to a
# minimal HIP C++ stub that uses the builtin or LLVM intrinsic name.
#
# NOTE: The MFMA intrinsic names below use the underscore form
# (`__builtin_amdgcn_mfma_f32_32x32x8bf16_1k`) that the AMD HIP compiler
# accepts.  The `llvm.amdgcn.mfma.*` patterns the fixtures assert on
# come out of MLIR codegen; the C++ builtin is the surface a HIP
# program can directly invoke for compile-only validation.
AMDGCN_PATTERN_STUBS: dict[str, str] = {
    "llvm.amdgcn.mfma.f32.32x32x8bf16.1k": r'''
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef float  f32x16 __attribute__((ext_vector_type(16)));
extern "C" __global__ void k_mfma_32x32x8_bf16(const bf16x4 *a, const bf16x4 *b, f32x16 *c) {
    c[0] = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a[0], b[0], c[0], 0, 0, 0);
}
''',
    "llvm.amdgcn.mfma.f32.16x16x16bf16.1k": r'''
typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef float  f32x4 __attribute__((ext_vector_type(4)));
extern "C" __global__ void k_mfma_16x16x16_bf16(const bf16x4 *a, const bf16x4 *b, f32x4 *c) {
    c[0] = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a[0], b[0], c[0], 0, 0, 0);
}
''',
    "llvm.amdgcn.mfma.f32.32x32x16f8f8": r'''
typedef long long  i64x1  __attribute__((ext_vector_type(1)));
typedef float      f32x16 __attribute__((ext_vector_type(16)));
extern "C" __global__ void k_mfma_32x32x16_f8(const i64x1 *a, const i64x1 *b, f32x16 *c) {
    // CDNA 3 FP8 MFMA intrinsic — packed FP8 in i64 vectors.
    c[0] = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a[0][0], b[0][0], c[0], 0, 0, 0);
}
''',
    "llvm.amdgcn.mfma.f32.32x32x32f4f4": r'''
typedef long long  i64x2  __attribute__((ext_vector_type(2)));
typedef float      f32x16 __attribute__((ext_vector_type(16)));
// CDNA 4 (gfx950) FP4 MFMA — preliminary intrinsic name from ROCm 7.2.3.
extern "C" __global__ void k_mfma_32x32x32_f4(const i64x2 *a, const i64x2 *b, f32x16 *c) {
    c[0] = __builtin_amdgcn_mfma_f32_32x32x32_fp4_fp4(a[0], b[0], c[0], 0, 0, 0);
}
''',
    "llvm.amdgcn.global.load.lds": r'''
extern "C" __global__ void k_global_load_lds(int *src, __attribute__((address_space(3))) int *dst) {
    __builtin_amdgcn_global_load_lds(src, dst, 4, 0, 0);
}
''',
    "llvm.amdgcn.s.barrier": r'''
extern "C" __global__ void k_s_barrier() {
    __builtin_amdgcn_s_barrier();
}
''',
    "llvm.amdgcn.wmma.f32.16x16x16": r'''
typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef float  f32x8 __attribute__((ext_vector_type(8)));
// RDNA 3 (gfx1100) WMMA intrinsic.
extern "C" __global__ void k_wmma_16x16x16_bf16(const bf16x16 *a, const bf16x16 *b, f32x8 *c) {
    c[0] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[0], b[0], c[0]);
}
''',
    "llvm.amdgcn.buffer.load": r'''
typedef int i32x4 __attribute__((ext_vector_type(4)));
extern "C" __global__ void k_buffer_load(i32x4 *rsrc, int *out) {
    // buffer_load with a v4i32 resource descriptor — the canonical
    // AMD buffer intrinsic shape.
    *out = __builtin_amdgcn_buffer_load_i32(rsrc[0], 0, 0, false, false);
}
''',
}


def _check_hipcc_version(hipcc: Path) -> tuple[int, int, int]:
    """Return the (major, minor, patch) of `hipcc --version`."""
    try:
        out = subprocess.check_output([str(hipcc), "--version"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: failed to invoke {hipcc}: {e}", file=sys.stderr)
        return (0, 0, 0)
    # Example: "HIP version: 7.2.3.61234-abc"
    match = re.search(r"HIP version:\s*(\d+)\.(\d+)\.(\d+)", out)
    if not match:
        # Also try "AMD HIP version 7.2.3"
        match = re.search(r"AMD HIP version (\d+)\.(\d+)\.(\d+)", out)
    if not match:
        return (0, 0, 0)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _compile_one(hipcc: Path, source: str, arch: str = "gfx942") -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile(
        suffix=".hip", mode="w", delete=False
    ) as tmp:
        tmp.write("#include <hip/hip_runtime.h>\n")
        tmp.write("#include <hip/hip_bf16.h>\n")
        tmp.write(source)
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            [
                str(hipcc),
                "-S",
                "--offload-arch=" + arch,
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
    ap.add_argument("--hipcc", type=Path, default=None,
                    help="Path to hipcc executable (auto-detected if absent)")
    ap.add_argument("--fixtures", type=Path,
                    default=ROOT / "tests" / "tessera-ir" / "phase8" / "rocm_7_2",
                    help="Directory containing H-4 lit fixtures")
    ap.add_argument("--arch", default="gfx942",
                    help="ROCm offload arch (default gfx942 = MI300X)")
    args = ap.parse_args()

    hipcc = args.hipcc or shutil.which("hipcc")
    if hipcc is None:
        print("hipcc not found on PATH and --hipcc not supplied — skipping.")
        print("This is the expected state on dev boxes without ROCm installed.")
        return 0
    hipcc = Path(hipcc)

    version = _check_hipcc_version(hipcc)
    if version < MIN_HIP_VERSION:
        print(
            f"ERROR: hipcc version {version} < required {MIN_HIP_VERSION}.  "
            f"Tessera pins ROCm 7.2.3.",
            file=sys.stderr,
        )
        return 1
    print(f"OK: hipcc {'.'.join(map(str, version))} at {hipcc}")

    fixtures = sorted(args.fixtures.glob("*.mlir"))
    if not fixtures:
        print(f"ERROR: no fixtures found under {args.fixtures}", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    for fixture in fixtures:
        body = fixture.read_text()
        matched = [p for p in AMDGCN_PATTERN_STUBS if p in body]
        if not matched:
            print(f"  {fixture.name}: no AMDGCN pattern requires compile-check")
            continue
        for pat in matched:
            # CDNA 4-specific intrinsics need gfx950; RDNA 3 needs gfx1100.
            arch_for_pat = args.arch
            if "f4f4" in pat or "fp4" in pat:
                arch_for_pat = "gfx950"
            elif "wmma" in pat:
                arch_for_pat = "gfx1100"
            stub = AMDGCN_PATTERN_STUBS[pat]
            ok, err = _compile_one(hipcc, stub, arch=arch_for_pat)
            if ok:
                print(f"  {fixture.name}: ✓ {pat} ({arch_for_pat})")
            else:
                print(f"  {fixture.name}: ✗ {pat} ({arch_for_pat})")
                print(f"    hipcc stderr: {err.strip()[:300]}")
                failures.append((fixture.name, pat))

    if failures:
        print()
        print(f"FAILED: {len(failures)} AMDGCN intrinsics rejected by hipcc {version}:")
        for name, pat in failures:
            print(f"  {name}: {pat}")
        return 1

    print()
    print(f"OK: all {len(fixtures)} fixtures' AMDGCN intrinsics accepted by hipcc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
