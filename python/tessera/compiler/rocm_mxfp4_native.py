"""Exact gfx1201 MXFP4 W4A8 native package.

This correctness-first kernel is the executable specification for the later
WMMA schedule.  It keeps the OCP K=32 scale boundary explicit on device and
therefore never uses the lossy row-reference fold.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .native_artifact import (
    BufferBinding,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    OrderingSemantics,
    ScalarArgument,
    ShapeGuard,
)
from .rocm_native import (
    ROCMNativePackage,
    _driver_selected_device_libraries,
    _rocm_clang,
    _rocm_path,
    _version_fingerprint,
)
from .rocm_mxfp4 import (
    MXFP4_AITER_SHUFFLED_LAYOUT_V1,
    MXFP4_CHECKPOINT_LAYOUT_V1,
    MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
    MXFP4_TRANSPOSED_LAYOUT_V1,
    mxfp4_weight_layout,
)


GFX_MXFP4_W4A8_EXACT_ABI = (
    "tessera.rocm.mxfp4_w4a8.a_b_sa_sb_o_m_n_k."
    "e4m3_e2m1_e8m0_bf16.exact.v1"
)
GFX_MXFP4_W4A8_WMMA_ABI = (
    "tessera.rocm.mxfp4_w4a8.a_b_sa_sb_o_m_n_k."
    "e4m3_e2m1_e8m0_bf16.wmma_exact.v1"
)
GFX_MXFP4_W4A8_WMMA_FRAGMENT_ABI = (
    "tessera.rocm.mxfp4_w4a8.a_b_sa_sb_o_m_n_k."
    "e4m3_e2m1_e8m0_bf16.wmma_exact.fragment.v1"
)
_GFX_MXFP4_PHYSICAL_CONTRACT = "rocm_mxfp4_w4a8_exact_v1"
_GFX_MXFP4_POINTER_ABI = "a_b_lhs_scale_rhs_scale_d_m_n_k"


@dataclass(frozen=True)
class MXFP4Schedule:
    """Concrete gfx1201 schedule axes consumed by native code generation."""

    workload: str
    group_m: int = 1
    split_k: int = 1
    stages: int = 1
    waves_per_eu: int = 0
    cache_modifier: str = "default"
    k_step_schedule: str = "isolated_scale_group"

    def __post_init__(self) -> None:
        if self.workload not in {"decode", "prefill"}:
            raise ValueError("MXFP4 workload must be 'decode' or 'prefill'")
        if self.group_m not in {1, 2, 4, 8}:
            raise ValueError("MXFP4 group_m must be 1, 2, 4, or 8")
        if self.split_k not in {1, 2, 4, 8}:
            raise ValueError("MXFP4 split_k must be 1, 2, 4, or 8")
        if self.stages != 1 or self.waves_per_eu != 0:
            raise ValueError(
                "unimplemented MXFP4 schedule axes must remain at their fail-closed defaults"
            )
        if self.cache_modifier != "default":
            raise ValueError("MXFP4 cache_modifier is not implemented")
        if self.k_step_schedule not in {"isolated_scale_group", "relaxed"}:
            raise ValueError(
                "MXFP4 k_step_schedule must be 'isolated_scale_group' or 'relaxed'"
            )
        if self.workload == "decode" and self.group_m != 1:
            raise ValueError("decode does not admit prefill group-M staging")
        if self.workload == "prefill" and self.split_k != 1:
            raise ValueError("prefill does not admit decode split-K reduction")


@dataclass(frozen=True)
class MXFP4RouteReceipt:
    """Inspectable production selection or refusal for one static shape."""

    accepted: bool
    workload: str
    requested_layout: str | None
    selected_layout: str | None
    abi_id: str | None
    reason: str
    schedule: MXFP4Schedule

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "workload": self.workload,
            "requested_layout": self.requested_layout,
            "selected_layout": self.selected_layout,
            "abi_id": self.abi_id,
            "reason": self.reason,
            "schedule": {
                "group_m": self.schedule.group_m,
                "split_k": self.schedule.split_k,
                "stages": self.schedule.stages,
                "waves_per_eu": self.schedule.waves_per_eu,
                "cache_modifier": self.schedule.cache_modifier,
                "k_step_schedule": self.schedule.k_step_schedule,
            },
        }


def select_mxfp4_schedule(m: int, n: int, k: int) -> MXFP4Schedule:
    """Select only schedules whose generated implementation is present."""

    if min(m, n, k) <= 0 or k % 32:
        raise ValueError("MXFP4 W4A8 requires positive M/N and K divisible by 32")
    if m <= 16:
        return MXFP4Schedule("decode", split_k=8)
    if m <= 64:
        return MXFP4Schedule("decode", split_k=4)
    group_m = 8 if m >= 128 else 4
    return MXFP4Schedule("prefill", group_m=group_m)


def select_mxfp4_route(
    m: int,
    n: int,
    k: int,
    *,
    requested_layout: str | None = None,
    schedule: MXFP4Schedule | None = None,
) -> MXFP4RouteReceipt:
    """Explain the layout/ABI decision without compiling or allocating."""
    selected_schedule = schedule or select_mxfp4_schedule(m, n, k)
    if requested_layout is not None:
        mxfp4_weight_layout(requested_layout)
    if requested_layout in {
        MXFP4_CHECKPOINT_LAYOUT_V1,
        MXFP4_AITER_SHUFFLED_LAYOUT_V1,
    }:
        return MXFP4RouteReceipt(
            False,
            selected_schedule.workload,
            requested_layout,
            None,
            None,
            (
                "checkpoint layout is a load-time source, not a launch ABI"
                if requested_layout == MXFP4_CHECKPOINT_LAYOUT_V1
                else "AITER shuffled layout is incompatible and has no proved converter"
            ),
            selected_schedule,
        )
    selected = requested_layout
    if selected is None:
        selected = (
            MXFP4_GFX12_FRAGMENT_LAYOUT_V1
            if selected_schedule.workload == "decode" and n % 16 == 0
            else MXFP4_TRANSPOSED_LAYOUT_V1
        )
    if selected == MXFP4_GFX12_FRAGMENT_LAYOUT_V1 and n % 16:
        return MXFP4RouteReceipt(
            False,
            selected_schedule.workload,
            requested_layout,
            None,
            None,
            "gfx12 fragment layout requires N divisible by 16",
            selected_schedule,
        )
    if selected not in {
        MXFP4_TRANSPOSED_LAYOUT_V1,
        MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
    }:
        return MXFP4RouteReceipt(
            False,
            selected_schedule.workload,
            requested_layout,
            None,
            None,
            f"layout {selected!r} is not an executable gfx1201 MXFP4 ABI",
            selected_schedule,
        )
    fragment = selected == MXFP4_GFX12_FRAGMENT_LAYOUT_V1
    return MXFP4RouteReceipt(
        True,
        selected_schedule.workload,
        requested_layout,
        selected,
        GFX_MXFP4_W4A8_WMMA_FRAGMENT_ABI if fragment else GFX_MXFP4_W4A8_WMMA_ABI,
        (
            "decode selects preconverted contiguous lane words"
            if fragment and selected_schedule.workload == "decode"
            else "explicit fragment prefill uses private lane-word loads pending multistage staging"
            if fragment
            else "prefill retains proved transposed LDS staging"
        ),
        selected_schedule,
    )


def _target_string_attr(operation: str, name: str) -> str:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*\"([^\"]*)\"", operation)
    if match is None:
        raise ValueError(f"gfx1201 scaled WMMA Target IR is missing {name}")
    return match.group(1)


def _target_integer_attr(operation: str, name: str) -> int:
    match = re.search(
        rf"\b{re.escape(name)}\s*=\s*(-?\d+)(?:\s*:\s*i64)?", operation
    )
    if match is None:
        raise ValueError(f"gfx1201 scaled WMMA Target IR is missing {name}")
    return int(match.group(1))


def _exact_scaled_wmma_directive(target_ir: str) -> str:
    directives = [
        line.strip()
        for line in target_ir.splitlines()
        if "tessera_rocm.scaled_wmma_gemm" in line
        and GFX_MXFP4_W4A8_WMMA_ABI in line
    ]
    if len(directives) != 1:
        raise ValueError(
            "gfx1201 native packaging requires exactly one exact packed "
            "tessera_rocm.scaled_wmma_gemm directive"
        )
    return directives[0]


def _exact_scaled_wmma_tile_carrier(tile_ir: str) -> str:
    carriers = [
        line.strip()
        for line in tile_ir.splitlines()
        if "tile.scaled_matmul_kernel" in line
        and f'physical_contract = "{_GFX_MXFP4_PHYSICAL_CONTRACT}"' in line
    ]
    if len(carriers) != 1:
        raise ValueError(
            "gfx1201 native packaging requires exactly one exact packed "
            "tile.scaled_matmul_kernel carrier"
        )
    return carriers[0]


def _schedule_hash(operation: str, *, carrier: str) -> str:
    match = re.search(
        r"\btessera\.schedule_hash\s*=\s*\"([^\"]+)\"", operation
    )
    if match is None:
        raise ValueError(f"gfx1201 scaled WMMA {carrier} is missing tessera.schedule_hash")
    return match.group(1)


def package_scaled_wmma_target_ir(
    tile_ir: str,
    target_ir: str,
    *,
    pipeline_name: str = "tessera-lower-to-rocm",
) -> ROCMNativePackage:
    """Materialize the exact packed scaled-WMMA Target IR as gfx1201 HSACO.

    This is the strict bridge between the generic Graph/Schedule/Tile pipeline
    and the separately proved RDNA4 generator. Logical block-scaled directives,
    approximate folding policies, and partially described contracts remain
    fail-closed instead of silently selecting a different physical ABI.
    """

    operation = _exact_scaled_wmma_directive(target_ir)
    strings = {
        name: _target_string_attr(operation, name)
        for name in (
            "abi",
            "scale_format",
            "partial_combine",
            "physical_contract",
            "output",
            "package_abi",
            "k_step_schedule",
        )
    }
    expected_strings = {
        "abi": _GFX_MXFP4_POINTER_ABI,
        "scale_format": "e8m0",
        "partial_combine": "scale_outer_product_then_add",
        "physical_contract": _GFX_MXFP4_PHYSICAL_CONTRACT,
        "output": "bf16",
        "package_abi": GFX_MXFP4_W4A8_WMMA_ABI,
        "k_step_schedule": "isolated_scale_group",
    }
    for name, expected_string in expected_strings.items():
        if strings[name] != expected_string:
            raise ValueError(
                f"gfx1201 scaled WMMA Target IR requires {name}={expected_string!r}"
            )

    integers = {
        name: _target_integer_attr(operation, name)
        for name in ("m", "n", "k", "instruction_k", "scale_k", "macro_k")
    }
    if min(integers["m"], integers["n"], integers["k"]) <= 0:
        raise ValueError("gfx1201 scaled WMMA Target IR requires positive static M/N/K")
    if integers["k"] % 32:
        raise ValueError("gfx1201 scaled WMMA Target IR requires K divisible by 32")
    for name, expected_integer in (
        ("instruction_k", 16), ("scale_k", 32), ("macro_k", 32)
    ):
        if integers[name] != expected_integer:
            raise ValueError(
                f"gfx1201 scaled WMMA Target IR requires {name}={expected_integer}"
            )

    policy_match = re.search(r"\bnumeric_policy\s*=\s*\{([^}]*)\}", operation)
    if policy_match is None:
        raise ValueError("gfx1201 scaled WMMA Target IR is missing numeric_policy")
    policy = policy_match.group(1)
    expected_policy = {
        "accum": "f32",
        "execution_mode": "exact_per_block",
        "storage": "e4m3_raw_u8",
    }
    for name, expected in expected_policy.items():
        if _target_string_attr(policy, name) != expected:
            raise ValueError(
                f"gfx1201 scaled WMMA Target IR requires numeric_policy.{name}="
                f"{expected!r}"
            )

    tile_operation = _exact_scaled_wmma_tile_carrier(tile_ir)
    tile_schedule_hash = _schedule_hash(tile_operation, carrier="Tile IR carrier")
    target_schedule_hash = _schedule_hash(operation, carrier="Target IR directive")
    if tile_schedule_hash != target_schedule_hash:
        raise ValueError(
            "gfx1201 scaled WMMA Tile/Target tessera.schedule_hash mismatch"
        )

    package = package_mxfp4_w4a8_wmma(
        integers["m"], integers["n"], integers["k"],
        pipeline_name=pipeline_name,
        # The current Target directive names the legacy WMMA ABI and its
        # [K/2,N] payload. Never let shape-based production selection silently
        # reinterpret that buffer as the distinct fragment-order ABI.
        weight_layout=MXFP4_TRANSPOSED_LAYOUT_V1,
    )
    target_ir_sha256 = hashlib.sha256(target_ir.encode()).hexdigest()
    image = replace(package.image, target_ir_digest=target_ir_sha256)
    provenance = {
        **package.descriptor.provenance,
        "materializer": "tessera_rocm.scaled_wmma_gemm",
        "physical_contract": _GFX_MXFP4_PHYSICAL_CONTRACT,
        "tile_ir_sha256": hashlib.sha256(tile_ir.encode()).hexdigest(),
        "target_ir_sha256": target_ir_sha256,
    }
    provenance["schedule_hash"] = target_schedule_hash
    descriptor = replace(
        package.descriptor,
        image_digest=image.image_digest,
        provenance=provenance,
    )
    return ROCMNativePackage(
        tile_ir=tile_ir,
        target_ir=target_ir,
        backend_ir=package.target_ir,
        image=image,
        descriptor=descriptor,
    )


def _rocm_hipcc(rocm_path: Path) -> Path | None:
    """Return the HIP driver that owns ``--genco`` code-object emission.

    ``amdclang++`` is still used by :func:`_driver_selected_device_libraries`
    to fingerprint the exact OCML/OCKL/OCLC selection, but it is not a drop-in
    replacement for the HIP wrapper on split ROCm installations: the raw
    driver on Tajasarus deliberately rejects the wrapper-only ``--genco``
    option.
    """

    configured = os.environ.get("TESSERA_ROCM_HIPCC")
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_file() else None
    candidates = [rocm_path / "bin" / "hipcc"]
    if rocm_path.name == "core":
        candidates.append(rocm_path.parent / "bin" / "hipcc")
    candidates.append(Path("/opt/rocm/bin/hipcc"))
    found = shutil.which("hipcc")
    if found:
        candidates.append(Path(found))
    return next((path for path in candidates if path.is_file()), None)


def _rocm_offload_bundler(rocm_path: Path) -> Path | None:
    candidates = [
        rocm_path / "lib" / "llvm" / "bin" / "clang-offload-bundler",
        rocm_path / "llvm" / "bin" / "clang-offload-bundler",
        rocm_path / "bin" / "clang-offload-bundler",
    ]
    found = shutil.which("clang-offload-bundler")
    if found:
        candidates.append(Path(found))
    return next((path for path in candidates if path.is_file()), None)


def _extract_gfx1201_hsaco(compiled: Path, output: Path, rocm_path: Path) -> bytes:
    """Return a raw HSACO from raw or LLVM 23 bundled HIP output."""

    payload = compiled.read_bytes()
    if payload.startswith(b"\x7fELF"):
        return payload
    if not payload.startswith(b"__CLANG_OFFLOAD_BUNDLE__"):
        raise RuntimeError("MXFP4 W4A8 compiler output is neither ELF nor a HIP bundle")
    bundler = _rocm_offload_bundler(rocm_path)
    if bundler is None:
        raise RuntimeError("MXFP4 W4A8 HIP bundle requires clang-offload-bundler")
    result = subprocess.run(
        [
            str(bundler), "-unbundle", "-type=o",
            "-targets=hipv4-amdgcn-amd-amdhsa--gfx1201",
            f"-input={compiled}", f"-output={output}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode or not output.is_file():
        detail = result.stderr.strip() or f"clang-offload-bundler exited {result.returncode}"
        raise RuntimeError(f"MXFP4 W4A8 HSACO extraction failed: {detail}")
    payload = output.read_bytes()
    if not payload.startswith(b"\x7fELF"):
        raise RuntimeError("MXFP4 W4A8 extracted device image is not an ELF HSACO")
    return payload


def emit_mxfp4_w4a8_exact_hip(*, entry: str = "tessera_mxfp4_w4a8_exact") -> str:
    """Emit the independent scalar baseline for the exact per-group route."""

    if not entry or not entry.replace("_", "a").isalnum():
        raise ValueError("MXFP4 entry must be a C identifier")
    return f'''#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__ float tessera_e2m1(uint8_t code) {{
  const float table[8] = {{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f}};
  float value = table[code & 7u];
  return (code & 8u) ? -value : value;
}}

__device__ __forceinline__ float tessera_e4m3fn(uint8_t code) {{
  const unsigned sign = code >> 7;
  const unsigned exponent = (code >> 3) & 15u;
  const unsigned mantissa = code & 7u;
  float value;
  if (exponent == 0u)
    value = ldexpf((float)mantissa, -9);
  else if (exponent == 15u && mantissa == 7u)
    value = __uint_as_float(0x7fc00000u);
  else
    value = ldexpf(1.0f + (float)mantissa * 0.125f,
                   (int)exponent - 7);
  return sign ? -value : value;
}}

__device__ __forceinline__ uint16_t tessera_bf16_rne(float value) {{
  uint32_t bits = __float_as_uint(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return (uint16_t)(bits >> 16);
}}

extern "C" __global__ void {entry}(
    const uint8_t *__restrict__ a_e4m3,
    const uint8_t *__restrict__ b_e2m1_packed,
    const float *__restrict__ a_scale,
    const uint8_t *__restrict__ b_scale_e8m0,
    uint16_t *__restrict__ output_bf16,
    int64_t M, int64_t N, int64_t K) {{
  const int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t m = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;

  float accum = 0.0f;
  const float row_scale = a_scale[m];
  for (int64_t group = 0; group < K / 32; ++group) {{
    float partial = 0.0f;
    const int64_t k0 = group * 32;
    for (int64_t offset = 0; offset < 32; ++offset) {{
      const int64_t k = k0 + offset;
      const uint8_t packed = b_e2m1_packed[(k >> 1) * N + n];
      const uint8_t weight = (k & 1) ? (packed >> 4) : (packed & 15u);
      partial = fmaf(tessera_e4m3fn(a_e4m3[m * K + k]),
                     tessera_e2m1(weight), partial);
    }}
    const uint8_t exponent = b_scale_e8m0[group * N + n];
    const float scale = exponent == 0u
        ? 0.0f : ldexpf(row_scale, (int)exponent - 127);
    accum = fmaf(partial, scale, accum);
  }}
  output_bf16[m * N + n] = tessera_bf16_rne(accum);
}}
'''


def emit_mxfp4_w4a8_wmma_llvmir(
    *,
    entry: str = "tessera_mxfp4_w4a8_wmma",
    schedule: MXFP4Schedule | None = None,
    weight_layout: str = MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
) -> str:
    """Emit the exact K32-scaled W4A8 route using two RDNA4 FP8 WMMAs.

    One wave owns one 16x16 output tile.  Each K=32 scale group is isolated in
    a zeroed FP32 fragment, evaluated as two K=16 WMMAs, multiplied by the
    activation/weight scale outer product, and only then joined to the running
    accumulator.  Thus the optimized route preserves the scalar baseline's
    accumulation boundary rather than approximating it with a final epilogue.
    """

    if not entry or not entry.replace("_", "a").isalnum():
        raise ValueError("MXFP4 entry must be a C identifier")
    schedule = schedule or MXFP4Schedule("decode")
    if weight_layout not in {
        MXFP4_TRANSPOSED_LAYOUT_V1,
        MXFP4_GFX12_FRAGMENT_LAYOUT_V1,
    }:
        mxfp4_weight_layout(weight_layout)
        raise ValueError(f"MXFP4 WMMA cannot consume weight layout {weight_layout!r}")
    fragment_weights = weight_layout == MXFP4_GFX12_FRAGMENT_LAYOUT_V1
    group_m = schedule.group_m
    split_k = schedule.split_k
    grouped_waves = group_m > 1
    # Fragment order already reduces each K16 slab to one contiguous lane
    # word. Keep those loads private to each wave until the separately tracked
    # multistage/padded prefill staging protocol is implemented and proved.
    shared_b = grouped_waves and not fragment_weights
    split_reduce = split_k > 1
    waves_per_workgroup = max(group_m, split_k)

    lines = [
        '; Exact gfx1201 MXFP4 W4A8: packed E2M1 -> E4M3, two WMMA per K32 group.',
        'target triple = "amdgcn-amd-amdhsa"',
        '',
        '@tessera_e2m1_to_e4m3 = private unnamed_addr addrspace(4) constant '
        '[8 x i8] [i8 0, i8 48, i8 56, i8 60, i8 64, i8 68, i8 72, i8 76], align 8',
    ]
    if shared_b:
        lines.append(
            '@tessera_mxfp4_b_lds = internal addrspace(3) global '
            '[512 x i8] undef, align 16'
        )
    if split_reduce:
        lines.append(
            '@tessera_mxfp4_partial_lds = internal addrspace(3) global '
            f'[{split_k * 1024} x i8] undef, align 16'
        )
    lines += [
        '',
        f'define protected amdgpu_kernel void @{entry}(',
        '    ptr addrspace(1) %a, ptr addrspace(1) %b,',
        '    ptr addrspace(1) %a_scale, ptr addrspace(1) %b_scale,',
        '    ptr addrspace(1) %out, i64 %M, i64 %N, i64 %K) #0 {',
        'entry:',
        '  %tid = call i32 @llvm.amdgcn.workitem.id.x()',
        '  %lane32 = and i32 %tid, 31',
        '  %wave32 = lshr i32 %tid, 5',
        '  %lane16_32 = and i32 %lane32, 15',
        '  %half32 = lshr i32 %lane32, 4',
        '  %lane16 = zext i32 %lane16_32 to i64',
        '  %lane32_64 = zext i32 %lane32 to i64',
        '  %half = zext i32 %half32 to i64',
        '  %bidx = call i32 @llvm.amdgcn.workgroup.id.x()',
        '  %bidy = call i32 @llvm.amdgcn.workgroup.id.y()',
        '  %bidx64 = zext i32 %bidx to i64',
        '  %bidy64 = zext i32 %bidy to i64',
        '  %tile_n = mul i64 %bidx64, 16',
        f'  %tile_group_m = mul i64 %bidy64, {16 * group_m}',
        '  %wave64 = zext i32 %wave32 to i64',
        f'  %wave_m = mul i64 %wave64, {16 if grouped_waves else 0}',
        '  %tile_m = add i64 %tile_group_m, %wave_m',
        '  %col = add i64 %tile_n, %lane16',
        '  %col_in = icmp ult i64 %col, %N',
        '  %col_safe = select i1 %col_in, i64 %col, i64 0',
        '  %a_row = add i64 %tile_m, %lane16',
        '  %a_row_in = icmp ult i64 %a_row, %M',
        '  %a_row_safe = select i1 %a_row_in, i64 %a_row, i64 0',
        '  %half8 = mul i64 %half, 8',
        '  %out_row_base = add i64 %tile_m, %half8',
    ]

    scale_vec = 'poison'
    for j in range(8):
        lines += [
            f'  %row_{j} = add i64 %out_row_base, {j}',
            f'  %row_in_{j} = icmp ult i64 %row_{j}, %M',
            f'  %row_safe_{j} = select i1 %row_in_{j}, i64 %row_{j}, i64 0',
            f'  %as_ptr_{j} = getelementptr float, ptr addrspace(1) %a_scale, i64 %row_safe_{j}',
            f'  %as_load_{j} = load float, ptr addrspace(1) %as_ptr_{j}, align 4',
            f'  %as_{j} = select i1 %row_in_{j}, float %as_load_{j}, float 0.000000e+00',
            f'  %as_vec_{j} = insertelement <8 x float> {scale_vec}, float %as_{j}, i64 {j}',
        ]
        scale_vec = f'%as_vec_{j}'
    loop_backedge = 'b.wait' if shared_b else 'group.body'
    first_group = '%wave64' if split_reduce else '0'
    store_target = 'split.store' if split_reduce else 'store.header'
    lines += [
        '  %groups = udiv i64 %K, 32',
        '  %k_steps = udiv i64 %K, 16',
        '  br label %group.header',
        '',
        'group.header:',
        f'  %group = phi i64 [ {first_group}, %entry ], [ %group_next, %{loop_backedge} ]',
        f'  %running = phi <8 x float> [ zeroinitializer, %entry ], [ %running_next, %{loop_backedge} ]',
        '  %group_done = icmp uge i64 %group, %groups',
        f'  br i1 %group_done, label %{store_target}, label %group.body',
        '',
        'group.body:',
        '  %kbase = mul i64 %group, 32',
    ]

    def build_fragment(kind: str, slab: int) -> tuple[list[str], list[str], str]:
        load_lines: list[str] = []
        pack_lines: list[str] = []
        words: list[str] = []
        fragment_word: str | None = None
        if kind == 'b' and fragment_weights:
            prefix = f'b_s{slab}_fragment'
            load_lines.extend([
                f'  %{prefix}_group2 = mul i64 %group, 2',
                f'  %{prefix}_kstep = add i64 %{prefix}_group2, {slab}',
                f'  %{prefix}_ntile = mul i64 %bidx64, %k_steps',
                f'  %{prefix}_tile_step = add i64 %{prefix}_ntile, %{prefix}_kstep',
                f'  %{prefix}_slot_base = mul i64 %{prefix}_tile_step, 32',
                f'  %{prefix}_slot = add i64 %{prefix}_slot_base, %lane32_64',
                f'  %{prefix}_byte = mul i64 %{prefix}_slot, 4',
                f'  %{prefix}_ptr = getelementptr i8, ptr addrspace(1) %b, i64 %{prefix}_byte',
                f'  %{prefix}_word = load i32, ptr addrspace(1) %{prefix}_ptr, align 4',
            ])
            fragment_word = f'%{prefix}_word'
        for word in range(2):
            packed: str | None = None
            for byte in range(4):
                h = word * 4 + byte
                k_off = slab * 16 + h
                tag = f'{kind}_s{slab}_{h}'
                load_lines.extend([
                    f'  %{tag}_k0 = add i64 %kbase, {k_off}',
                    f'  %{tag}_k = add i64 %{tag}_k0, %half8',
                ])
                if kind == 'a':
                    load_lines.extend([
                        f'  %{tag}_rowbase = mul i64 %a_row_safe, %K',
                        f'  %{tag}_idx = add i64 %{tag}_rowbase, %{tag}_k',
                        f'  %{tag}_ptr = getelementptr i8, ptr addrspace(1) %a, i64 %{tag}_idx',
                        f'  %{tag}_raw = load i8, ptr addrspace(1) %{tag}_ptr, align 1',
                    ])
                    pack_lines.extend([
                        f'  %{tag}_byte = select i1 %a_row_in, i8 %{tag}_raw, i8 0',
                        f'  %{tag}_z = zext i8 %{tag}_byte to i32',
                    ])
                else:
                    if fragment_word is None:
                        load_lines.extend([
                            f'  %{tag}_kh = lshr i64 %{tag}_k, 1',
                            f'  %{tag}_rowbase = mul i64 %{tag}_kh, %N',
                            f'  %{tag}_idx = add i64 %{tag}_rowbase, %col_safe',
                            f'  %{tag}_ptr = getelementptr i8, ptr addrspace(1) %b, i64 %{tag}_idx',
                            f'  %{tag}_raw8 = load i8, ptr addrspace(1) %{tag}_ptr, align 1',
                        ])
                        pack_lines.append(f'  %{tag}_raw = zext i8 %{tag}_raw8 to i32')
                        code = f'%{tag}_raw'
                        if h & 1:
                            pack_lines.append(f'  %{tag}_shift = lshr i32 {code}, 4')
                            code = f'%{tag}_shift'
                    else:
                        shift = (h // 2) * 8 + (h & 1) * 4
                        code = fragment_word
                        if shift:
                            pack_lines.append(
                                f'  %{tag}_shift = lshr i32 {fragment_word}, {shift}'
                            )
                            code = f'%{tag}_shift'
                    pack_lines.extend([
                        f'  %{tag}_code = and i32 {code}, 15',
                        f'  %{tag}_mag = and i32 %{tag}_code, 7',
                        f'  %{tag}_mag64 = zext i32 %{tag}_mag to i64',
                        f'  %{tag}_tab = getelementptr inbounds [8 x i8], ptr addrspace(4) '
                        f'@tessera_e2m1_to_e4m3, i64 0, i64 %{tag}_mag64',
                        f'  %{tag}_mag8 = load i8, ptr addrspace(4) %{tag}_tab, align 1',
                        f'  %{tag}_magz = zext i8 %{tag}_mag8 to i32',
                        f'  %{tag}_sign = and i32 %{tag}_code, 8',
                        f'  %{tag}_signbit = shl i32 %{tag}_sign, 4',
                        f'  %{tag}_e4m3 = or i32 %{tag}_magz, %{tag}_signbit',
                        f'  %{tag}_z = select i1 %col_in, i32 %{tag}_e4m3, i32 0',
                    ])
                value = f'%{tag}_z'
                if byte:
                    pack_lines.append(f'  %{tag}_placed = shl i32 {value}, {byte * 8}')
                    value = f'%{tag}_placed'
                if packed is None:
                    packed = value
                else:
                    pack_lines.append(f'  %{tag}_or = or i32 {packed}, {value}')
                    packed = f'%{tag}_or'
            assert packed is not None
            words.append(packed)
        pack_lines.extend([
            f'  %{kind}_s{slab}_v0 = insertelement <2 x i32> poison, i32 {words[0]}, i64 0',
            f'  %{kind}_s{slab}_v = insertelement <2 x i32> %{kind}_s{slab}_v0, i32 {words[1]}, i64 1',
        ])
        return load_lines, pack_lines, f'%{kind}_s{slab}_v'

    a_fragments = [build_fragment('a', 0), build_fragment('a', 1)]
    b_fragments = [build_fragment('b', 0), build_fragment('b', 1)]
    # ROCM-MXFP4-W4A8-1: the lds-copy-depth principle without an LDS roundtrip.
    # Make all independent A/B memory operations visible before any byte
    # decode or WMMA dependency so LLVM can keep multiple VMEM operations in
    # flight instead of manufacturing a load/decode/wait chain per element.
    if shared_b:
        for load_batch, _, _ in a_fragments:
            lines.extend(load_batch)
        for _, packing, _ in a_fragments:
            lines.extend(packing)
        a0, a1 = (fragment[2] for fragment in a_fragments)
        lines += [
            '  %is_loader_wave = icmp eq i32 %wave32, 0',
            '  br i1 %is_loader_wave, label %b.load, label %b.wait',
            '',
            'b.load:',
        ]
        for load_batch, _, _ in b_fragments:
            lines.extend(load_batch)
        for _, packing, _ in b_fragments:
            lines.extend(packing)
        for slab, fragment in enumerate(b_fragments):
            offset = slab * 256
            lines += [
                f'  %b_lds_lane_{slab} = mul i32 %lane32, 8',
                f'  %b_lds_off_{slab} = add i32 %b_lds_lane_{slab}, {offset}',
                f'  %b_lds_ptr_{slab} = getelementptr i8, ptr addrspace(3) '
                f'@tessera_mxfp4_b_lds, i32 %b_lds_off_{slab}',
                f'  store <2 x i32> {fragment[2]}, ptr addrspace(3) '
                f'%b_lds_ptr_{slab}, align 8',
            ]
        lines += [
            '  br label %b.wait',
            '',
            'b.wait:',
            '  call void @llvm.amdgcn.s.barrier()',
        ]
        b_values: list[str] = []
        for slab in range(2):
            offset = slab * 256
            lines += [
                f'  %b_lds_read_lane_{slab} = mul i32 %lane32, 8',
                f'  %b_lds_read_off_{slab} = add i32 %b_lds_read_lane_{slab}, {offset}',
                f'  %b_lds_read_ptr_{slab} = getelementptr i8, ptr addrspace(3) '
                f'@tessera_mxfp4_b_lds, i32 %b_lds_read_off_{slab}',
                f'  %b_lds_v_{slab} = load <2 x i32>, ptr addrspace(3) '
                f'%b_lds_read_ptr_{slab}, align 8',
            ]
            b_values.append(f'%b_lds_v_{slab}')
        b0, b1 = b_values
    else:
        # Preserve the decode load-batching contract: expose every independent
        # A/B VMEM operation before starting byte unpack/decode dependencies.
        for load_batch, _, _ in (*a_fragments, *b_fragments):
            lines.extend(load_batch)
        for _, packing, _ in (*a_fragments, *b_fragments):
            lines.extend(packing)
        a0, a1 = (fragment[2] for fragment in a_fragments)
        b0, b1 = (fragment[2] for fragment in b_fragments)
    lines += [
        f'  %partial0 = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8('
        f'<2 x i32> {a0}, <2 x i32> {b0}, <8 x float> zeroinitializer)',
        f'  %partial = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8('
        f'<2 x i32> {a1}, <2 x i32> {b1}, <8 x float> %partial0)',
        '  %bs_row = mul i64 %group, %N',
        '  %bs_idx = add i64 %bs_row, %col_safe',
        '  %bs_ptr = getelementptr i8, ptr addrspace(1) %b_scale, i64 %bs_idx',
        '  %bs_raw = load i8, ptr addrspace(1) %bs_ptr, align 1',
        '  %bs_z = zext i8 %bs_raw to i32',
        '  %bs_bits = shl i32 %bs_z, 23',
        '  %bs_pow2 = bitcast i32 %bs_bits to float',
        '  %bs_is_zero = icmp eq i8 %bs_raw, 0',
        '  %bs_value0 = select i1 %bs_is_zero, float 0.000000e+00, float %bs_pow2',
        '  %bs_value = select i1 %col_in, float %bs_value0, float 0.000000e+00',
        '  %bs_seed = insertelement <8 x float> poison, float %bs_value, i64 0',
        '  %bs_vec = shufflevector <8 x float> %bs_seed, <8 x float> poison, <8 x i32> zeroinitializer',
        f'  %scale_vec = fmul <8 x float> {scale_vec}, %bs_vec',
        '  %scaled_partial = fmul <8 x float> %partial, %scale_vec',
        '  %running_next = fadd <8 x float> %running, %scaled_partial',
        f'  %group_next = add i64 %group, {split_k}',
    ]
    if shared_b:
        lines.append('  call void @llvm.amdgcn.s.barrier()')
    if schedule.k_step_schedule == "isolated_scale_group":
        # The Tile/Target carrier states a backend-neutral no-motion boundary.
        # AMD lowers that contract to a compiler scheduling barrier, not a
        # workgroup synchronization instruction: no target intrinsic leaks
        # into shared IR, and no runtime wait is added here.
        lines.append('  call void @llvm.amdgcn.sched.barrier(i32 0)')
    lines += ['  br label %group.header', '']
    final_running = '%running'
    if split_reduce:
        lines += [
            'split.store:',
            '  %partial_tid64 = zext i32 %tid to i64',
            '  %partial_off = mul i64 %partial_tid64, 32',
            '  %partial_ptr = getelementptr i8, ptr addrspace(3) '
            '@tessera_mxfp4_partial_lds, i64 %partial_off',
            '  store <8 x float> %running, ptr addrspace(3) %partial_ptr, align 16',
            '  call void @llvm.amdgcn.s.barrier()',
            '  %is_reduction_wave = icmp eq i32 %wave32, 0',
            '  br i1 %is_reduction_wave, label %split.reduce, label %split.done',
            '',
            'split.reduce:',
        ]
        reduced = 'zeroinitializer'
        for split in range(split_k):
            lines += [
                f'  %split_lane_{split} = add i32 %lane32, {split * 32}',
                f'  %split_lane64_{split} = zext i32 %split_lane_{split} to i64',
                f'  %split_off_{split} = mul i64 %split_lane64_{split}, 32',
                f'  %split_ptr_{split} = getelementptr i8, ptr addrspace(3) '
                f'@tessera_mxfp4_partial_lds, i64 %split_off_{split}',
                f'  %split_v_{split} = load <8 x float>, ptr addrspace(3) '
                f'%split_ptr_{split}, align 16',
                f'  %split_sum_{split} = fadd <8 x float> {reduced}, %split_v_{split}',
            ]
            reduced = f'%split_sum_{split}'
        final_running = reduced
        lines += ['  br label %store.header', '', 'split.done:', '  ret void', '']
    lines.append('store.header:')

    for j in range(8):
        next_label = f'store.check.{j + 1}' if j < 7 else 'store.done'
        lines += [
            f'  %store_ok_{j} = and i1 %row_in_{j}, %col_in',
            f'  br i1 %store_ok_{j}, label %store.do.{j}, label %{next_label}',
            '',
            f'store.do.{j}:',
            f'  %out_f_{j} = extractelement <8 x float> {final_running}, i64 {j}',
            f'  %out_bits_{j} = bitcast float %out_f_{j} to i32',
            f'  %out_top_{j} = lshr i32 %out_bits_{j}, 16',
            f'  %out_lsb_{j} = and i32 %out_top_{j}, 1',
            f'  %out_bias_{j} = add i32 %out_lsb_{j}, 32767',
            f'  %out_round_{j} = add i32 %out_bits_{j}, %out_bias_{j}',
            f'  %out_shift_{j} = lshr i32 %out_round_{j}, 16',
            f'  %out_bf16_{j} = trunc i32 %out_shift_{j} to i16',
            f'  %out_rowoff_{j} = mul i64 %row_{j}, %N',
            f'  %out_idx_{j} = add i64 %out_rowoff_{j}, %col',
            f'  %out_ptr_{j} = getelementptr i16, ptr addrspace(1) %out, i64 %out_idx_{j}',
            f'  store i16 %out_bf16_{j}, ptr addrspace(1) %out_ptr_{j}, align 2',
            f'  br label %{next_label}',
            '',
        ]
        if j < 7:
            lines.append(f'store.check.{j + 1}:')
    lines += [
        'store.done:',
        '  ret void',
        '}',
        '',
        'declare i32 @llvm.amdgcn.workitem.id.x()',
        'declare i32 @llvm.amdgcn.workgroup.id.x()',
        'declare i32 @llvm.amdgcn.workgroup.id.y()',
    ]
    if shared_b or split_reduce:
        lines.append('declare void @llvm.amdgcn.s.barrier()')
    if schedule.k_step_schedule == "isolated_scale_group":
        lines.append('declare void @llvm.amdgcn.sched.barrier(i32 immarg)')
    lines += [
        'declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8('
        '<2 x i32>, <2 x i32>, <8 x float>)',
        '',
        f'attributes #0 = {{ "amdgpu-flat-work-group-size"="{32 * waves_per_workgroup},{32 * waves_per_workgroup}" }}',
        '',
    ]
    return '\n'.join(lines)


def mxfp4_w4a8_descriptor(
    image: NativeImageArtifact,
    *,
    m: int,
    n: int,
    k: int,
    entry: str = "tessera_mxfp4_w4a8_exact",
    abi_id: str = GFX_MXFP4_W4A8_EXACT_ABI,
    route: str = "exact_per_block_scalar_baseline",
    workgroup: tuple[int, int, int] = (16, 16, 1),
    weight_layout: str = MXFP4_TRANSPOSED_LAYOUT_V1,
) -> LaunchDescriptor:
    """Build the exact five-buffer W4A8 launch contract."""

    if min(m, n, k) <= 0 or k % 32:
        raise ValueError("MXFP4 W4A8 requires positive M/N and K divisible by 32")
    if image.target != "rocm_gfx1201" or image.architecture != "gfx1201":
        raise ValueError("MXFP4 W4A8 native packages are exact gfx1201 artifacts")
    mxfp4_weight_layout(weight_layout)
    if weight_layout == MXFP4_GFX12_FRAGMENT_LAYOUT_V1 and n % 16:
        raise ValueError("gfx12 fragment-order MXFP4 requires N divisible by 16")
    if weight_layout == MXFP4_TRANSPOSED_LAYOUT_V1:
        b_shape = (k // 2, n)
    elif weight_layout == MXFP4_GFX12_FRAGMENT_LAYOUT_V1:
        b_shape = (n, k // 2)
    else:
        raise ValueError(f"MXFP4 launch cannot consume weight layout {weight_layout!r}")
    return LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol=entry,
        abi_id=abi_id,
        buffers=(
            BufferBinding(0, "a", "input", "uint8", 2, "row_major", 1),
            # BufferBinding.layout describes the host array's stride class.
            # The packed-byte interpretation is the versioned weight_layout
            # provenance field and the distinct package ABI below.
            BufferBinding(1, "b_packed", "input", "uint8", 2, "row_major", 1),
            BufferBinding(2, "a_scale", "input", "fp32", 1, "row_major", 4),
            BufferBinding(3, "b_scale", "input", "uint8", 2, "row_major", 1),
            BufferBinding(4, "output", "output", "bf16", 2, "row_major", 2),
        ),
        scalars=(
            ScalarArgument(5, "M", "int64"),
            ScalarArgument(6, "N", "int64"),
            ScalarArgument(7, "K", "int64"),
        ),
        shape_guards=(
            ShapeGuard("a", 0, "eq", m), ShapeGuard("a", 1, "eq", k),
            ShapeGuard("b_packed", 0, "eq", b_shape[0]),
            ShapeGuard("b_packed", 1, "eq", b_shape[1]),
            ShapeGuard("a_scale", 0, "eq", m),
            ShapeGuard("b_scale", 0, "eq", k // 32), ShapeGuard("b_scale", 1, "eq", n),
            ShapeGuard("output", 0, "eq", m), ShapeGuard("output", 1, "eq", n),
        ),
        geometry=LaunchGeometry(
            grid=((n + 15) // 16, (m + 15) // 16, 1),
            workgroup=workgroup,
        ),
        ordering=OrderingSemantics(
            ordered_submission=True,
            residency="none",
            synchronization=("completion",),
        ),
        provenance={
            "work_item": "ROCM-MXFP4-W4A8-1",
            "sync_key": "ROCM-MXFP4-PHYSICAL-CONTRACT-2026-09-21",
            "route": route,
            "architecture": "gfx1201",
            "shape": [m, n, k],
            "activation_storage": "e4m3_raw_uint8",
            "weight_storage": "e2m1_packed_low_nibble_even_k",
            "weight_layout": weight_layout,
            "activation_scale": "fp32_per_token",
            "weight_scale": "e8m0_k32_kn_major",
            "scale_group_k": 32,
            "accum": "fp32",
            "output": "bf16",
            "numeric_policy": "exact_per_block",
        },
    )


def package_mxfp4_w4a8_exact(
    m: int,
    n: int,
    k: int,
    *,
    pipeline_name: str = "tessera-lower-to-rocm",
    entry: str = "tessera_mxfp4_w4a8_exact",
) -> ROCMNativePackage:
    """Compile the exact scalar gfx1201 baseline to one HSACO package."""

    if min(m, n, k) <= 0 or k % 32:
        raise ValueError("MXFP4 W4A8 requires positive M/N and K divisible by 32")
    source = emit_mxfp4_w4a8_exact_hip(entry=entry)
    rocm_path = _rocm_path()
    compiler = _rocm_hipcc(rocm_path)
    if compiler is None:
        raise RuntimeError("MXFP4 W4A8 packaging requires the HIP compiler driver")
    device_libraries = _driver_selected_device_libraries(arch="gfx1201")
    with tempfile.TemporaryDirectory(prefix="tessera-mxfp4-") as directory:
        source_path = Path(directory) / "kernel.hip"
        bundle_path = Path(directory) / "kernel.hipfb"
        image_path = Path(directory) / "kernel.hsaco"
        source_path.write_text(source)
        command = [
            str(compiler), "-x", "hip", "-O3", "--genco",
            "--offload-arch=gfx1201", f"--rocm-path={rocm_path}",
            str(source_path), "-o", str(bundle_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode or not bundle_path.is_file():
            detail = result.stderr.strip() or f"AMD clang exited {result.returncode}"
            raise RuntimeError(f"MXFP4 W4A8 HSACO compilation failed: {detail}")
        payload = _extract_gfx1201_hsaco(bundle_path, image_path, rocm_path)
    toolchain_fingerprint = hashlib.sha256(
        (str(rocm_path) + "|gfx1201|O3|exact_per_block").encode()
    ).hexdigest()
    image = NativeImageArtifact(
        target="rocm_gfx1201",
        architecture="gfx1201",
        pipeline_name=pipeline_name,
        compiler_fingerprint=_version_fingerprint(compiler),
        toolchain_fingerprint=toolchain_fingerprint,
        target_ir_digest=hashlib.sha256(source.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, GFX_MXFP4_W4A8_EXACT_ABI),),
        compile_state="cold",
        device_libraries=device_libraries,
    )
    descriptor = mxfp4_w4a8_descriptor(image, m=m, n=n, k=k, entry=entry)
    semantic_ir = (
        f"rocm.mxfp4_w4a8 exact_per_block M={m} N={n} K={k} "
        "activation=e4m3 scale_a=fp32_per_token weight=e2m1_packed "
        "scale_b=e8m0_k32 accum=fp32 output=bf16"
    )
    return ROCMNativePackage(semantic_ir, source, " ".join(command[:-2]), image, descriptor)


def package_mxfp4_w4a8_wmma(
    m: int,
    n: int,
    k: int,
    *,
    pipeline_name: str = "tessera-lower-to-rocm",
    entry: str = "tessera_mxfp4_w4a8_wmma",
    schedule: MXFP4Schedule | None = None,
    weight_layout: str | None = None,
) -> ROCMNativePackage:
    """Compile the exact two-WMMA-per-K32 gfx1201 production route."""

    if min(m, n, k) <= 0 or k % 32:
        raise ValueError("MXFP4 W4A8 requires positive M/N and K divisible by 32")
    schedule = schedule or select_mxfp4_schedule(m, n, k)
    receipt = select_mxfp4_route(
        m, n, k, requested_layout=weight_layout, schedule=schedule
    )
    if not receipt.accepted or receipt.selected_layout is None or receipt.abi_id is None:
        raise ValueError(f"MXFP4 route refused: {receipt.reason}")
    weight_layout = receipt.selected_layout
    abi_id = receipt.abi_id
    source = emit_mxfp4_w4a8_wmma_llvmir(
        entry=entry, schedule=schedule, weight_layout=weight_layout
    )
    rocm_path = _rocm_path()
    compiler = _rocm_clang(rocm_path)
    if compiler is None:
        raise RuntimeError("MXFP4 W4A8 WMMA packaging requires AMD clang")
    with tempfile.TemporaryDirectory(prefix="tessera-mxfp4-wmma-") as directory:
        source_path = Path(directory) / "kernel.ll"
        image_path = Path(directory) / "kernel.hsaco"
        source_path.write_text(source)
        command = [
            str(compiler), "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1201",
            "-x", "ir", "-O3", "-nogpulib", "-shared",
            str(source_path), "-o", str(image_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode or not image_path.is_file():
            detail = result.stderr.strip() or f"AMD clang exited {result.returncode}"
            raise RuntimeError(f"MXFP4 W4A8 WMMA HSACO compilation failed: {detail}")
        payload = image_path.read_bytes()
    if not payload.startswith(b"\x7fELF"):
        raise RuntimeError("MXFP4 W4A8 WMMA compiler output is not an ELF HSACO")
    image = NativeImageArtifact(
        target="rocm_gfx1201",
        architecture="gfx1201",
        pipeline_name=pipeline_name,
        compiler_fingerprint=_version_fingerprint(compiler),
        toolchain_fingerprint=hashlib.sha256(
            (str(rocm_path) + f"|gfx1201|O3|wmma_exact_k32|{schedule}").encode()
        ).hexdigest(),
        target_ir_digest=hashlib.sha256(source.encode()).hexdigest(),
        binary_format="hsaco",
        payload=payload,
        entry_points=(NativeEntryPoint(entry, abi_id),),
        compile_state="cold",
        device_libraries=(),
    )
    descriptor = mxfp4_w4a8_descriptor(
        image,
        m=m,
        n=n,
        k=k,
        entry=entry,
        abi_id=abi_id,
        route="exact_per_block_fp8_wmma",
        workgroup=(32 * max(schedule.group_m, schedule.split_k), 1, 1),
        weight_layout=weight_layout,
    )
    descriptor = replace(
        descriptor,
        geometry=replace(
            descriptor.geometry,
            grid=(
                (n + 15) // 16,
                (m + 16 * schedule.group_m - 1) // (16 * schedule.group_m),
                1,
            ),
        ),
        provenance={
            **descriptor.provenance,
            "workload": schedule.workload,
            "group_m": schedule.group_m,
            "split_k": schedule.split_k,
            "stages": schedule.stages,
            "waves_per_eu": schedule.waves_per_eu,
            "cache_modifier": schedule.cache_modifier,
            "k_step_schedule": schedule.k_step_schedule,
            "selection_receipt": receipt.as_dict(),
        },
    )
    semantic_ir = (
        f"rocm.mxfp4_w4a8 wmma_exact_per_block M={m} N={n} K={k} "
        "activation=e4m3 scale_a=fp32_per_token weight=e2m1_packed "
        "scale_b=e8m0_k32 instruction=v_wmma_f32_16x16x16_fp8_fp8 "
        f"accum=fp32 output=bf16 workload={schedule.workload} "
        f"group_m={schedule.group_m} k_step_schedule={schedule.k_step_schedule} "
        f"weight_layout={weight_layout}"
    )
    return ROCMNativePackage(semantic_ir, source, " ".join(command[:-2]), image, descriptor)


def package_mxfp4_w4a8(
    m: int,
    n: int,
    k: int,
    *,
    route: str = "wmma",
    pipeline_name: str = "tessera-lower-to-rocm",
) -> ROCMNativePackage:
    """Package the proved gfx1201 route, defaulting to the production WMMA ABI.

    The scalar route remains available as an executable specification and
    device oracle.  Callers must opt into it explicitly so a production call
    cannot silently lose the native FP8 matrix instruction.
    """

    if route == "wmma":
        return package_mxfp4_w4a8_wmma(
            m, n, k, pipeline_name=pipeline_name
        )
    if route == "scalar_reference":
        return package_mxfp4_w4a8_exact(
            m, n, k, pipeline_name=pipeline_name
        )
    raise ValueError(
        "MXFP4 W4A8 route must be 'wmma' or 'scalar_reference'"
    )


__all__ = [
    "GFX_MXFP4_W4A8_EXACT_ABI",
    "GFX_MXFP4_W4A8_WMMA_ABI",
    "GFX_MXFP4_W4A8_WMMA_FRAGMENT_ABI",
    "MXFP4Schedule",
    "MXFP4RouteReceipt",
    "emit_mxfp4_w4a8_exact_hip",
    "emit_mxfp4_w4a8_wmma_llvmir",
    "mxfp4_w4a8_descriptor",
    "package_mxfp4_w4a8",
    "package_mxfp4_w4a8_exact",
    "package_mxfp4_w4a8_wmma",
    "package_scaled_wmma_target_ir",
    "select_mxfp4_schedule",
    "select_mxfp4_route",
]
