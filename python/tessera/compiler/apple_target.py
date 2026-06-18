"""tessera.compiler.apple_target — Apple GPU feature-limits target profile.

Audit Action 10 (2026-06-01) — per-arch capability table for Apple
Silicon GPUs. Parallels ``gpu_target.py`` (NVIDIA) and
``rocm_target.py`` (AMD): a static-per-arch feature matrix + dtype set
+ instruction-shape table, plus a profile dataclass that exposes
``supports_*`` predicates the lowering passes consult.

Architectures tracked:

* ``M1`` (Apple7, 2020) — first Apple Silicon GPU
* ``M2`` (Apple8, 2022) — adds native ``bfloat`` + simdgroup matrix
* ``M3`` (Apple9, 2023) — adds hardware ray tracing, dynamic caching,
  mesh shaders
* ``M4`` (Apple10, 2024) — adds Metal 4 + neural accelerators + MTL4
  packaged ML pipelines (``MTL4MachineLearningPipelineState``)
* ``M5`` (Apple11, 2025) — neural-accelerator throughput improvements

The static table is the COMPILATION-TIME contract — what the backend
can lower to for each arch. A separate ``probe_apple_runtime_limits()``
helper consults the live runtime for HOST-SPECIFIC limits
(``maxThreadgroupMemoryLength`` etc.) that vary by SKU within an arch.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional


class AppleGPUArch(IntEnum):
    """Apple GPU architecture family identifiers. Integer values
    follow Apple's internal ``MTLGPUFamilyApple*`` numbering so
    monotone comparisons match generation ordering."""
    APPLE7 = 7    # M1 (2020 — first Apple Silicon)
    APPLE8 = 8    # M2 (2022 — bfloat16 + simdgroup matrix)
    APPLE9 = 9    # M3 (2023 — RT, dynamic caching, mesh shaders)
    APPLE10 = 10  # M4 (2024 — Metal 4 + neural accelerators)
    APPLE11 = 11  # M5 (2025 — neural-accelerator throughput)


#: Target Metal release that Tessera's Apple GPU backend is built
#: against. MSL 4.0 (Metal 4) shipped with macOS 26 (~2025).
TESSERA_TARGET_METAL: str = "4.0"
TESSERA_TARGET_MACOS_FOR_MTL4: str = "26.0"


# ---------------------------------------------------------------------
# Per-arch feature matrix.
#
# Status values:
#   "ready"        — supported by Metal 4 on this arch
#   "tba"          — present in the arch but not yet exposed by Tessera
#   "not_supported"— architecturally unavailable
# ---------------------------------------------------------------------
_APPLE_FEATURES: dict[AppleGPUArch, dict[str, str]] = {
    AppleGPUArch.APPLE7: {
        # M1 baseline. No native bfloat, no Metal 4, no neural
        # accelerators. Metal 3 + MPS / MPSGraph paths are the
        # entire surface here.
        "metal3":                "ready",
        "metal4":                "not_supported",
        "mpsgraph":              "ready",
        "simdgroup":             "ready",
        "simdgroup_async_copy":  "ready",
        "simdgroup_matrix":      "not_supported",
        "bfloat":                "not_supported",
        "ray_tracing":           "not_supported",
        "dynamic_caching":       "not_supported",
        "mesh_shaders":          "not_supported",
        "neural_accelerators":   "not_supported",
        "mtl4_packaged_ml":      "not_supported",
        "mtl4_compiler":         "not_supported",
        "mtl4_command_queue":    "not_supported",
        "function_pointers":     "ready",
        "argument_buffers":      "ready",
    },
    AppleGPUArch.APPLE8: {
        # M2 — adds native bfloat + simdgroup matrix multiply
        # (``simdgroup_bfloat8x8`` etc.). Still pre-Metal-4.
        "metal3":                "ready",
        "metal4":                "not_supported",
        "mpsgraph":              "ready",
        "simdgroup":             "ready",
        "simdgroup_async_copy":  "ready",
        "simdgroup_matrix":      "ready",
        "bfloat":                "ready",
        "ray_tracing":           "not_supported",
        "dynamic_caching":       "not_supported",
        "mesh_shaders":          "not_supported",
        "neural_accelerators":   "not_supported",
        "mtl4_packaged_ml":      "not_supported",
        "mtl4_compiler":         "not_supported",
        "mtl4_command_queue":    "not_supported",
        "function_pointers":     "ready",
        "argument_buffers":      "ready",
    },
    AppleGPUArch.APPLE9: {
        # M3 — hardware RT, dynamic caching, mesh shaders. Still
        # Metal 3 (Metal 4 ships with M4 + macOS 26).
        "metal3":                "ready",
        "metal4":                "not_supported",
        "mpsgraph":              "ready",
        "simdgroup":             "ready",
        "simdgroup_async_copy":  "ready",
        "simdgroup_matrix":      "ready",
        "bfloat":                "ready",
        "ray_tracing":           "ready",
        "dynamic_caching":       "ready",
        "mesh_shaders":          "ready",
        "neural_accelerators":   "not_supported",
        "mtl4_packaged_ml":      "not_supported",
        "mtl4_compiler":         "not_supported",
        "mtl4_command_queue":    "not_supported",
        "function_pointers":     "ready",
        "argument_buffers":      "ready",
    },
    AppleGPUArch.APPLE10: {
        # M4 — Metal 4 baseline. MTL4 compiler / command queue /
        # packaged ML all light up. Neural accelerators first appear.
        "metal3":                "ready",
        "metal4":                "ready",
        "mpsgraph":              "ready",
        "simdgroup":             "ready",
        "simdgroup_async_copy":  "ready",
        "simdgroup_matrix":      "ready",
        "bfloat":                "ready",
        "ray_tracing":           "ready",
        "dynamic_caching":       "ready",
        "mesh_shaders":          "ready",
        "neural_accelerators":   "ready",
        "mtl4_packaged_ml":      "ready",
        "mtl4_compiler":         "ready",
        "mtl4_command_queue":    "ready",
        "function_pointers":     "ready",
        "argument_buffers":      "ready",
    },
    AppleGPUArch.APPLE11: {
        # M5 — Metal 4 + improved neural-accelerator throughput.
        # Same compile-time surface as M4; throughput differences
        # surface through benchmark numbers, not the feature matrix.
        "metal3":                "ready",
        "metal4":                "ready",
        "mpsgraph":              "ready",
        "simdgroup":             "ready",
        "simdgroup_async_copy":  "ready",
        "simdgroup_matrix":      "ready",
        "bfloat":                "ready",
        "ray_tracing":           "ready",
        "dynamic_caching":       "ready",
        "mesh_shaders":          "ready",
        "neural_accelerators":   "ready",
        "mtl4_packaged_ml":      "ready",
        "mtl4_compiler":         "ready",
        "mtl4_command_queue":    "ready",
        "function_pointers":     "ready",
        "argument_buffers":      "ready",
    },
}


# Per-arch dtype matrix accepted by the Apple Metal backend. Canonical
# Tessera dtype spellings (validated by ``tessera.dtype.canonicalize_dtype``).
# No Apple GPU supports fp64 natively as of M5; fp64 ops fall back to
# CPU per Decision #19 (hardware-free Target IR keeps the option open).
_APPLE_DTYPES: dict[AppleGPUArch, frozenset[str]] = {
    AppleGPUArch.APPLE7: frozenset({
        "fp32", "fp16", "int8", "int16", "int32", "int64", "bool",
    }),
    AppleGPUArch.APPLE8: frozenset({
        "fp32", "fp16", "bf16",
        "int8", "int16", "int32", "int64", "bool",
    }),
    AppleGPUArch.APPLE9: frozenset({
        "fp32", "fp16", "bf16",
        "int8", "int16", "int32", "int64", "bool",
    }),
    AppleGPUArch.APPLE10: frozenset({
        "fp32", "fp16", "bf16",
        "int8", "int16", "int32", "int64", "bool",
    }),
    AppleGPUArch.APPLE11: frozenset({
        "fp32", "fp16", "bf16",
        "int8", "int16", "int32", "int64", "bool",
    }),
}


# Default per-arch limits. These are the COMPILATION-TIME conservative
# floor — the host runtime can report HIGHER limits via
# ``probe_apple_runtime_limits()`` for the specific SKU. The lowering
# passes consult these to decide which kernel variant is reachable.
@dataclass(frozen=True)
class _ArchDefaults:
    threadgroup_memory_bytes: int  # MTLDevice.maxThreadgroupMemoryLength floor
    max_threads_per_threadgroup: int
    simdgroup_size: int
    max_argument_buffer_bindings: int


_APPLE_ARCH_DEFAULTS: dict[AppleGPUArch, _ArchDefaults] = {
    # All Apple GPUs use a 32-lane SIMD width and a 1024 max threads /
    # threadgroup ceiling. Threadgroup memory floor is the Apple docs
    # number: 32 KB on the lowest-end M1, ~64 KB on M2+ (per Apple
    # GPU family table). M3+ raises further with dynamic caching but
    # the static floor stays at 32 KB to be safe.
    AppleGPUArch.APPLE7:  _ArchDefaults(32 * 1024, 1024, 32, 31),
    AppleGPUArch.APPLE8:  _ArchDefaults(32 * 1024, 1024, 32, 31),
    AppleGPUArch.APPLE9:  _ArchDefaults(32 * 1024, 1024, 32, 31),
    AppleGPUArch.APPLE10: _ArchDefaults(32 * 1024, 1024, 32, 31),
    AppleGPUArch.APPLE11: _ArchDefaults(32 * 1024, 1024, 32, 31),
}


# Per-arch MSL toolchain string. Used by build-system glue + the
# ``apple_arch_string()`` helper.
_APPLE_ARCH_STRINGS: dict[AppleGPUArch, str] = {
    AppleGPUArch.APPLE7:  "apple7",
    AppleGPUArch.APPLE8:  "apple8",
    AppleGPUArch.APPLE9:  "apple9",
    AppleGPUArch.APPLE10: "apple10",
    AppleGPUArch.APPLE11: "apple11",
}


class TesseraAppleGPUTargetError(Exception):
    """Raised when an AppleGPUTargetProfile has invalid settings."""


@dataclass
class AppleGPUTargetProfile:
    """Describes the Apple GPU target for a ``@jit(target=...)`` function.

    Defaults to ``APPLE10`` (M4) — the first arch where Metal 4 +
    packaged ML are fully reachable. The lowering passes consult
    ``supports_*`` predicates to decide which kernel variant to emit.

    Attributes:
        arch                    : ``AppleGPUArch`` enum member
        threadgroup_memory_bytes: Override the threadgroup-memory budget;
                                  None = use ``_ArchDefaults`` floor
        prefer_packaged_ml      : When the kernel ships as an
                                  ``.mtlpackage``, prefer the packaged-ML
                                  dispatch path over runtime MSL compile
                                  (default True on M4+, False on older)
    """
    arch: AppleGPUArch = AppleGPUArch.APPLE10
    threadgroup_memory_bytes: Optional[int] = None
    prefer_packaged_ml: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.arch, AppleGPUArch):
            try:  # type: ignore[unreachable]
                self.arch = AppleGPUArch(int(self.arch))
            except (ValueError, KeyError):
                raise TesseraAppleGPUTargetError(
                    f"Unknown Apple GPU arch {self.arch!r}. Use an "
                    f"AppleGPUArch enum member, e.g. AppleGPUArch.APPLE10."
                )
        if (self.threadgroup_memory_bytes is not None
                and self.threadgroup_memory_bytes <= 0):
            raise TesseraAppleGPUTargetError(
                f"threadgroup_memory_bytes must be positive, got "
                f"{self.threadgroup_memory_bytes}"
            )
        # On pre-M4 arches packaged ML is structurally unavailable —
        # silently force-off rather than misleading the lowering passes.
        if (self.prefer_packaged_ml
                and apple_feature_status(self.arch, "mtl4_packaged_ml")
                    != "ready"):
            self.prefer_packaged_ml = False

    # ── Capability queries ──────────────────────────────────────────────

    @property
    def supports_metal4(self) -> bool:
        return apple_feature_status(self.arch, "metal4") == "ready"

    @property
    def supports_packaged_ml(self) -> bool:
        return (apple_feature_status(self.arch, "mtl4_packaged_ml")
                == "ready")

    @property
    def supports_bfloat(self) -> bool:
        return apple_feature_status(self.arch, "bfloat") == "ready"

    @property
    def supports_simdgroup_matrix(self) -> bool:
        return (apple_feature_status(self.arch, "simdgroup_matrix")
                == "ready")

    @property
    def supports_ray_tracing(self) -> bool:
        return apple_feature_status(self.arch, "ray_tracing") == "ready"

    @property
    def supports_dynamic_caching(self) -> bool:
        return (apple_feature_status(self.arch, "dynamic_caching")
                == "ready")

    @property
    def supports_neural_accelerators(self) -> bool:
        return (apple_feature_status(self.arch, "neural_accelerators")
                == "ready")

    @property
    def supports_mesh_shaders(self) -> bool:
        return apple_feature_status(self.arch, "mesh_shaders") == "ready"

    @property
    def threadgroup_memory_capacity_bytes(self) -> int:
        if self.threadgroup_memory_bytes is not None:
            return self.threadgroup_memory_bytes
        return _APPLE_ARCH_DEFAULTS[self.arch].threadgroup_memory_bytes

    @property
    def max_threads_per_threadgroup(self) -> int:
        return _APPLE_ARCH_DEFAULTS[self.arch].max_threads_per_threadgroup

    @property
    def simdgroup_size(self) -> int:
        return _APPLE_ARCH_DEFAULTS[self.arch].simdgroup_size

    @property
    def max_argument_buffer_bindings(self) -> int:
        return (_APPLE_ARCH_DEFAULTS[self.arch]
                .max_argument_buffer_bindings)

    @property
    def apple_features(self) -> frozenset[str]:
        """Set of all features marked ``ready`` for this arch."""
        return apple_feature_set(self.arch)

    @property
    def apple_arch(self) -> str:
        return apple_arch_string(self.arch)

    @property
    def dtype_set(self) -> frozenset[str]:
        return _APPLE_DTYPES[self.arch]


def apple_feature_status(arch: AppleGPUArch, feature: str) -> str:
    """Return the per-arch status for ``feature``. Raises ``KeyError``
    if the feature name is unknown — keeps typos loud instead of
    silently returning ``not_supported``."""
    return _APPLE_FEATURES[arch][feature]


def apple_feature_set(arch: AppleGPUArch) -> frozenset[str]:
    """Return the set of feature names with status ``"ready"`` on
    ``arch``. Useful for ``X in profile.apple_features`` checks."""
    return frozenset(
        name for name, status in _APPLE_FEATURES[arch].items()
        if status == "ready"
    )


def apple_arch_string(arch: AppleGPUArch) -> str:
    """Return the canonical Apple arch identifier
    (``"apple10"`` etc.) — used by build-system glue."""
    return _APPLE_ARCH_STRINGS[arch]


def apple_arch_defaults(arch: AppleGPUArch) -> _ArchDefaults:
    """Return the static per-arch limits floor. The runtime probe can
    report higher numbers — see ``probe_apple_runtime_limits``."""
    return _APPLE_ARCH_DEFAULTS[arch]


def apple_threadgroup_tiled_softmax_n_cap(
    arch: AppleGPUArch = AppleGPUArch.APPLE10,
    *,
    runtime_limits: "Optional[AppleRuntimeLimits]" = None,
    elem_bytes: int = 4,
) -> int:
    """Feature-limit-derived N ceiling for the threadgroup-tiled
    matmul→softmax kernel (P1, 2026-06-02).

    The tiled kernel holds one row of N scores in threadgroup memory and
    accumulates in fp32, so the largest N it can serve is
    ``threadgroup_memory_budget // bytes_per_score``. On every current
    Apple arch the static floor is 32 KB ⇒ 8192 fp32 scores — which is
    exactly the constant the runtime used to hardcode. Deriving it from
    the arch limit makes the cap self-documenting and self-scaling: a
    higher-memory SKU (reported by ``runtime_limits``) automatically
    raises it, and the portable static floor is never undercut.

    ``runtime_limits`` (from ``probe_apple_runtime_limits``) is consulted
    only when it reports a budget *above* the static floor — opt-in
    tuning that never drops below the portable artifact's assumption.
    """
    floor = _APPLE_ARCH_DEFAULTS[arch].threadgroup_memory_bytes
    budget = floor
    if (runtime_limits is not None
            and runtime_limits.max_threadgroup_memory_bytes > floor):
        budget = runtime_limits.max_threadgroup_memory_bytes
    return budget // max(1, int(elem_bytes))


# Per-thread stack budget the hand-written fused MSL kernels are compiled
# with (`float scores[...]` / `float out[...]` arrays). 1 KiB of fp32 per
# array — the source of the historical "N <= 256" / "head_dim <= 256" caps.
_FUSED_KERNEL_STACK_BUDGET_BYTES = 1024


def apple_fused_chain_score_cap(
    arch: AppleGPUArch = AppleGPUArch.APPLE10,
    *,
    elem_bytes: int = 4,
) -> int:
    """P2 (2026-06-09) — feature-table-derived per-thread cap for the
    fused matmul→softmax(→matmul) / matmul→gelu / matmul→rmsnorm /
    moe_swiglu stack-array kernels.

    These kernels hold one row of scores in a per-thread stack array, so
    the cap is the per-thread stack budget over the accumulator width —
    1 KiB / 4 B = 256 on every current arch, exactly the constant the
    runtime used to hardcode. Deriving it here makes the cap
    self-documenting and single-sourced; arch is accepted so a future
    family with a different stack budget plugs in per-arch values."""
    del arch  # uniform budget on apple7–apple11; per-arch knob reserved
    return _FUSED_KERNEL_STACK_BUDGET_BYTES // max(1, int(elem_bytes))


def apple_flash_attn_head_dim_cap(
    arch: AppleGPUArch = AppleGPUArch.APPLE10,
) -> int:
    """Max head_dim the single-kernel online-softmax flash-attention MSL
    serves — same per-thread fp32 stack budget as the fused chains."""
    return apple_fused_chain_score_cap(arch)


def apple_threadgroup_threads_per_row(
    arch: AppleGPUArch = AppleGPUArch.APPLE10,
) -> int:
    """Cooperating threads per row for threadgroup-tiled row kernels
    (tiled matmul→softmax) — one SIMD-group wide on every Apple arch."""
    return _APPLE_ARCH_DEFAULTS[arch].simdgroup_size


def apple_supports_native_bf16(
    arch: AppleGPUArch = AppleGPUArch.APPLE10,
    *,
    runtime_limits: "Optional[AppleRuntimeLimits]" = None,
) -> bool:
    """P2 (2026-06-09) — bf16 native-vs-host-upcast gate from the feature
    table (``bfloat`` is ready on apple8+; apple7/M1 host-upcasts).

    When ``runtime_limits`` carries a positive ``MTLGPUFamilyApple*``
    raw value the live family wins (e.g. 1007 = apple7 ⇒ False); the
    static arch default is the off-Metal floor."""
    if runtime_limits is not None and runtime_limits.apple_gpu_family > 0:
        return runtime_limits.apple_gpu_family >= 1008
    return apple_feature_status(arch, "bfloat") == "ready"


# ---- Runtime probe ------------------------------------------------------

@dataclass(frozen=True)
class AppleRuntimeLimits:
    """Live host limits queried from ``MTLDevice``. The static
    ``_APPLE_ARCH_DEFAULTS`` floor is what the BACKEND can assume; this
    is what the SPECIFIC SKU on this host actually exposes. Lowering
    passes should prefer the static floor (portable artifact); runtime
    dispatch can consult the probe for opt-in tuning."""
    max_threadgroup_memory_bytes: int
    supports_packaged_ml: bool
    supports_metal4: bool
    apple_gpu_family: int  # ``MTLGPUFamilyApple*`` raw value


def probe_apple_runtime_limits() -> Optional[AppleRuntimeLimits]:
    """Query the live ``MetalDeviceContext`` for host-specific limits.
    Returns ``None`` on non-Darwin / when the runtime dylib isn't
    loadable / when no Metal device is available.

    Probes:

    * ``[device maxThreadgroupMemoryLength]`` — actual byte cap on
      this SKU (M3 Pro reports 32 KB, M3 Max reports 32 KB too;
      M4 / M5 hit higher numbers depending on family).
    * Metal 4 + packaged-ML availability via the existing
      ``tessera_apple_gpu_metal4_probe`` C ABI.
    * ``MTLGPUFamilyApple*`` integer.
    """
    try:
        from .._apple_gpu_dispatch import apple_gpu_runtime, bind_symbol
    except ImportError:
        return None

    if apple_gpu_runtime() is None:
        return None

    # Threadgroup memory probe.
    tgm_probe = bind_symbol(
        "tessera_apple_gpu_max_threadgroup_memory_length",
        (), restype=ctypes.c_int64)
    if tgm_probe is None:
        # Older runtime — fall back to "unknown" sentinel 0; callers
        # should treat 0 as "consult the static floor."
        tgm = 0
    else:
        tgm = int(tgm_probe())
        if tgm < 0:
            tgm = 0

    # Metal 4 / packaged-ML probe (reuses the PK1 capability probe).
    m4_probe = bind_symbol(
        "tessera_apple_gpu_metal4_probe",
        (ctypes.POINTER(ctypes.c_int32),),
        restype=ctypes.c_int32)
    if m4_probe is None:
        m4_ok = False
        packaged_ml_ok = False
    else:
        caps = ctypes.c_int32(0)
        m4_ok = bool(m4_probe(ctypes.byref(caps)))
        # Packaged ML is part of the MTL4 surface. Until a finer-grained
        # capability bit lands, treat m4_ok as authoritative for both.
        packaged_ml_ok = m4_ok

    # Apple GPU family integer (best-effort — older runtimes return -1).
    fam_probe = bind_symbol(
        "tessera_apple_gpu_family_integer",
        (), restype=ctypes.c_int32)
    if fam_probe is None:
        family = -1
    else:
        family = int(fam_probe())

    return AppleRuntimeLimits(
        max_threadgroup_memory_bytes=tgm,
        supports_packaged_ml=packaged_ml_ok,
        supports_metal4=m4_ok,
        apple_gpu_family=family,
    )


# ---------------------------------------------------------------------
# Feature-probe vocabulary (MLX mining, 2026-06-17). The static
# ``_APPLE_ARCH_DEFAULTS`` floor above is the COMPILE-REQUIRED surface;
# ``probe_apple_runtime_limits`` is the RUNTIME-OBSERVED surface. MLX's
# ``device.cpp`` surfaces sharper probes — Metal language version, OS
# availability, architecture generation, and **NAX availability** (the
# M5/Apple10-class matrix unit). This section adds the explicit kind
# classification + the grounded NAX gate (consumed by the steel-attention
# NAX path in ``apple_sdpa_schedules.select_attn_schedule``).
# ---------------------------------------------------------------------
class AppleProbeKind(Enum):
    COMPILE_REQUIRED = "compile_required"   # baked into the kernel; must hold to build
    RUNTIME_OBSERVED = "runtime_observed"   # queried from MTLDevice; varies per machine


@dataclass(frozen=True)
class AppleFeatureProbe:
    name: str
    kind: AppleProbeKind
    detail: str


#: The sharper probe vocabulary MLX surfaces (the four the deep-dive called out).
APPLE_FEATURE_PROBES: tuple[AppleFeatureProbe, ...] = (
    AppleFeatureProbe(
        "metal_language_version", AppleProbeKind.COMPILE_REQUIRED,
        "MTLLanguageVersion the kernel compiles at (bf16→3.1, tensors/ML→4.0)"),
    AppleFeatureProbe(
        "apple_gpu_family", AppleProbeKind.COMPILE_REQUIRED,
        "MTLGPUFamily — gates simdgroup_matrix / MTLTensor / ML encoding (Apple7+)"),
    AppleFeatureProbe(
        "os_availability", AppleProbeKind.RUNTIME_OBSERVED,
        "__builtin_available(macOS X) — NAX needs 26.2; FP8/FP4 MTLTensor need 27.0"),
    AppleFeatureProbe(
        "arch_generation", AppleProbeKind.RUNTIME_OBSERVED,
        "GPU architecture generation (MLX device.cpp get_architecture_gen) — gates NAX"),
    AppleFeatureProbe(
        "nax_available", AppleProbeKind.RUNTIME_OBSERVED,
        "NAX matrix unit (M5/Apple10-class): macOS 26.2+ AND arch_gen ≥ (arch=='p'?18:17)"),
)


#: NAX requires at least this macOS (MLX ``device.cpp`` ``__builtin_available``).
NAX_MIN_MACOS: tuple[int, int] = (26, 2)


def nax_available(
    macos_version: tuple[int, int], arch_char: str, arch_gen: int,
) -> bool:
    """Whether the NAX matrix unit is usable — a RUNTIME-OBSERVED probe grounded
    from MLX ``device.cpp:899-916``: ``macOS ≥ 26.2`` AND
    ``arch_gen ≥ (arch=='p' ? 18 : 17)`` (``arch_char`` = last char of the device
    architecture string). The M5/Apple10-class gate the steel-attention NAX path
    uses (``apple_sdpa_schedules``)."""
    if macos_version < NAX_MIN_MACOS:
        return False
    return arch_gen >= (18 if arch_char == "p" else 17)


def apple_probes_by_kind(kind: AppleProbeKind) -> tuple[AppleFeatureProbe, ...]:
    return tuple(p for p in APPLE_FEATURE_PROBES if p.kind is kind)


__all__ = [
    "AppleGPUArch",
    "AppleGPUTargetProfile",
    "AppleRuntimeLimits",
    "AppleProbeKind",
    "AppleFeatureProbe",
    "APPLE_FEATURE_PROBES",
    "NAX_MIN_MACOS",
    "nax_available",
    "apple_probes_by_kind",
    "TesseraAppleGPUTargetError",
    "TESSERA_TARGET_METAL",
    "TESSERA_TARGET_MACOS_FOR_MTL4",
    "apple_arch_defaults",
    "apple_arch_string",
    "apple_feature_set",
    "apple_feature_status",
    "apple_threadgroup_tiled_softmax_n_cap",
    "apple_fused_chain_score_cap",
    "apple_flash_attn_head_dim_cap",
    "apple_threadgroup_threads_per_row",
    "apple_supports_native_bf16",
    "probe_apple_runtime_limits",
]
