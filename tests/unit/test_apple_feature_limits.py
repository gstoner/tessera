"""Audit Action 10 — AppleFeatureLimits / AppleGPUTargetProfile.

Per-arch capability table for Apple Silicon GPUs (M1 / M2 / M3 / M4 /
M5). Mirrors the structure already shipped for NVIDIA (``gpu_target.py``)
and AMD (``rocm_target.py``).

These tests pin:

* **Feature matrix completeness** — every arch declares a status for
  every feature name (no typo'd / missing rows).
* **Generation gating** — features that didn't exist on older arches
  are properly marked ``not_supported`` (bfloat pre-M2, MTL4 pre-M4,
  neural accelerators pre-M4, ray tracing pre-M3, dynamic caching
  pre-M3, mesh shaders pre-M3).
* **dtype set** — pre-M2 lacks ``bf16``; M2+ has it. No Apple arch
  ships native fp64.
* **AppleGPUTargetProfile defaults + invariants** — M4 default, valid
  arch coercion, packaged-ML auto-off on pre-M4, threadgroup-memory
  override validation.
* **Static limits floor** — simdgroup is 32 across all arches;
  max-threads-per-threadgroup is 1024.
* **Helper functions** — ``apple_feature_status`` raises on typos;
  ``apple_feature_set`` returns only ``"ready"`` features.
* **Runtime probe** — degrades gracefully when the optional new
  C ABI symbols aren't present.
"""

from __future__ import annotations

import pytest

from tessera.compiler.apple_target import (
    AppleGPUArch,
    AppleGPUTargetProfile,
    AppleRuntimeLimits,
    TESSERA_TARGET_MACOS_FOR_MTL4,
    TESSERA_TARGET_METAL,
    TesseraAppleGPUTargetError,
    apple_arch_defaults,
    apple_arch_string,
    apple_feature_set,
    apple_feature_status,
    probe_apple_runtime_limits,
)


ALL_ARCHES = (
    AppleGPUArch.APPLE7,
    AppleGPUArch.APPLE8,
    AppleGPUArch.APPLE9,
    AppleGPUArch.APPLE10,
    AppleGPUArch.APPLE11,
)


# ---- Feature matrix completeness + consistency -------------------------

def test_every_arch_declares_status_for_every_feature():
    """Pick one canonical arch as the schema and verify every other
    arch declares the same feature set. Catches typos / silently
    dropped rows."""
    canonical = set(apple_feature_set(AppleGPUArch.APPLE10)) | {
        name for name in (
            "metal3", "metal4", "mpsgraph", "simdgroup",
            "simdgroup_async_copy", "simdgroup_matrix", "bfloat",
            "ray_tracing", "dynamic_caching", "mesh_shaders",
            "neural_accelerators", "mtl4_packaged_ml",
            "mtl4_compiler", "mtl4_command_queue",
            "function_pointers", "argument_buffers",
        )
    }
    # Every arch must answer apple_feature_status for every feature.
    for arch in ALL_ARCHES:
        for f in canonical:
            apple_feature_status(arch, f)  # raises KeyError if missing


def test_unknown_feature_name_raises_keyerror():
    """Typos should be loud."""
    with pytest.raises(KeyError):
        apple_feature_status(AppleGPUArch.APPLE10, "warp_specialization")


def test_status_values_only_use_known_strings():
    valid_statuses = {"ready", "tba", "not_supported"}
    for arch in ALL_ARCHES:
        # All feature names — iterate via the matrix from the module.
        from tessera.compiler.apple_target import _APPLE_FEATURES
        for fname, status in _APPLE_FEATURES[arch].items():
            assert status in valid_statuses, (
                f"arch={arch.name} feature={fname!r} status={status!r} "
                f"not in {sorted(valid_statuses)}")


# ---- Generation gating -------------------------------------------------

def test_bfloat_starts_at_m2():
    assert (apple_feature_status(AppleGPUArch.APPLE7, "bfloat")
            == "not_supported")
    for arch in (AppleGPUArch.APPLE8, AppleGPUArch.APPLE9,
                 AppleGPUArch.APPLE10, AppleGPUArch.APPLE11):
        assert apple_feature_status(arch, "bfloat") == "ready"


def test_ray_tracing_starts_at_m3():
    for arch in (AppleGPUArch.APPLE7, AppleGPUArch.APPLE8):
        assert (apple_feature_status(arch, "ray_tracing")
                == "not_supported")
    for arch in (AppleGPUArch.APPLE9, AppleGPUArch.APPLE10,
                 AppleGPUArch.APPLE11):
        assert apple_feature_status(arch, "ray_tracing") == "ready"


def test_dynamic_caching_starts_at_m3():
    for arch in (AppleGPUArch.APPLE7, AppleGPUArch.APPLE8):
        assert (apple_feature_status(arch, "dynamic_caching")
                == "not_supported")
    for arch in (AppleGPUArch.APPLE9, AppleGPUArch.APPLE10,
                 AppleGPUArch.APPLE11):
        assert (apple_feature_status(arch, "dynamic_caching")
                == "ready")


def test_metal4_starts_at_m4():
    for arch in (AppleGPUArch.APPLE7, AppleGPUArch.APPLE8,
                 AppleGPUArch.APPLE9):
        assert apple_feature_status(arch, "metal4") == "not_supported"
        assert (apple_feature_status(arch, "mtl4_packaged_ml")
                == "not_supported")
        assert (apple_feature_status(arch, "neural_accelerators")
                == "not_supported")
    for arch in (AppleGPUArch.APPLE10, AppleGPUArch.APPLE11):
        assert apple_feature_status(arch, "metal4") == "ready"
        assert (apple_feature_status(arch, "mtl4_packaged_ml")
                == "ready")
        assert (apple_feature_status(arch, "neural_accelerators")
                == "ready")


def test_metal3_always_ready():
    """Every Apple GPU since M1 supports the Metal 3 surface."""
    for arch in ALL_ARCHES:
        assert apple_feature_status(arch, "metal3") == "ready"


def test_simdgroup_async_copy_always_ready():
    """simdgroup_async_copy is the Apple equivalent of NVIDIA's
    cp.async — available on every Apple GPU we support."""
    for arch in ALL_ARCHES:
        assert (apple_feature_status(arch, "simdgroup_async_copy")
                == "ready")


# ---- dtype matrix ------------------------------------------------------

def test_dtype_set_bfloat_gating():
    # M1 lacks bf16.
    p_m1 = AppleGPUTargetProfile(arch=AppleGPUArch.APPLE7)
    assert "bf16" not in p_m1.dtype_set
    assert "fp16" in p_m1.dtype_set
    assert "fp32" in p_m1.dtype_set
    # M2+ has bf16.
    for arch in (AppleGPUArch.APPLE8, AppleGPUArch.APPLE9,
                 AppleGPUArch.APPLE10, AppleGPUArch.APPLE11):
        p = AppleGPUTargetProfile(arch=arch)
        assert "bf16" in p.dtype_set, arch.name


def test_no_apple_arch_ships_native_fp64():
    """fp64 is structurally absent on Apple GPUs as of M5; ops that
    require fp64 must fall back to CPU per Decision #19."""
    for arch in ALL_ARCHES:
        p = AppleGPUTargetProfile(arch=arch)
        assert "fp64" not in p.dtype_set, (
            f"{arch.name} unexpectedly claims native fp64")


# ---- AppleGPUTargetProfile construction --------------------------------

def test_profile_default_arch_is_m4():
    """Default arch is APPLE10 (M4) — the first generation where
    Metal 4 + packaged ML are fully reachable."""
    p = AppleGPUTargetProfile()
    assert p.arch == AppleGPUArch.APPLE10


def test_profile_default_has_packaged_ml_enabled():
    p = AppleGPUTargetProfile()
    assert p.prefer_packaged_ml is True


def test_profile_coerces_int_arch():
    p = AppleGPUTargetProfile(arch=10)  # type: ignore[arg-type]
    assert p.arch == AppleGPUArch.APPLE10


def test_profile_rejects_unknown_arch():
    with pytest.raises(TesseraAppleGPUTargetError, match="Unknown"):
        AppleGPUTargetProfile(arch=99)  # type: ignore[arg-type]


def test_profile_force_off_packaged_ml_on_pre_m4():
    """Asking for packaged ML on a pre-M4 arch silently force-disables
    rather than allowing a misleading flag to flow into the lowering."""
    for arch in (AppleGPUArch.APPLE7, AppleGPUArch.APPLE8,
                 AppleGPUArch.APPLE9):
        p = AppleGPUTargetProfile(arch=arch, prefer_packaged_ml=True)
        assert p.prefer_packaged_ml is False, arch.name


def test_profile_rejects_invalid_threadgroup_memory():
    with pytest.raises(TesseraAppleGPUTargetError, match="must be positive"):
        AppleGPUTargetProfile(threadgroup_memory_bytes=0)
    with pytest.raises(TesseraAppleGPUTargetError, match="must be positive"):
        AppleGPUTargetProfile(threadgroup_memory_bytes=-1)


# ---- Capability predicates ---------------------------------------------

def test_capability_predicates_track_feature_matrix():
    p_m1 = AppleGPUTargetProfile(arch=AppleGPUArch.APPLE7)
    assert p_m1.supports_metal4 is False
    assert p_m1.supports_packaged_ml is False
    assert p_m1.supports_bfloat is False
    assert p_m1.supports_neural_accelerators is False
    assert p_m1.supports_ray_tracing is False

    p_m3 = AppleGPUTargetProfile(arch=AppleGPUArch.APPLE9)
    assert p_m3.supports_metal4 is False
    assert p_m3.supports_bfloat is True
    assert p_m3.supports_ray_tracing is True
    assert p_m3.supports_dynamic_caching is True
    assert p_m3.supports_neural_accelerators is False

    p_m4 = AppleGPUTargetProfile(arch=AppleGPUArch.APPLE10)
    assert p_m4.supports_metal4 is True
    assert p_m4.supports_packaged_ml is True
    assert p_m4.supports_bfloat is True
    assert p_m4.supports_neural_accelerators is True
    assert p_m4.supports_simdgroup_matrix is True

    p_m5 = AppleGPUTargetProfile(arch=AppleGPUArch.APPLE11)
    assert p_m5.supports_metal4 is True
    assert p_m5.supports_packaged_ml is True
    assert p_m5.supports_neural_accelerators is True


# ---- Static limits floor -----------------------------------------------

def test_simdgroup_size_is_32_across_all_arches():
    """Apple GPU SIMD width is 32, by spec, across every generation
    we support."""
    for arch in ALL_ARCHES:
        p = AppleGPUTargetProfile(arch=arch)
        assert p.simdgroup_size == 32, arch.name


def test_max_threads_per_threadgroup_is_1024():
    for arch in ALL_ARCHES:
        p = AppleGPUTargetProfile(arch=arch)
        assert p.max_threads_per_threadgroup == 1024, arch.name


def test_threadgroup_memory_capacity_uses_static_floor_when_no_override():
    p = AppleGPUTargetProfile()
    assert p.threadgroup_memory_capacity_bytes == 32 * 1024


def test_threadgroup_memory_capacity_honors_override():
    p = AppleGPUTargetProfile(threadgroup_memory_bytes=65536)
    assert p.threadgroup_memory_capacity_bytes == 65536


# ---- Helper functions --------------------------------------------------

def test_apple_feature_set_includes_only_ready_features():
    s = apple_feature_set(AppleGPUArch.APPLE10)
    assert "metal4" in s
    assert "bfloat" in s
    assert "neural_accelerators" in s
    s_m1 = apple_feature_set(AppleGPUArch.APPLE7)
    assert "metal4" not in s_m1
    assert "bfloat" not in s_m1
    assert "neural_accelerators" not in s_m1


def test_apple_arch_string_returns_apple_n():
    assert apple_arch_string(AppleGPUArch.APPLE7) == "apple7"
    assert apple_arch_string(AppleGPUArch.APPLE10) == "apple10"
    assert apple_arch_string(AppleGPUArch.APPLE11) == "apple11"


def test_apple_arch_defaults_returns_arch_floor():
    d = apple_arch_defaults(AppleGPUArch.APPLE10)
    assert d.simdgroup_size == 32
    assert d.max_threads_per_threadgroup == 1024
    assert d.threadgroup_memory_bytes == 32 * 1024


def test_target_metal_version_pin_is_4_0():
    """Apple GPU backend's compile target is MSL 4.0 / macOS 26."""
    assert TESSERA_TARGET_METAL == "4.0"
    assert TESSERA_TARGET_MACOS_FOR_MTL4 == "26.0"


# ---- Runtime probe (graceful degradation) ------------------------------

def test_runtime_probe_returns_limits_or_none():
    """The probe should never raise. It either returns a fully-populated
    ``AppleRuntimeLimits`` (when the runtime + symbols load), or
    ``None`` on a host where the dylib isn't buildable. Optional newer
    symbols falling back to sentinels is part of the contract."""
    result = probe_apple_runtime_limits()
    if result is None:
        # Off-Darwin / no runtime — that's a valid skip path.
        return
    assert isinstance(result, AppleRuntimeLimits)
    # max_threadgroup_memory_bytes is a non-negative int; 0 = "probe
    # symbol not yet shipped — consult static floor."
    assert result.max_threadgroup_memory_bytes >= 0
    # The probe's metal4 / packaged_ml flags are consistent —
    # packaged_ml is a strict subset of metal4 in the Apple stack.
    if result.supports_packaged_ml:
        assert result.supports_metal4
    # apple_gpu_family is an int (≥ 7 if known, -1 sentinel otherwise).
    assert isinstance(result.apple_gpu_family, int)


def test_runtime_probe_packaged_ml_agrees_with_existing_helper():
    """Cross-check: the new probe should agree with the existing
    ``packaged_ml_available()`` helper on this host."""
    from tessera.apple_mlpkg import packaged_ml_available
    runtime_result = probe_apple_runtime_limits()
    if runtime_result is None:
        return  # off-Darwin, both helpers degrade
    assert (runtime_result.supports_packaged_ml
            == packaged_ml_available())
