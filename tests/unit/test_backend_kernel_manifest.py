"""Sprint E — Backend kernel manifest tests.

Locks the per-op × per-target × per-dtype matrix synthesized from
``capabilities.TARGET_CAPABILITIES`` + Apple GPU kernel inventory +
x86 AMX backend.

What's guarded:

  1. ``BackendKernelEntry`` dataclass shape + dtype normalization at
     construction (aliases like ``"f32"`` route to ``"fp32"``).
  2. Per-op manifest content matches what each backend actually ships:
       - Apple GPU has fused MSL kernels for matmul / softmax / gelu /
         rope / flash_attn / rmsnorm.
       - Apple CPU has Accelerate (cblas_sgemm + BNNS bf16/fp16) for
         matmul/gemm; numpy reference for the rest.
       - x86 AMX ships bf16 GEMM.
       - NVIDIA / ROCm / Metalium ship Target IR artifacts (gated on
         Phase G/H/I execution).
  3. Audit walker classifies every dtype as canonical; no aliases /
     unknowns leak into the manifest.
  4. Registry attaches ``metadata["backend_kernel_manifest"]`` per op.
  5. Per-target dtype dedup: if a capability lists both ``"fp32"`` and
     ``"f32"``, the entry stores only the canonical form once.
"""

from __future__ import annotations

import pytest

from tessera.compiler.backend_manifest import (
    BackendKernelEntry,
    all_manifests,
    audit_backend_dtypes,
    manifest_for,
    manifest_summary,
)
from tessera.compiler.primitive_coverage import coverage_for
from tessera.dtype import TesseraDtypeError


# ──────────────────────────────────────────────────────────────────────────
#                        BackendKernelEntry dataclass
# ──────────────────────────────────────────────────────────────────────────

class TestBackendKernelEntry:
    def test_construction_with_canonical_dtypes(self):
        entry = BackendKernelEntry(
            target="apple_gpu",
            status="fused",
            dtypes=("fp32", "fp16"),
        )
        assert entry.target == "apple_gpu"
        assert entry.status == "fused"
        assert entry.dtypes == ("fp32", "fp16")

    def test_alias_dtypes_normalized(self):
        entry = BackendKernelEntry(
            target="cpu",
            status="reference",
            dtypes=("f32", "bfloat16", "half"),
        )
        assert entry.dtypes == ("fp32", "bf16", "fp16")

    def test_duplicate_dtypes_deduped(self):
        """Both `'fp32'` and `'f32'` resolve to canonical `'fp32'`; the
        entry stores it once."""
        entry = BackendKernelEntry(
            target="apple_cpu", status="fused", dtypes=("fp32", "f32", "fp32"),
        )
        assert entry.dtypes == ("fp32",)

    def test_invalid_status_rejected(self):
        with pytest.raises(ValueError, match="status must be one of"):
            BackendKernelEntry(target="cpu", status="bogus")

    def test_tf32_in_dtypes_rejected(self):
        with pytest.raises(TesseraDtypeError, match="math_mode"):
            BackendKernelEntry(target="cpu", status="reference", dtypes=("tf32",))

    def test_as_dict_serializable(self):
        entry = BackendKernelEntry(
            target="apple_gpu",
            status="fused",
            dtypes=("fp32", "fp16", "bf16"),
            feature_flags=("metal", "msl"),
            notes="Custom MSL kernel",
        )
        d = entry.as_dict()
        assert d == {
            "target": "apple_gpu",
            "status": "fused",
            "dtypes": ["fp32", "fp16", "bf16"],
            "feature_flags": ["metal", "msl"],
            "notes": "Custom MSL kernel",
        }


# ──────────────────────────────────────────────────────────────────────────
#                  Per-op manifest content
# ──────────────────────────────────────────────────────────────────────────

class TestMatmulManifest:
    """matmul is the highest-coverage op; touches every backend."""

    def test_x86_ships_fused_amx_bf16(self):
        entries = {e.target: e for e in manifest_for("matmul")}
        assert "x86" in entries
        x = entries["x86"]
        assert x.status == "fused"
        assert "bf16" in x.dtypes
        assert "amx" in x.feature_flags

    def test_apple_cpu_ships_fused_accelerate(self):
        entries = {e.target: e for e in manifest_for("matmul")}
        assert "apple_cpu" in entries
        ac = entries["apple_cpu"]
        assert ac.status == "fused"
        for dt in ("fp32", "fp16", "bf16"):
            assert dt in ac.dtypes

    def test_apple_gpu_ships_fused_mps(self):
        entries = {e.target: e for e in manifest_for("matmul")}
        assert "apple_gpu" in entries
        ag = entries["apple_gpu"]
        assert ag.status == "fused"
        for dt in ("fp32", "fp16", "bf16"):
            assert dt in ag.dtypes
        assert "metal" in ag.feature_flags

    def test_nvidia_sm90_has_artifact(self):
        entries = {e.target: e for e in manifest_for("matmul")}
        assert "nvidia_sm90" in entries
        assert entries["nvidia_sm90"].status == "artifact_only"
        assert "wgmma" in entries["nvidia_sm90"].feature_flags

    def test_cpu_reference_path_present(self):
        entries = {e.target: e for e in manifest_for("matmul")}
        assert "cpu" in entries
        assert entries["cpu"].status == "reference"


class TestAppleGPUMSLKernels:
    """The Apple GPU MSL kernel inventory ships fused kernels for a
    specific list; sprint E records every one."""

    @pytest.mark.parametrize("name", [
        "matmul", "softmax", "softmax_safe", "gelu", "rope",
        "flash_attn", "rmsnorm",
    ])
    def test_apple_gpu_kernel_is_fused(self, name):
        entries = {e.target: e for e in manifest_for(name)}
        assert "apple_gpu" in entries, f"{name} missing from apple_gpu manifest"
        assert entries["apple_gpu"].status == "fused"


class TestFlashAttnManifest:
    def test_apple_gpu_supports_three_dtypes(self):
        entries = {e.target: e for e in manifest_for("flash_attn")}
        ag = entries["apple_gpu"]
        assert ag.status == "fused"
        assert set(ag.dtypes) >= {"fp32", "fp16", "bf16"}

    def test_no_x86_fused_attention(self):
        """x86 doesn't ship a fused attention kernel (only AMX GEMM)."""
        entries = {e.target: e for e in manifest_for("flash_attn")}
        assert "x86" not in entries


# ──────────────────────────────────────────────────────────────────────────
#                        Aggregate manifest
# ──────────────────────────────────────────────────────────────────────────

class TestAggregateManifest:
    def test_all_manifests_nonempty_for_op_specs_subset(self):
        m = all_manifests()
        # Every common op in OP_SPECS should have at least one backend.
        for name in ("matmul", "softmax", "gelu", "layer_norm", "flash_attn"):
            assert name in m, f"{name} missing from aggregate manifest"
            assert m[name], f"{name} has empty manifest"

    def test_manifest_summary_records_targets(self):
        summary = manifest_summary()
        # Every backend we ship should appear.
        for target in ("x86", "apple_cpu", "apple_gpu", "cpu",
                       "nvidia_sm80", "nvidia_sm90", "nvidia_sm100",
                       "nvidia_sm120", "rocm", "metalium"):
            assert target in summary, f"{target} missing from summary"

    def test_apple_gpu_has_fused_kernels(self):
        summary = manifest_summary()
        ag = summary["apple_gpu"]
        # Sprint E covers ≥6 fused Apple GPU kernels (matmul, softmax,
        # softmax_safe, gelu, rope, flash_attn, rmsnorm).
        assert ag.get("fused", 0) >= 6, f"apple_gpu fused count: {ag}"

    def test_nvidia_sm90_all_artifact_only(self):
        summary = manifest_summary()
        sm90 = summary["nvidia_sm90"]
        # SM90 ships Target IR artifacts; execution gated on Phase G.
        assert "artifact_only" in sm90
        assert sm90.get("fused", 0) == 0  # no fused exec yet

    def test_cpu_reference_count_large(self):
        summary = manifest_summary()
        # CPU reference path covers most of OP_SPECS.
        assert summary["cpu"].get("reference", 0) >= 200


# ──────────────────────────────────────────────────────────────────────────
#                  Backend dtype audit walker
# ──────────────────────────────────────────────────────────────────────────

class TestBackendDtypeAudit:
    def test_zero_unknown_dtypes(self):
        buckets = audit_backend_dtypes()
        assert buckets["unknown"] == [], (
            f"unknown dtype strings in manifest: {buckets['unknown'][:5]}"
        )

    def test_zero_aliases(self):
        """All backend manifest entries should store canonical spellings."""
        buckets = audit_backend_dtypes()
        assert buckets["alias"] == [], (
            f"alias dtype strings in manifest: {buckets['alias'][:5]}"
        )

    def test_planned_gated_only_via_explicit_metalium_blockfp(self):
        """Only the explicit Tenstorrent `metalium_blockfp` target may
        carry planned/gated dtypes (`bfp8`/`bfp4`) — Sprint I-2 added
        those deliberately to surface the block-FP family without
        promoting them to the general dtype matrix.  All other targets
        must stay 0 planned-gated."""
        buckets = audit_backend_dtypes()
        for op_name, key, dt in buckets["planned_gated"]:
            assert "metalium_blockfp" in key, (
                f"planned-gated dtype {dt!r} on {key!r} should only appear "
                f"via the metalium_blockfp target"
            )
        # Both bfp8 and bfp4 must be present (Sprint I-2 ships them).
        gated_dtypes = {dt for _, _, dt in buckets["planned_gated"]}
        assert "bfp8" in gated_dtypes
        assert "bfp4" in gated_dtypes

    def test_canonical_bucket_substantial(self):
        buckets = audit_backend_dtypes()
        # Per-target × per-op × per-dtype = hundreds of slots.
        assert len(buckets["canonical"]) >= 500


# ──────────────────────────────────────────────────────────────────────────
#                  Registry attaches manifest as metadata
# ──────────────────────────────────────────────────────────────────────────

class TestRegistryManifestAttached:
    def test_matmul_has_backend_kernel_manifest_metadata(self):
        e = coverage_for("matmul")
        assert "backend_kernel_manifest" in e.metadata
        m = e.metadata["backend_kernel_manifest"]
        assert isinstance(m, list)
        assert len(m) >= 5  # several backends

    def test_manifest_entries_are_plain_dicts(self):
        """The metadata stores plain dicts (JSON-friendly) — not the
        frozen-dataclass instances."""
        e = coverage_for("flash_attn")
        m = e.metadata["backend_kernel_manifest"]
        for entry in m:
            assert isinstance(entry, dict)
            assert set(entry) >= {"target", "status", "dtypes",
                                  "feature_flags", "notes"}

    def test_manifest_dtypes_canonical_in_metadata(self):
        """No alias leakage from the metadata path."""
        e = coverage_for("matmul")
        m = e.metadata["backend_kernel_manifest"]
        for entry in m:
            for dt in entry["dtypes"]:
                # Canonical names don't contain the "f32"/"f16"/"i*" alias
                # patterns the doc rejects at storage.
                assert dt not in {"f32", "f64", "f16", "i8", "i16", "i32", "i64", "tf32"}

    def test_ops_without_backend_skip_manifest_field(self):
        """Python-runtime primitives (pytree, autodiff transforms) don't
        carry a backend manifest — their entries should be absent from
        ``all_manifests``."""
        m = all_manifests()
        for name in ("tree_flatten", "vjp", "vmap", "shard_map"):
            assert name not in m, f"{name} unexpectedly has a backend manifest"
