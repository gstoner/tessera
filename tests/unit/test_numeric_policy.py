"""Sprint C2 — NumericPolicy as a first-class registry attribute.

Locks the public contract that `docs/reference/tessera_tensor_attributes.md`
names: the sixth tensor attribute is `numeric_policy(storage, accum,
rounding, scale, quant_axis, deterministic, math_mode)`, separate from
the storage `dtype` axis.

These guard tests cover:

  1. `NumericPolicy` dataclass shape + alias normalization at
     construction time.
  2. The `_NUMERIC_POLICY_BY_NAME_FACTORIES` table covers the families
     the doc enumerates (matmul / attention / spectral / normalization /
     stable-reduction / quantization).
  3. The registry attaches `metadata.numeric_policy` to every promoted op.
  4. The dtype audit walker classifies every storage/accum slot as
     canonical (no aliases / no unknowns / no un-annotated planned-gated).
  5. TF32 stays out of `storage` (it routes through `math_mode` instead).
"""

from __future__ import annotations

import pytest

from tessera.compiler.primitive_coverage import (
    NumericPolicy,
    _NUMERIC_POLICY_BY_NAME_FACTORIES,
    _policy_for_name,
    all_primitive_coverages,
    audit_canonical_dtypes,
    assert_canonical_dtypes,
    coverage_for,
)
from tessera.dtype import TesseraDtypeError


# ──────────────────────────────────────────────────────────────────────────
#                       NumericPolicy dataclass
# ──────────────────────────────────────────────────────────────────────────

class TestNumericPolicy:
    def test_default_construction_storage_only(self):
        np_ = NumericPolicy(storage="fp32")
        assert np_.storage == "fp32"
        assert np_.accum is None
        assert np_.rounding == "round_to_nearest_even"
        assert np_.scale is None
        assert np_.quant_axis is None
        assert np_.deterministic is False
        assert np_.math_mode is None

    def test_storage_alias_normalized_at_construction(self):
        np_ = NumericPolicy(storage="f32")
        assert np_.storage == "fp32"

    def test_accum_alias_normalized_at_construction(self):
        np_ = NumericPolicy(storage="bf16", accum="f32")
        assert np_.accum == "fp32"

    def test_unknown_storage_rejected(self):
        with pytest.raises(TesseraDtypeError):
            NumericPolicy(storage="frobnicate")

    def test_unknown_accum_rejected(self):
        with pytest.raises(TesseraDtypeError):
            NumericPolicy(storage="fp32", accum="frobnicate")

    def test_tf32_rejected_in_storage(self):
        """TF32 is not a storage dtype.  Use math_mode='tf32' instead."""
        with pytest.raises(TesseraDtypeError, match="math_mode"):
            NumericPolicy(storage="tf32")

    def test_math_mode_accepts_tf32(self):
        np_ = NumericPolicy(storage="fp32", math_mode="tf32")
        assert np_.storage == "fp32"
        assert np_.math_mode == "tf32"

    def test_planned_gated_storage_accepted(self):
        """Allowed via allow_planned_gated=True at construction."""
        np_ = NumericPolicy(storage="nvfp4", accum="fp32", scale="blockfp_per_stage")
        assert np_.storage == "nvfp4"
        assert np_.scale == "blockfp_per_stage"

    def test_as_metadata_dict_round_trip(self):
        np_ = NumericPolicy(
            storage="bf16", accum="fp32",
            scale="per_channel_symmetric", quant_axis=1,
            deterministic=True,
        )
        md = np_.as_metadata_dict()
        assert md["storage"] == "bf16"
        assert md["accum"] == "fp32"
        assert md["scale"] == "per_channel_symmetric"
        assert md["quant_axis"] == 1
        assert md["deterministic"] is True
        assert md["math_mode"] is None
        assert md["rounding"] == "round_to_nearest_even"

    def test_frozen_dataclass(self):
        np_ = NumericPolicy(storage="fp32")
        with pytest.raises((AttributeError, Exception)):
            np_.storage = "fp16"  # type: ignore


# ──────────────────────────────────────────────────────────────────────────
#               Per-family policy table coverage
# ──────────────────────────────────────────────────────────────────────────

# The doc's "promoted ops" set: every entry in the named families
# *should* carry a numeric_policy.  This sanity-checks that the table
# didn't accidentally lose names during refactoring.

MATMUL_FAMILY = (
    "matmul", "gemm", "batched_gemm", "einsum", "factorized_matmul",
    "linear_general", "conv1d", "conv2d", "conv3d", "conv_transpose",
    "depthwise_conv1d", "depthwise_conv2d", "qkv_projection",
    "fused_epilogue",
)

ATTENTION_FAMILY = (
    "flash_attn", "multi_head_attention", "gqa_attention", "mqa_attention",
    "mla_decode", "mla_decode_fused", "linear_attn", "lightning_attention",
    "deepseek_sparse_attention", "gated_attention", "hybrid_attention",
    "gated_deltanet", "kimi_delta_attention", "modified_delta_attention",
    "attn_sliding_window", "attn_compressed_blocks", "attn_top_k_blocks",
)

SPECTRAL_FAMILY = (
    "fft", "ifft", "rfft", "irfft", "stft", "istft", "dct",
    "spectral_conv", "spectral_filter",
)

NORMALIZATION_FAMILY = (
    "layer_norm", "rmsnorm", "rmsnorm_safe", "weight_norm", "spectral_norm",
)

STABLE_REDUCTION_FAMILY = (
    "softmax", "softmax_safe", "online_softmax", "online_softmax_state",
)

QUANT_FAMILY = (
    "quantize_int8", "dequantize_int8",
    "quantize_int4", "dequantize_int4",
    "quantize_fp8", "dequantize_fp8",
    "quantize_fp4", "dequantize_fp4",
    "fake_quantize", "calibration_observer",
)


class TestPolicyTableCoverage:
    @pytest.mark.parametrize("name", MATMUL_FAMILY)
    def test_matmul_family_has_policy(self, name):
        assert name in _NUMERIC_POLICY_BY_NAME_FACTORIES, f"{name} missing policy"
        np_ = _policy_for_name(name)
        assert np_ is not None
        assert np_.storage == "bf16"
        assert np_.accum == "fp32"

    @pytest.mark.parametrize("name", ATTENTION_FAMILY)
    def test_attention_family_has_policy(self, name):
        np_ = _policy_for_name(name)
        assert np_ is not None
        assert np_.storage == "bf16"
        assert np_.accum == "fp32"
        assert np_.deterministic is True

    @pytest.mark.parametrize("name", SPECTRAL_FAMILY)
    def test_spectral_family_has_policy(self, name):
        np_ = _policy_for_name(name)
        assert np_ is not None
        assert np_.storage == "fp32"
        assert np_.accum == "fp32"

    @pytest.mark.parametrize("name", NORMALIZATION_FAMILY)
    def test_normalization_family_has_policy(self, name):
        np_ = _policy_for_name(name)
        assert np_ is not None
        assert np_.storage == "bf16"
        assert np_.accum == "fp32"

    @pytest.mark.parametrize("name", STABLE_REDUCTION_FAMILY)
    def test_stable_reduction_family_has_policy(self, name):
        np_ = _policy_for_name(name)
        assert np_ is not None
        assert np_.storage == "fp32"
        assert np_.accum == "fp32"
        assert np_.deterministic is True

    @pytest.mark.parametrize("name", QUANT_FAMILY)
    def test_quant_family_has_policy(self, name):
        np_ = _policy_for_name(name)
        assert np_ is not None
        # All quantization ops need a scale + accum; storage varies.
        assert np_.scale is not None, f"{name} missing scale"
        assert np_.accum == "fp32", f"{name} accum should be fp32"


# ──────────────────────────────────────────────────────────────────────────
#               Registry metadata + audit walker
# ──────────────────────────────────────────────────────────────────────────

class TestRegistryMetadata:
    def test_metadata_carries_numeric_policy_dict(self):
        e = coverage_for("matmul")
        assert "numeric_policy" in e.metadata
        np_meta = e.metadata["numeric_policy"]
        assert np_meta["storage"] == "bf16"
        assert np_meta["accum"] == "fp32"

    def test_metadata_quant_carries_scale_and_axis(self):
        e = coverage_for("quantize_int8")
        np_meta = e.metadata["numeric_policy"]
        assert np_meta["storage"] == "int8"
        assert np_meta["scale"] == "per_tensor_symmetric"

    def test_metadata_calibration_observer_deterministic(self):
        e = coverage_for("calibration_observer")
        np_meta = e.metadata["numeric_policy"]
        assert np_meta["deterministic"] is True
        assert np_meta["storage"] == "fp32"

    def test_metadata_spectral_storage_is_fp32(self):
        for name in ("fft", "ifft", "rfft", "irfft", "stft", "istft", "dct"):
            e = coverage_for(name)
            assert "numeric_policy" in e.metadata, f"{name} missing"
            assert e.metadata["numeric_policy"]["storage"] == "fp32"

    def test_promoted_count_meets_target(self):
        """Sprint C2 plan called for ~40 promoted ops; the table currently
        covers ≥40 (matmul-family, attention-family, spectral-family,
        normalization, stable-reduction, quantization)."""
        reg = all_primitive_coverages()
        with_policy = [
            n for n, e in reg.items()
            if "numeric_policy" in (e.metadata or {})
        ]
        assert len(with_policy) >= 40, (
            f"only {len(with_policy)} ops have numeric_policy; expected ≥40"
        )


class TestRegistryDtypeAuditPicksUpNumericPolicy:
    def test_audit_walker_finds_numeric_policy_storage_slots(self):
        buckets = audit_canonical_dtypes()
        # Sprint C2 should bump the canonical bucket — every numeric_policy
        # contributes at least one storage + one accum slot.
        assert len(buckets["canonical"]) >= 80, (
            f"expected ≥80 canonical dtype slots after Sprint C2, "
            f"got {len(buckets['canonical'])}"
        )

    def test_audit_walker_zero_unknown_after_c2(self):
        buckets = audit_canonical_dtypes()
        assert buckets["unknown"] == [], (
            f"unknown dtype slots after Sprint C2: {buckets['unknown']}"
        )

    def test_audit_walker_zero_aliases_after_c2(self):
        """All numeric_policy entries should construct with canonical
        spellings (the NumericPolicy dataclass normalizes at __post_init__)."""
        buckets = audit_canonical_dtypes()
        assert buckets["alias"] == [], (
            f"alias dtype slots after Sprint C2: {buckets['alias']}"
        )

    def test_assert_canonical_dtypes_still_passes(self):
        assert_canonical_dtypes()

    def test_numeric_policy_walker_keys_exposed(self):
        """The walker reports `numeric_policy.storage` / `accum` as the
        metadata key (so violators can be tracked to their slot)."""
        buckets = audit_canonical_dtypes()
        all_keys = {k for _, k, _ in buckets["canonical"]}
        # We should see the namespaced numeric_policy.* keys.
        assert any(k.startswith("numeric_policy.") for k in all_keys), (
            f"walker didn't surface numeric_policy.* keys; saw: {sorted(all_keys)}"
        )
        # And storage / accum specifically.
        assert "numeric_policy.storage" in all_keys
        assert "numeric_policy.accum" in all_keys
