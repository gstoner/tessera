"""DiffusionGemma Phase F — quantization & vision staging (planning hooks)."""

from __future__ import annotations

import dataclasses

import pytest

from tessera.models.diffusion_gemma import (
    DiffusionGemmaConfig,
    DiffusionGemmaDimError,
    estimated_param_counts,
)
from tessera.models import staging as st


def test_bf16_is_the_executable_baseline():
    p = st.plan_quantization(DiffusionGemmaConfig(), "bf16")
    assert p.executable is True
    assert p.bytes_per_param == 2.0


@pytest.mark.parametrize("target", ["fp8_e4m3", "fp8_e5m2", "nvfp4", "q4_0", "int4"])
def test_low_precision_targets_are_planning_only(target):
    p = st.plan_quantization(DiffusionGemmaConfig(), target)
    assert p.executable is False
    assert "planning hook only" in p.notes


def test_bf16_footprint_matches_param_estimate():
    cfg = DiffusionGemmaConfig()
    p = st.plan_quantization(cfg, "bf16")
    assert p.total_bytes == int(estimated_param_counts(cfg)["total"]) * 2


def test_footprint_scales_with_precision():
    cfg = DiffusionGemmaConfig()
    bf16 = st.plan_quantization(cfg, "bf16").total_bytes
    fp8 = st.plan_quantization(cfg, "fp8_e4m3").total_bytes
    nvfp4 = st.plan_quantization(cfg, "nvfp4").total_bytes
    assert bf16 == 2 * fp8           # 16-bit vs 8-bit
    assert fp8 == 2 * nvfp4          # 8-bit vs 4-bit


def test_unknown_quant_target_rejected():
    with pytest.raises(st.StagingError, match="unknown quant target"):
        st.plan_quantization(DiffusionGemmaConfig(), "bogus")


# ── Vision staging ───────────────────────────────────────────────────────────

def test_vision_metadata_validates():
    cfg = DiffusionGemmaConfig()
    vm = st.default_vision_metadata(cfg)
    st.validate_vision_metadata(vm)   # no raise
    assert vm.encoder_params == cfg.vision_encoder_params
    assert vm.image_token_budgets == (70, 140, 280, 560, 1120)


def test_vision_execution_is_unsupported():
    assert st.vision_execution_supported() is False


def test_vision_metadata_rejects_bad_budgets():
    bad = st.VisionMetadata(encoder_params=550_000_000,
                            supported_modalities=("text", "image"),
                            image_token_budgets=(70, 140))
    with pytest.raises(st.StagingError, match="image_token_budgets"):
        st.validate_vision_metadata(bad)


# ── Importer ─────────────────────────────────────────────────────────────────

def test_importer_marks_vision_unsupported_but_text_executable():
    m = st.import_model_metadata(DiffusionGemmaConfig())
    assert m.text_executable is True and m.text_target == "bf16"
    assert m.vision_supported_for_execution is False
    assert m.vision.encoder_params == 550_000_000
    assert m.quant_plan.executable is True


def test_importer_validates_config_dims():
    # A GQA-invalid config must be rejected at import (before any staging).
    bad = dataclasses.replace(DiffusionGemmaConfig(), num_attention_heads=15, num_kv_heads=8)
    with pytest.raises(DiffusionGemmaDimError):
        st.import_model_metadata(bad)
