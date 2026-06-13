"""DiffusionGemma Phase F — quantization & vision staging (planning hooks).

BF16 is the first execution baseline. This module adds:

  * FP8 / NVFP4 / Q4_0 quantization **planning hooks** — per-target weight-byte
    plans (which tensors, what scheme, the memory footprint) without executing
    any quantized kernel. Only BF16 is execution-ready; the low-precision targets
    are plans pending the BF16 graph + sampler correctness gate.
  * a vision-metadata validator — the importer validates the Gemma 4 26B A4B
    vision metadata (encoder params, supported image-token budgets) and marks
    vision **unsupported for execution** (text-only block diffusion first).

Nothing here claims quantized or vision *execution*; these are staging hooks +
metadata validation, consistent with the audit rule "update coverage only when
runtime proof is real."
"""

from __future__ import annotations

from dataclasses import dataclass

from .diffusion_gemma import DiffusionGemmaConfig, estimated_param_counts, verify_config


# Bytes-per-parameter by target (Q4_0 / NVFP4 ≈ 4-bit incl. small block scales).
_BYTES_PER_PARAM = {
    "bf16": 2.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "nvfp4": 0.5,
    "q4_0": 0.5,
    "int4": 0.5,
}
# Only BF16 has an execution path today; the rest are planning hooks.
EXECUTABLE_TARGETS = frozenset({"bf16"})

# Gemma 4 supported visual-token budgets (from the model card).
GEMMA4_IMAGE_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


class StagingError(ValueError):
    """Raised when a quantization target or vision metadata is invalid."""


@dataclass(frozen=True)
class QuantizationPlan:
    target: str
    bytes_per_param: float
    total_params: int
    total_bytes: int
    gb: float
    executable: bool
    notes: str


def plan_quantization(config: DiffusionGemmaConfig, target: str = "bf16") -> QuantizationPlan:
    """Plan (do not execute) the weight quantization footprint for ``target``.

    Returns the per-target memory footprint derived from the config's parameter
    count. ``executable`` is True only for BF16; low-precision targets are plans
    pending the BF16-correctness gate.
    """
    t = target.lower()
    if t not in _BYTES_PER_PARAM:
        raise StagingError(
            f"unknown quant target {target!r}; expected one of {sorted(_BYTES_PER_PARAM)}")
    verify_config(config)
    total = int(estimated_param_counts(config)["total"])
    bpp = _BYTES_PER_PARAM[t]
    total_bytes = int(total * bpp)
    executable = t in EXECUTABLE_TARGETS
    note = ("BF16 execution baseline" if executable else
            f"planning hook only — {t} execution gated on BF16 graph + sampler correctness")
    return QuantizationPlan(
        target=t, bytes_per_param=bpp, total_params=total, total_bytes=total_bytes,
        gb=round(total_bytes / 1e9, 1), executable=executable, notes=note)


@dataclass(frozen=True)
class VisionMetadata:
    encoder_params: int
    supported_modalities: tuple[str, ...]
    image_token_budgets: tuple[int, ...]


def default_vision_metadata(config: DiffusionGemmaConfig) -> VisionMetadata:
    return VisionMetadata(
        encoder_params=config.vision_encoder_params,
        supported_modalities=config.modalities,
        image_token_budgets=GEMMA4_IMAGE_TOKEN_BUDGETS,
    )


def validate_vision_metadata(meta: VisionMetadata) -> None:
    """Validate vision metadata shape/values (importer-side). Does NOT enable
    vision execution — see :func:`vision_execution_supported`."""
    if meta.encoder_params <= 0:
        raise StagingError("vision encoder_params must be positive")
    if "image" not in meta.supported_modalities:
        raise StagingError("vision metadata present but 'image' not in modalities")
    if tuple(meta.image_token_budgets) != GEMMA4_IMAGE_TOKEN_BUDGETS:
        raise StagingError(
            f"image_token_budgets {meta.image_token_budgets} != supported "
            f"{GEMMA4_IMAGE_TOKEN_BUDGETS}")


def vision_execution_supported() -> bool:
    """Vision execution is deferred — text-only block diffusion lands first."""
    return False


@dataclass(frozen=True)
class ModelManifest:
    config: DiffusionGemmaConfig
    text_executable: bool
    text_target: str
    vision_supported_for_execution: bool
    vision: VisionMetadata
    quant_plan: QuantizationPlan


def import_model_metadata(
    config: DiffusionGemmaConfig,
    *,
    target: str = "bf16",
    vision: VisionMetadata | None = None,
) -> ModelManifest:
    """Importer entry — validate the config dims + vision metadata and produce a
    manifest. Text is executable on BF16; vision is validated but marked
    unsupported for execution."""
    verify_config(config)
    vm = vision if vision is not None else default_vision_metadata(config)
    validate_vision_metadata(vm)
    plan = plan_quantization(config, target)
    return ModelManifest(
        config=config,
        text_executable=plan.executable,
        text_target=plan.target,
        vision_supported_for_execution=vision_execution_supported(),
        vision=vm,
        quant_plan=plan,
    )


__all__ = [
    "StagingError",
    "QuantizationPlan",
    "VisionMetadata",
    "ModelManifest",
    "EXECUTABLE_TARGETS",
    "GEMMA4_IMAGE_TOKEN_BUDGETS",
    "plan_quantization",
    "default_vision_metadata",
    "validate_vision_metadata",
    "vision_execution_supported",
    "import_model_metadata",
]
