"""Gumiho — hybrid speculative decoding on the Tessera Apple backend.

Port of "Gumiho: A Hybrid Architecture to Prioritize Early Tokens in
Speculative Decoding" (ICML'25, arXiv:2503.10135) — a serial 2-layer
Transformer head for the accuracy-critical early tokens + parallel MLP heads
for the rest, with Full Tree Attention verification. The draft + verify dense
math runs on the Apple GPU/CPU compiler backend; acceptance + KV advance reuse
``tessera.speculative``.
"""

from .config import GumihoConfig, tiny_config
from .decode import (
    DecodeMetrics,
    GumihoSummary,
    PrecisionSummary,
    run_gumiho_demo,
    run_multistep_decode,
    run_precision_demo,
    run_training_demo,
)
from .onchip import OnchipStepSummary, run_onchip_step_demo
from .prefix_shared import (
    PrefixSharedVerifier,
    PrefixSharingSummary,
    run_prefix_sharing_demo,
)
from .resident import (
    ForLoopDraftResult,
    ResidentDraftResult,
    ResidentSerialDraft,
    serial_draft_forloop,
    validate_resident_draft,
    validate_serial_forloop,
)
from .training import distill, trajectory_contexts

__all__ = [
    "GumihoConfig",
    "tiny_config",
    "GumihoSummary",
    "run_gumiho_demo",
    "DecodeMetrics",
    "run_multistep_decode",
    "run_training_demo",
    "distill",
    "trajectory_contexts",
    "ResidentSerialDraft",
    "ResidentDraftResult",
    "validate_resident_draft",
    "OnchipStepSummary",
    "run_onchip_step_demo",
    "ForLoopDraftResult",
    "serial_draft_forloop",
    "validate_serial_forloop",
    "PrecisionSummary",
    "run_precision_demo",
    "PrefixSharedVerifier",
    "PrefixSharingSummary",
    "run_prefix_sharing_demo",
]
