"""
tessera_diffusion_llm
=====================

Three diffusion language model variants with shared transformer backbone
and Tessera compiler integration.

Quick-start::

    from tessera_diffusion_llm import MDLM, MDLMConfig

    cfg   = MDLMConfig.debug_tiny()
    model = MDLM(cfg)
    loss  = model.compute_loss(input_ids)
    out   = model.generate(batch_size=2, seq_len=32)

Model variants:
  • MDLM                  — Masked Discrete Diffusion (Shi et al. 2024)
  • ContinuousDiffusionLLM — Gaussian diffusion in embedding space (Ho et al. 2020)
  • FlowMatchingLLM       — Rectified Flow (Liu et al. 2022)
"""

from .configs import (
    TransformerConfig,
    MDLMConfig,
    ContinuousDiffusionConfig,
    FlowMatchingConfig,
)
from .models import (
    DiffusionTransformer,
    DiffusionTransformerBlock,
    MDLM,
    ContinuousDiffusionLLM,
    FlowMatchingLLM,
)
from .schedules import (
    NoiseSchedule,
    MaskSchedule,
    cosine_beta_schedule,
    linear_beta_schedule,
    sqrt_beta_schedule,
    cosine_mask_schedule,
    linear_mask_schedule,
    ddpm_step,
    ddpm_sample,
    ddim_step,
    ddim_sample,
    ode_euler_step,
    flow_ode_sample,
    mdlm_step,
    mdlm_sample,
)
from .training import (
    DiffusionTrainer,
    TrainerConfig,
    mdlm_elbo_loss,
    continuous_diffusion_loss,
    flow_matching_loss,
)
from .inference import DiffusionGenerator, GeneratorConfig
from .utils import count_parameters, param_summary, tokens_to_human

__version__ = "0.1.0"

__all__ = [
    # Configs
    "TransformerConfig",
    "MDLMConfig",
    "ContinuousDiffusionConfig",
    "FlowMatchingConfig",
    # Models
    "DiffusionTransformer",
    "DiffusionTransformerBlock",
    "MDLM",
    "ContinuousDiffusionLLM",
    "FlowMatchingLLM",
    # Schedules
    "NoiseSchedule",
    "MaskSchedule",
    "cosine_beta_schedule",
    "linear_beta_schedule",
    "sqrt_beta_schedule",
    "cosine_mask_schedule",
    "linear_mask_schedule",
    # Samplers
    "ddpm_step",
    "ddpm_sample",
    "ddim_step",
    "ddim_sample",
    "ode_euler_step",
    "flow_ode_sample",
    "mdlm_step",
    "mdlm_sample",
    # Training
    "DiffusionTrainer",
    "TrainerConfig",
    "mdlm_elbo_loss",
    "continuous_diffusion_loss",
    "flow_matching_loss",
    # Inference
    "DiffusionGenerator",
    "GeneratorConfig",
    # Utils
    "count_parameters",
    "param_summary",
    "tokens_to_human",
]
