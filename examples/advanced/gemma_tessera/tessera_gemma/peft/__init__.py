"""
tessera_gemma/peft — Parameter-Efficient Fine-Tuning adapters.

Exports:
    LoRAAdapter           — single low-rank adapter module
    LoRALinear            — nn.Linear wrapper with multi-adapter support
    apply_lora            — apply LoRA to model layers by substring pattern
    apply_lora_regex      — apply LoRA by full-regex rule set
    lora_state_dict       — extract only LoRA weights for checkpointing
    load_lora_state_dict  — restore LoRA weights into a model
    merge_lora / unmerge_lora — bake adapters into base weights
    QLinearSim            — Int4 quantisation simulation
    apply_qlora_sim       — wrap target layers in QLinearSim
    freeze_by_regex       — freeze parameters by name pattern
    param_groups_with_adapter_lrmult — per-adapter LR multipliers
"""

from .lora import (
    LoRAAdapter,
    LoRALinear,
    apply_lora,
    apply_lora_regex,
    lora_state_dict,
    load_lora_state_dict,
    merge_lora,
    unmerge_lora,
    QLinearSim,
    apply_qlora_sim,
    freeze_by_regex,
    param_groups_with_adapter_lrmult,
)

__all__ = [
    "LoRAAdapter",
    "LoRALinear",
    "apply_lora",
    "apply_lora_regex",
    "lora_state_dict",
    "load_lora_state_dict",
    "merge_lora",
    "unmerge_lora",
    "QLinearSim",
    "apply_qlora_sim",
    "freeze_by_regex",
    "param_groups_with_adapter_lrmult",
]
