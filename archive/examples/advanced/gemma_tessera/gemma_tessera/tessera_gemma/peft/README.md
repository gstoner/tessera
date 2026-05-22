
# Tessera-Gemma PEFT

This folder adds **LoRA** adapters for PyTorch `nn.Linear` layers, plus small helpers:

- `apply_lora(model, patterns=...)` — wrap selected Linear layers.
- `lora_state_dict(model)` / `load_lora_state_dict(model, state)` — save/load adapters.
- `merge_lora(model)` / `unmerge_lora(model)` — fuse/unfuse LoRA into base weights (for export/inference).

> Inspired by the Gemma JAX PEFT mini-lib (LoRA adapters, quantization wrappers, module interception) while targeting **PyTorch** and this Tessera port.
