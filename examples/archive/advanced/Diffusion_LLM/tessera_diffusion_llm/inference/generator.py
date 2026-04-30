"""
tessera_diffusion_llm/inference/generator.py

High-level generation utilities for all three diffusion model variants.

DiffusionGenerator wraps any of the three model classes with a unified
interface for sampling, prompt conditioning, batched generation, and
decoding back to text (with an optional tokenizer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    """Generation hyper-parameters."""
    # Denoising steps (None = model default)
    num_steps:   Optional[int] = None
    # Temperature for final token sampling; 1.0 = no scaling
    temperature: float         = 1.0
    # Top-k filtering (0 = off, use argmax)
    top_k:       int           = 0
    # DDPM vs DDIM (continuous model only)
    sampler:     str           = "ddpm"
    # DDIM stochasticity (0=det, 1=DDPM)
    eta:         float         = 0.0
    # ODE solver for flow matching ("euler" | "midpoint")
    solver:      str           = "euler"
    # Max number of sequences per device call (splits larger batches)
    max_batch:   int           = 64


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class DiffusionGenerator:
    """
    Unified inference wrapper for MDLM / ContinuousDiffusionLLM / FlowMatchingLLM.

    Usage::

        model = MDLM(MDLMConfig())
        gen   = DiffusionGenerator(model, GeneratorConfig(num_steps=50))
        ids   = gen.generate(batch_size=8, seq_len=128)
        texts = gen.decode(ids, tokenizer)
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: Optional[GeneratorConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model  = model
        self.cfg    = cfg or GeneratorConfig()
        self.device = device or next(model.parameters()).device
        self.model.eval()

    # -----------------------------------------------------------------------
    # Primary generation method
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        seq_len: int,
        prompt_ids: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        """Generate token sequences.

        Automatically dispatches to the correct sampling method based on
        the model type.

        Args:
            batch_size: Number of sequences to generate.
            seq_len:    Length of each sequence.
            prompt_ids: Optional (1, P) or (B, P) conditioning prefix.

        Returns:
            (B, seq_len) integer token tensor on CPU.
        """
        model_cls = type(self.model).__name__
        cfg = self.cfg

        # Broadcast prompt to batch
        if prompt_ids is not None:
            if prompt_ids.shape[0] == 1 and batch_size > 1:
                prompt_ids = prompt_ids.expand(batch_size, -1)
            prompt_ids = prompt_ids.to(self.device)

        # Split large batches
        all_tokens: List[torch.Tensor] = []
        for start in range(0, batch_size, cfg.max_batch):
            end = min(start + cfg.max_batch, batch_size)
            sub_bs = end - start
            sub_prompt = None
            if prompt_ids is not None:
                sub_prompt = prompt_ids[start:end]

            tokens = self._generate_sub(sub_bs, seq_len, sub_prompt, model_cls, cfg)
            all_tokens.append(tokens.cpu())

        return torch.cat(all_tokens, dim=0)

    # -----------------------------------------------------------------------
    # Convenience wrappers
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def generate_from_text(
        self,
        prompt: str,
        seq_len: int,
        batch_size: int = 1,
        tokenizer: Any = None,
    ) -> torch.LongTensor:
        """Encode prompt → generate → return token tensor.

        Args:
            prompt:     Text prompt.
            seq_len:    Total generation length (including prompt).
            batch_size: Number of sequences.
            tokenizer:  HuggingFace-style tokenizer with `encode` method.

        Returns:
            (B, seq_len) token tensor.
        """
        if tokenizer is None:
            raise ValueError("tokenizer must be provided for text prompts")
        ids = tokenizer.encode(prompt, return_tensors="pt")  # (1, P)
        return self.generate(batch_size=batch_size, seq_len=seq_len, prompt_ids=ids)

    def decode(
        self,
        token_ids: torch.LongTensor,
        tokenizer: Any,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode a batch of token ids to strings.

        Args:
            token_ids:           (B, T) integer tensor.
            tokenizer:           HuggingFace-style tokenizer.
            skip_special_tokens: Whether to strip special tokens.

        Returns:
            List of decoded strings, one per sequence.
        """
        return [
            tokenizer.decode(ids.tolist(), skip_special_tokens=skip_special_tokens)
            for ids in token_ids
        ]

    # -----------------------------------------------------------------------
    # Internal dispatch
    # -----------------------------------------------------------------------

    def _generate_sub(
        self,
        batch_size: int,
        seq_len: int,
        prompt_ids: Optional[torch.LongTensor],
        model_cls: str,
        cfg: GeneratorConfig,
    ) -> torch.LongTensor:
        kwargs: dict = dict(
            batch_size=batch_size,
            seq_len=seq_len,
            num_steps=cfg.num_steps,
            temperature=cfg.temperature,
            prompt_ids=prompt_ids,
        )

        if model_cls == "MDLM":
            return self.model.generate(**kwargs)

        elif model_cls == "ContinuousDiffusionLLM":
            return self.model.generate(
                **kwargs,
                sampler=cfg.sampler,
                eta=cfg.eta,
                top_k=cfg.top_k,
            )

        elif model_cls == "FlowMatchingLLM":
            return self.model.generate(
                **kwargs,
                solver=cfg.solver,
                top_k=cfg.top_k,
            )

        else:
            # Fallback: assume the model has a .generate() compatible API
            return self.model.generate(**kwargs)

    # -----------------------------------------------------------------------
    # Token-level diagnostics
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute per-sequence negative log-likelihood (NLL) proxy.

        For MDLM, approximates NLL as the masked CE loss averaged over
        multiple timestep samples.  For continuous/flow models, returns
        the compute_loss value (MSE in embedding space).

        Args:
            input_ids: (B, T) token ids.

        Returns:
            (B,) float tensor of per-sequence NLL proxies.
        """
        input_ids = input_ids.to(self.device)
        B = input_ids.shape[0]
        scores = torch.zeros(B, device=self.device)

        num_samples = 8   # average over 8 timestep draws
        for _ in range(num_samples):
            loss = self.model.compute_loss(input_ids)
            scores += loss.detach()

        return scores / num_samples
