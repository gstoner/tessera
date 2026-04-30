"""
tessera_diffusion_llm/training/trainer.py

Generic trainer compatible with all three diffusion model variants.

Features:
  • AdamW + cosine LR schedule with linear warmup
  • Gradient clipping
  • Periodic checkpoint saving (model state dict + optimizer)
  • Optional EMA of model weights
  • Tessera compile hook (wraps model with tessera.compile if available)
  • Simple logging to stdout and optional W&B

Usage::

    cfg      = MDLMConfig.debug_tiny()
    model    = MDLM(cfg)
    t_cfg    = TrainerConfig(num_epochs=3, batch_size=8, lr=3e-4)
    trainer  = DiffusionTrainer(model, t_cfg)
    trainer.fit(train_dataloader)
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Hyper-parameters for DiffusionTrainer."""
    # Optimisation
    lr: float                = 3e-4
    weight_decay: float      = 0.01
    beta1: float             = 0.9
    beta2: float             = 0.999
    eps: float               = 1e-8
    grad_clip: float         = 1.0

    # Schedule
    num_epochs: int          = 10
    warmup_steps: int        = 1000
    lr_decay_type: str       = "cosine"   # "cosine" | "linear" | "constant"

    # Batch
    batch_size: int          = 32

    # Checkpointing
    ckpt_dir: str            = "checkpoints"
    save_every_n_steps: int  = 1000
    keep_last_n: int         = 3

    # EMA
    use_ema: bool            = False
    ema_decay: float         = 0.9999

    # Logging
    log_every_n_steps: int   = 50
    use_wandb: bool          = False
    wandb_project: str       = "tessera-diffusion"

    # Tessera compile
    use_tessera_compile: bool = False


# ---------------------------------------------------------------------------
# EMA helper
# ---------------------------------------------------------------------------

class EMAModel:
    """Exponential Moving Average over model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad or name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA weights into model for evaluation/checkpointing."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = state


# ---------------------------------------------------------------------------
# LR schedule factory
# ---------------------------------------------------------------------------

def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    decay_type: str,
) -> LambdaLR:

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if decay_type == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        elif decay_type == "linear":
            return max(0.0, 1.0 - progress)
        else:   # constant
            return 1.0

    return LambdaLR(optimizer, lr_lambda=_lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DiffusionTrainer:
    """
    Minimal but complete training loop for diffusion language models.

    Supports MDLM, ContinuousDiffusionLLM, and FlowMatchingLLM through
    a common `model.compute_loss(input_ids, attn_mask)` interface.

    Args:
        model:   Diffusion model instance.
        cfg:     TrainerConfig.
        device:  Target device (auto-detected if None).
        loss_fn: Custom loss function `(model, input_ids, attn_mask) → scalar`.
                 Defaults to `model.compute_loss`.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainerConfig,
        device: Optional[torch.device] = None,
        loss_fn: Optional[Callable] = None,
    ) -> None:
        self.cfg    = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)

        # Optionally compile with Tessera
        if cfg.use_tessera_compile:
            try:
                import tessera
                self.model = tessera.compile(self.model)
                print("[Trainer] Model compiled with tessera.compile")
            except ImportError:
                print("[Trainer] tessera not found; running without compiler")

        self.loss_fn = loss_fn or self._default_loss

        # Optimizer
        no_decay = {"bias", "norm", "scale", "embed"}
        wd_params, no_wd_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                no_wd_params.append(p)
            else:
                wd_params.append(p)

        self.optimizer = AdamW(
            [
                {"params": wd_params,    "weight_decay": cfg.weight_decay},
                {"params": no_wd_params, "weight_decay": 0.0},
            ],
            lr=cfg.lr,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )

        self.ema = EMAModel(self.model, cfg.ema_decay) if cfg.use_ema else None
        self._step = 0
        self._scheduler: Optional[LambdaLR] = None   # set in fit()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def fit(
        self,
        dataloader: Iterable,
        val_dataloader: Optional[Iterable] = None,
    ) -> None:
        """Train for cfg.num_epochs epochs.

        Args:
            dataloader:     Yields dicts with "input_ids" (and optionally
                            "attention_mask") tensors.
            val_dataloader: Optional validation dataloader.
        """
        # Estimate total steps for LR schedule
        try:
            steps_per_epoch = len(dataloader)  # type: ignore[arg-type]
        except TypeError:
            steps_per_epoch = 1000  # fallback for IterableDataset

        total_steps = steps_per_epoch * self.cfg.num_epochs
        self._scheduler = _make_scheduler(
            self.optimizer,
            warmup_steps=self.cfg.warmup_steps,
            total_steps=total_steps,
            decay_type=self.cfg.lr_decay_type,
        )

        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)

        if self.cfg.use_wandb:
            try:
                import wandb
                wandb.init(project=self.cfg.wandb_project)
                wandb.watch(self.model, log_freq=self.cfg.log_every_n_steps)
            except ImportError:
                print("[Trainer] wandb not installed; skipping W&B init")

        for epoch in range(self.cfg.num_epochs):
            self.model.train()
            self._run_epoch(dataloader, epoch)

            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                print(f"[Trainer] Epoch {epoch+1} val_loss={val_loss:.4f}")

    def evaluate(self, dataloader: Iterable) -> float:
        """Run one pass over a validation dataloader and return mean loss."""
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in dataloader:
                loss = self._step_loss(batch)
                total += loss.item()
                count += 1
        return total / max(count, 1)

    def save_checkpoint(self, tag: str = "") -> str:
        """Save model + optimizer state dict.  Returns path to saved file."""
        fname = os.path.join(
            self.cfg.ckpt_dir,
            f"ckpt_step{self._step}{('_' + tag) if tag else ''}.pt",
        )
        ema_state = self.ema.state_dict() if self.ema else {}
        torch.save(
            {
                "step": self._step,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict(),
                "ema_state":   ema_state,
            },
            fname,
        )
        self._prune_old_checkpoints()
        return fname

    def load_checkpoint(self, path: str) -> None:
        """Restore model, optimizer, and EMA from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])
        self._step = ckpt.get("step", 0)
        if self.ema and ckpt.get("ema_state"):
            self.ema.load_state_dict(ckpt["ema_state"])
        print(f"[Trainer] Loaded checkpoint from {path} (step {self._step})")

    # -----------------------------------------------------------------------
    # Private
    # -----------------------------------------------------------------------

    def _run_epoch(self, dataloader: Iterable, epoch: int) -> None:
        t0 = time.time()
        for batch in dataloader:
            loss = self._step_loss(batch)
            self.optimizer.zero_grad()
            loss.backward()

            if self.cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.optimizer.step()
            if self._scheduler is not None:
                self._scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)

            self._step += 1

            if self._step % self.cfg.log_every_n_steps == 0:
                elapsed = time.time() - t0
                lr_now = self.optimizer.param_groups[0]["lr"]
                print(
                    f"[Trainer] epoch={epoch+1} step={self._step} "
                    f"loss={loss.item():.4f} lr={lr_now:.2e} "
                    f"elapsed={elapsed:.1f}s"
                )
                if self.cfg.use_wandb:
                    try:
                        import wandb
                        wandb.log({"train/loss": loss.item(), "lr": lr_now}, step=self._step)
                    except Exception:
                        pass

            if (
                self.cfg.save_every_n_steps > 0
                and self._step % self.cfg.save_every_n_steps == 0
            ):
                ckpt_path = self.save_checkpoint()
                print(f"[Trainer] Saved checkpoint: {ckpt_path}")

    def _step_loss(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            ids  = batch["input_ids"].to(self.device)
            mask = batch.get("attention_mask", None)
            if mask is not None:
                mask = mask.to(self.device)
        elif isinstance(batch, (list, tuple)):
            ids  = batch[0].to(self.device)
            mask = batch[1].to(self.device) if len(batch) > 1 else None
        else:
            ids, mask = batch.to(self.device), None
        return self.loss_fn(self.model, ids, mask)

    @staticmethod
    def _default_loss(model: nn.Module, ids: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return model.compute_loss(ids, attn_mask=mask)

    def _prune_old_checkpoints(self) -> None:
        """Keep only the last cfg.keep_last_n checkpoints."""
        if self.cfg.keep_last_n <= 0:
            return
        ckpts = sorted(
            [f for f in os.listdir(self.cfg.ckpt_dir) if f.startswith("ckpt_step")],
            key=lambda x: os.path.getmtime(os.path.join(self.cfg.ckpt_dir, x)),
        )
        for old in ckpts[: -self.cfg.keep_last_n]:
            os.remove(os.path.join(self.cfg.ckpt_dir, old))
