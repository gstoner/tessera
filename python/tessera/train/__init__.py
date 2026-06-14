"""tessera.train — compact, agent-native MoE training surface.

This package is a deliberate, PithTrain-inspired re-shaping of Tessera's MoE
training story (paper: "PithTrain: A Compact and Agent-Native MoE Training
System", arXiv:2605.31463). It adopts PithTrain's four agent-native design
principles at the *application* and *engine* layers, while letting Tessera's
own compiler/runtime be the *operator* layer (the substitute for PithTrain's
DeepGEMM / FlashAttn / Triton compiled extensions).

Four principles, realized here
------------------------------
1. Compact codebase     — this package is small and reachable in one context
                          pass. Compactness is a *growth constraint*, not a
                          one-time number: new additions respect principles 2-4.
2. Python-native        — pure Python on Tessera primitives. No torch / jax /
                          flax at runtime (Architecture Decision #23). Failures
                          surface as located Tessera/Python tracebacks, never as
                          opaque native segfaults.
3. No implicit indirection — every model lives in ONE self-contained file under
                          ``models/`` and is instantiated *directly*. No plugin
                          registry, no ``ModuleSpec``/string-keyed resolution in
                          the model-construction read path. What runs at a call
                          site is identifiable by local reading.
4. Agent skills         — recurring training-framework tasks ship as in-repo
                          ``skills/<name>/SKILL.md`` playbooks with specific
                          scope, explicit prerequisites, and a *verifiable*
                          PASS/FAIL check (not agent self-assessment).

The agent-native firewall (hard rule)
-------------------------------------
Nothing under ``tessera.train`` may import the compiler's audit/registry
machinery — ``primitive_coverage``, ``op_catalog``, ``backend_manifest`` — or
any C++/MLIR. Those are exactly the "implicit indirection" PithTrain measures
as costly for agents. They belong behind ``@tessera.jit``; the compiler is the
*invisible* operator layer. Keeping the agent read-path free of them is the
whole point. ``test_train_agent_native_firewall`` (to be added) enforces this.

Status (Phase 1, honest)
------------------------
* Runs today on numpy and ``@jit(target="apple_gpu")`` (single node).
* Multi-node throughput (real EP/PP collectives, DualPipeV overlap, FP8 weight
  cache) is ``planned`` / hardware-gated behind the Phase G/H NVIDIA/ROCm
  frontier tracked in ``docs/audit/backend/BACKEND_AUDIT.md``. We scope the
  claim to *agent-task efficiency + Apple-executable single-node MoE*, not
  production throughput parity.

Layers
------
* ``tessera.train.models``  — application: one file per model, directly built.
* ``tessera.train.engine``  — engine: routing, load-balancing loss, MoE FFN.
* ``tessera.train.loop``    — training loops (RL first: GRPO/CISPO).
* ``tessera.train.skills/`` — in-repo agent skills.
"""

from __future__ import annotations

from .engine.moe import (
    MoERouter,
    MoEFeedForward,
    load_balancing_loss,
    router_z_loss,
    sparse_moe_dispatch,
    top_k_selection,
)
from .models.qwen3_moe import (
    Qwen3MoEConfig,
    Qwen3MoEBlock,
    Qwen3MoEModel,
)
from .models.traced_moe_policy import TracedMoEPolicy
from .loop.optimizer import adamw_step
from .loop.rl import GRPOConfig, grpo_step, grpo_surrogate, grpo_train_step

__all__ = [
    # engine
    "MoERouter",
    "MoEFeedForward",
    "load_balancing_loss",
    "router_z_loss",
    "sparse_moe_dispatch",
    "top_k_selection",
    # models
    "Qwen3MoEConfig",
    "Qwen3MoEBlock",
    "Qwen3MoEModel",
    "TracedMoEPolicy",
    # training loop (Tier-2 tape + AdamW)
    "adamw_step",
    "GRPOConfig",
    "grpo_step",
    "grpo_surrogate",
    "grpo_train_step",
]
