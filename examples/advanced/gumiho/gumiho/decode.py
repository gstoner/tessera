"""End-to-end Gumiho speculative step on the Apple compiler backend.

Drives one full hybrid-speculative step:

    context → target hidden
            → serial head (2 tokens) + parallel heads (5 tokens)   [draft compute]
            → Full Tree Attention: top-8 paths, one tree-masked target pass
            → tessera.speculative.batch_verify  (Leviathan acceptance)
            → tessera.speculative.advance_kv    (commit accepted prefix)

The draft + verification dense math runs on the Apple GPU (or CPU) backend; the
whole pipeline is also run with a float64 numpy backend and the two are
cross-checked, so the demo *proves* the backend executes Gumiho correctly
rather than just printing a schedule. Degrades to numpy off Apple Silicon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import tessera as ts
from tessera.cache import KVCacheHandle

from .backend import make_backend
from .config import GumihoConfig, tiny_config
from .draft import DraftBundle, build_draft
from .model import ParallelHeads, SerialHead, TargetModel, make_weights
from .training import distill, trajectory_contexts


@dataclass(frozen=True)
class GumihoSummary:
    backend: str
    serial_tokens: int
    parallel_heads: int
    total_draft_tokens: int
    num_paths: int
    num_tree_nodes: int
    accepted_length: int
    accepted_prefix: list
    backend_matches_reference: bool
    max_logprob_abs_err: float
    kv_pre_seq: int
    kv_advanced_to: int
    validated: bool

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"backend={self.backend} | draft={self.serial_tokens}serial+"
            f"{self.parallel_heads}parallel={self.total_draft_tokens} | "
            f"FTA paths={self.num_paths} tree_nodes={self.num_tree_nodes} | "
            f"accepted={self.accepted_length} {list(self.accepted_prefix)} | "
            f"kv {self.kv_pre_seq}->{self.kv_advanced_to} | "
            f"match_ref={self.backend_matches_reference} "
            f"(max_logp_err={self.max_logprob_abs_err:.2e}) "
            f"validated={self.validated}"
        )


def _draft_for(backend, cfg, weights, context):
    target = TargetModel(weights, cfg)
    serial = SerialHead(weights, cfg)
    parallel = ParallelHeads(weights, cfg)
    last_hidden, _logits = target.forward(backend, context)
    return build_draft(
        backend, cfg, target, serial, parallel,
        context_tokens=context, last_hidden=last_hidden[-1],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-step speculative decode + acceptance / speedup metrics
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DecodeMetrics:
    backend: str
    trained: bool
    num_prompts: int
    steps: int
    tokens_generated: int
    mean_accepted_length: float    # draft tokens accepted per step (0..total_draft)
    tokens_per_step: float         # accepted + 1 bonus == tokens per target pass
    speedup_vs_vanilla: float      # vanilla autoregressive = 1.0 token / target pass

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        tag = "trained" if self.trained else "untrained"
        return (f"[{tag}] backend={self.backend} prompts={self.num_prompts} "
                f"steps={self.steps} tokens={self.tokens_generated} | "
                f"mean_accepted={self.mean_accepted_length:.2f} "
                f"tokens/step={self.tokens_per_step:.2f} "
                f"speedup={self.speedup_vs_vanilla:.2f}x")


def run_multistep_decode(cfg: GumihoConfig, weights, *, prompts: np.ndarray,
                         max_new_tokens: int, target: str = "apple_gpu",
                         trained: bool = False, seed: int = 0) -> DecodeMetrics:
    """Greedy multi-step speculative decode. Each step runs one tree-masked
    target verification pass that commits ``accepted + 1`` tokens, so
    ``tokens_per_step`` is the per-target-pass speedup over vanilla decode."""
    be = make_backend(target, eps=cfg.rmsnorm_eps)
    tgt = TargetModel(weights, cfg)
    serial = SerialHead(weights, cfg)
    parallel = ParallelHeads(weights, cfg)
    prompts = np.atleast_2d(np.asarray(prompts, np.int64))

    steps = accepted_total = tokens_total = 0
    for prompt in prompts:
        seq = list(int(t) for t in prompt)
        generated = 0
        while generated < max_new_tokens:
            ctx = np.asarray(seq[-cfg.context_len:], np.int64)
            last_hidden, _ = tgt.forward(be, ctx)
            bundle = build_draft(be, cfg, tgt, serial, parallel,
                                 context_tokens=ctx, last_hidden=last_hidden[-1])
            result = ts.speculative.batch_verify(
                target_log_probs=bundle.target_log_probs,
                draft_log_probs=bundle.draft_log_probs,
                paths=bundle.paths,
                rng=np.random.default_rng(seed + steps + 1))
            j = result.accepted_prefix_length
            win = result.accepted_path_idx
            commit = [int(t) for t in bundle.paths[win, 1:1 + j]]
            commit.append(bundle.bonus_token(win, j))   # free target token
            seq.extend(commit)
            generated += len(commit)
            steps += 1
            accepted_total += j
            tokens_total += len(commit)

    mean_acc = accepted_total / steps
    tps = tokens_total / steps
    return DecodeMetrics(
        backend=be.name, trained=trained, num_prompts=len(prompts), steps=steps,
        tokens_generated=tokens_total, mean_accepted_length=mean_acc,
        tokens_per_step=tps, speedup_vs_vanilla=tps)


def run_training_demo(cfg: GumihoConfig | None = None, *, seed: int = 0,
                      target: str = "apple_gpu", num_prompts: int = 8,
                      horizon: int = 24, max_new_tokens: int = 24,
                      train_steps: int = 600) -> tuple[DecodeMetrics, DecodeMetrics]:
    """Distill the draft on the target's own trajectories, then measure
    multi-step decode acceptance before vs after training. Returns
    ``(untrained_metrics, trained_metrics)``."""
    cfg = cfg or tiny_config()
    weights = make_weights(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    prompts = rng.integers(0, cfg.vocab, size=(num_prompts, cfg.context_len),
                           dtype=np.int64)

    before = run_multistep_decode(cfg, weights, prompts=prompts,
                                  max_new_tokens=max_new_tokens, target=target,
                                  trained=False, seed=seed)
    contexts = trajectory_contexts(cfg, weights, prompts, horizon)
    trained_w = distill(cfg, weights, contexts=contexts, serial_steps=train_steps,
                        parallel_steps=train_steps, lr=0.05, seed=seed)
    after = run_multistep_decode(cfg, trained_w, prompts=prompts,
                                 max_new_tokens=max_new_tokens, target=target,
                                 trained=True, seed=seed)
    return before, after


def run_gumiho_demo(cfg: GumihoConfig | None = None, *, seed: int = 0,
                    target: str = "apple_gpu", tol: float = 1e-3) -> GumihoSummary:
    cfg = cfg or tiny_config()
    weights = make_weights(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    context = rng.integers(0, cfg.vocab, size=cfg.context_len, dtype=np.int64)

    # Backend (Apple GPU/CPU) and float64 reference drafts.
    be = make_backend(target, eps=cfg.rmsnorm_eps)
    ref = make_backend("numpy", eps=cfg.rmsnorm_eps)
    bundle: DraftBundle = _draft_for(be, cfg, weights, context)
    ref_bundle: DraftBundle = _draft_for(ref, cfg, weights, context)

    # Validate the backend path against the reference.
    same_paths = np.array_equal(bundle.paths, ref_bundle.paths)
    max_err = float(max(
        np.max(np.abs(bundle.draft_log_probs - ref_bundle.draft_log_probs)),
        np.max(np.abs(bundle.target_log_probs - ref_bundle.target_log_probs)),
    ))

    # Leviathan acceptance — deterministic under the seeded RNG.
    result = ts.speculative.batch_verify(
        target_log_probs=bundle.target_log_probs,
        draft_log_probs=bundle.draft_log_probs,
        paths=bundle.paths,
        rng=np.random.default_rng(seed + 1),
    )
    ref_result = ts.speculative.batch_verify(
        target_log_probs=ref_bundle.target_log_probs,
        draft_log_probs=ref_bundle.draft_log_probs,
        paths=ref_bundle.paths,
        rng=np.random.default_rng(seed + 1),
    )
    same_accept = (result.accepted_prefix_length == ref_result.accepted_prefix_length
                   and np.array_equal(result.accepted_prefix, ref_result.accepted_prefix))

    # Commit only the accepted prefix to a real KV cache (advance_kv trims the
    # rejected draft tokens). Append context + full winning path, then trim.
    cache = KVCacheHandle(num_heads=cfg.num_heads, head_dim=cfg.head_dim,
                          max_seq=cfg.context_len + cfg.total_draft_tokens + 4)
    kshape = (cfg.context_len, cfg.num_heads, cfg.head_dim)
    cache.append(rng.standard_normal(kshape).astype(np.float32),
                 rng.standard_normal(kshape).astype(np.float32))
    pre_seq = cache.current_seq
    win = bundle.paths[result.accepted_path_idx, 1:]
    dshape = (cfg.total_draft_tokens, cfg.num_heads, cfg.head_dim)
    cache.append(rng.standard_normal(dshape).astype(np.float32),
                 rng.standard_normal(dshape).astype(np.float32))
    ts.speculative.advance_kv(cache, pre_seq + result.accepted_prefix_length)

    validated = (same_paths and same_accept and max_err <= tol
                 and cache.current_seq == pre_seq + result.accepted_prefix_length)

    return GumihoSummary(
        backend=be.name,
        serial_tokens=cfg.serial_tokens,
        parallel_heads=cfg.parallel_heads,
        total_draft_tokens=cfg.total_draft_tokens,
        num_paths=bundle.num_paths,
        num_tree_nodes=bundle.num_tree_nodes,
        accepted_length=result.accepted_prefix_length,
        accepted_prefix=[int(t) for t in result.accepted_prefix],
        backend_matches_reference=bool(same_paths and same_accept),
        max_logprob_abs_err=max_err,
        kv_pre_seq=int(pre_seq),
        kv_advanced_to=int(cache.current_seq),
        validated=bool(validated),
    )
