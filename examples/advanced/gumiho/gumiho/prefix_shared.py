"""Paged-KV prefix sharing for the Gumiho tree verification.

The naive verifier re-runs the target over ``context + tree_nodes`` every decode
step, recomputing the context's K/V each time. But the context's K/V depend only
on the (causal) context tokens, so they can be **prefilled once into a paged KV
cache** and reused: each verification step then computes K/V for only the ``N``
tree nodes and attends to ``[cached context K/V] ⊕ [tree ancestors]``.

``PrefixSharedVerifier`` does exactly that and reproduces the dense
``build_draft`` target log-probs bit-for-bit (same attention, just a cached
prefix), while computing **N** K/V rows per verify instead of **C + N**. Across a
multi-step decode the saving compounds, since the context only grows.

The context K/V live in a :class:`tessera.cache.KVCacheHandle` — the same paged
cache manager the runtime ships — so committing accepted tokens is a cache
append and the prefix is shared across steps for free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tessera.cache import KVCacheHandle

from .config import GumihoConfig
from .model import GumihoWeights, TargetModel


@dataclass(frozen=True)
class PrefixVerifyResult:
    target_log_probs: np.ndarray   # [P, total_draft] — matches build_draft
    node_target_argmax: np.ndarray
    kv_rows_prefill: int           # context K/V rows computed once
    kv_rows_per_verify: int        # tree-node K/V rows per step (prefix-shared)
    kv_rows_full_recompute: int    # C + N the naive path recomputes each step


def _rms(x, gamma, eps):
    d = np.asarray(x, np.float64)
    n = d / np.sqrt((d * d).mean(-1, keepdims=True) + eps)
    return n * np.asarray(gamma, np.float64)


def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class PrefixSharedVerifier:
    """Tree verification with the context K/V served from a paged KV cache."""

    def __init__(self, weights: GumihoWeights, cfg: GumihoConfig) -> None:
        self.w = weights
        self.cfg = cfg
        self.L = weights.target_layer
        self._cache_k: KVCacheHandle | None = None
        self._cache_v: KVCacheHandle | None = None
        self._ctx_len = 0

    # ------------------------------------------------------------------
    def _project_kv(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """K/V for ``tokens`` — split of ``rmsnorm(embed) @ Wqkv``. Shapes
        ``[n, H, dh]``."""
        cfg, H, dh = self.cfg, self.cfg.num_heads, self.cfg.head_dim
        h = self.w.embed[np.asarray(tokens, np.int64)]
        qkv = _rms(h, self.L.ln1, cfg.rmsnorm_eps) @ self.L.wqkv.astype(np.float64)
        _q, k, v = np.split(qkv, 3, axis=-1)
        n = len(tokens)
        return (k.reshape(n, H, dh), v.reshape(n, H, dh))

    def _project_qkv(self, tokens: np.ndarray):
        cfg, H, dh = self.cfg, self.cfg.num_heads, self.cfg.head_dim
        h = self.w.embed[np.asarray(tokens, np.int64)]
        qkv = _rms(h, self.L.ln1, cfg.rmsnorm_eps) @ self.L.wqkv.astype(np.float64)
        q, k, v = np.split(qkv, 3, axis=-1)
        n = len(tokens)
        return (q.reshape(n, H, dh), k.reshape(n, H, dh), v.reshape(n, H, dh),
                h.astype(np.float64))

    def prefill(self, context: np.ndarray, *, reserve: int = 1024) -> int:
        """Compute + cache the context K/V. Returns rows cached. ``reserve``
        sizes the paged store for the tokens the decode will append."""
        context = np.asarray(context, np.int64)
        H, dh = self.cfg.num_heads, self.cfg.head_dim
        cap = len(context) + max(reserve, self.cfg.total_draft_tokens + 8)
        self._cache_k = KVCacheHandle(num_heads=H, head_dim=dh, max_seq=cap)
        self._cache_v = KVCacheHandle(num_heads=H, head_dim=dh, max_seq=cap)
        k, v = self._project_kv(context)
        # store K in the "keys" slot of one handle, V in another (the example
        # uses two handles so each is a clean [seq, H, dh] paged store).
        self._cache_k.append(k.astype(np.float32), k.astype(np.float32))
        self._cache_v.append(v.astype(np.float32), v.astype(np.float32))
        self._ctx_len = len(context)
        return self._ctx_len

    # ------------------------------------------------------------------
    def verify_tree(self, context, bundle) -> PrefixVerifyResult:
        """Verify a draft ``bundle`` (from ``draft.build_draft``) against the
        cached context prefix, reproducing its target log-probs while computing
        K/V for only the tree nodes."""
        if self._cache_k is None:
            self.prefill(context)
        paths = np.asarray(bundle.paths, np.int64)
        path_node_ids = np.asarray(bundle.path_node_ids, np.int64)
        num_nodes = bundle.num_tree_nodes
        depth = paths.shape[1] - 1
        # reconstruct the trie (token + parent per node) from the bundle.
        node_token = np.zeros(num_nodes, np.int64)
        node_parent = np.full(num_nodes, -1, np.int64)
        node_token[0] = int(paths[0, 0])
        for i in range(paths.shape[0]):
            for p in range(1, depth + 1):
                nid = int(path_node_ids[i, p])
                node_token[nid] = int(paths[i, p])
                node_parent[nid] = int(path_node_ids[i, p - 1])
        cfg, H, dh, d = self.cfg, self.cfg.num_heads, self.cfg.head_dim, self.cfg.d_model
        eps, scale = cfg.rmsnorm_eps, 1.0 / np.sqrt(dh)
        C = self._ctx_len
        K_ctx = np.asarray(self._cache_k.keys[:C], np.float64)   # [C, H, dh]
        V_ctx = np.asarray(self._cache_v.keys[:C], np.float64)

        tree_tokens = np.asarray(node_token[1:], np.int64)       # exclude root
        num_nodes = len(node_token)
        # tree-node K/V/Q — computed ONCE for the N tree nodes (the saving).
        q_t, k_t, v_t, h_t = self._project_qkv(tree_tokens)      # [N, H, dh]
        node_logits = np.empty((num_nodes, cfg.vocab), np.float64)

        # root node 0 is the last context position: its logits come from a
        # plain context-only forward at that position.
        node_logits[0] = self._context_position_logits(context)

        for n in range(1, num_nodes):
            ti = n - 1                                           # index into tree arrays
            anc = []
            cur = n
            while cur >= 1:
                anc.append(cur - 1)                              # tree-array index
                cur = node_parent[cur]
            anc = anc[::-1]
            K = np.concatenate([K_ctx, k_t[anc]], axis=0)        # [C+|anc|, H, dh]
            V = np.concatenate([V_ctx, v_t[anc]], axis=0)
            Q = q_t[ti]                                          # [H, dh]
            scores = np.einsum("hd,khd->hk", Q, K) * scale       # [H, Lk]
            attn = _softmax(scores)
            ctx = np.einsum("hk,khd->hd", attn, V).reshape(d)
            attn_out = ctx @ self.L.wo.astype(np.float64)
            hh = h_t[ti] + attn_out
            n2 = _rms(hh, self.L.ln2, eps)
            gate = n2 @ self.L.w_gate.astype(np.float64)
            up = n2 @ self.L.w_up.astype(np.float64)
            ffn = (gate / (1.0 + np.exp(-gate)) * up) @ self.L.w_down.astype(np.float64)
            hh = hh + ffn
            fin = _rms(hh, self.w.final_norm, eps)
            node_logits[n] = fin @ self.w.lm_head.astype(np.float64)

        log_probs = node_logits - _logsumexp_rows(node_logits)
        node_argmax = np.argmax(node_logits, axis=-1).astype(np.int64)

        # gather per-path target log-probs (identical to draft.build_draft).
        paths = np.asarray(paths, np.int64)
        P, depth1 = paths.shape
        depth = depth1 - 1
        target_lp = np.empty((P, depth), np.float64)
        for i in range(P):
            for p in range(1, depth + 1):
                pred = int(path_node_ids[i, p - 1])
                target_lp[i, p - 1] = log_probs[pred, int(paths[i, p])]

        N = len(tree_tokens)
        return PrefixVerifyResult(
            target_log_probs=target_lp, node_target_argmax=node_argmax,
            kv_rows_prefill=C, kv_rows_per_verify=N, kv_rows_full_recompute=C + N)

    def _context_position_logits(self, context: np.ndarray) -> np.ndarray:
        """Full causal forward over the context; return last-position logits."""
        be_target = TargetModel(self.w, self.cfg)
        from .backend import NumpyBackend
        _h, logits = be_target.forward(NumpyBackend(eps=self.cfg.rmsnorm_eps),
                                       np.asarray(context, np.int64))
        return np.asarray(logits[-1], np.float64)

    def commit(self, accepted_tokens: np.ndarray) -> None:
        """Append accepted tokens' K/V to the cache — the prefix grows, shared
        into the next step with no recompute."""
        if not len(accepted_tokens):
            return
        k, v = self._project_kv(np.asarray(accepted_tokens, np.int64))
        self._cache_k.append(k.astype(np.float32), k.astype(np.float32))
        self._cache_v.append(v.astype(np.float32), v.astype(np.float32))
        self._ctx_len += len(accepted_tokens)


def _logsumexp_rows(z):
    m = z.max(-1, keepdims=True)
    return m + np.log(np.exp(z - m).sum(-1, keepdims=True))


@dataclass(frozen=True)
class PrefixSharingSummary:
    num_prompts: int
    steps: int
    tokens_generated: int
    naive_kv_rows: int          # context+tree K/V recomputed every step
    shared_kv_rows: int         # prefill once + accepted appends + tree nodes
    kv_rows_saved: int
    reduction_pct: float
    verify_matches_recompute: bool
    max_target_logprob_err: float

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return (f"prompts={self.num_prompts} steps={self.steps} "
                f"tokens={self.tokens_generated} | K/V rows "
                f"{self.shared_kv_rows} vs {self.naive_kv_rows} naive "
                f"({self.reduction_pct:.0f}% fewer) | "
                f"verify_matches={self.verify_matches_recompute} "
                f"(err={self.max_target_logprob_err:.2e})")


def run_prefix_sharing_demo(cfg: GumihoConfig, weights, *, prompts,
                            max_new_tokens: int, seed: int = 0) -> PrefixSharingSummary:
    """Multi-step decode over the **full growing context**, serving the context
    K/V from a paged cache. Measures the K/V rows the prefix-shared path computes
    (prefill once + accepted-token appends + per-step tree nodes) vs. the naive
    path (full context + tree recomputed every step), and checks the prefix-
    shared verify reproduces the naive target log-probs."""
    import tessera as ts

    from .backend import NumpyBackend
    from .draft import build_draft
    from .model import ParallelHeads, SerialHead

    be = NumpyBackend(eps=cfg.rmsnorm_eps)
    tgt = TargetModel(weights, cfg)
    serial = SerialHead(weights, cfg)
    parallel = ParallelHeads(weights, cfg)
    prompts = np.atleast_2d(np.asarray(prompts, np.int64))

    naive = shared = steps = tokens = 0
    max_err = 0.0
    matches = True
    for prompt in prompts:
        seq = [int(t) for t in prompt]
        pv = PrefixSharedVerifier(weights, cfg)
        pv.prefill(np.asarray(seq, np.int64))
        shared += len(seq)                         # prefill the prompt K/V once
        generated = 0
        while generated < max_new_tokens:
            ctx = np.asarray(seq, np.int64)         # full growing context
            last_hidden, _ = tgt.forward(be, ctx)
            bundle = build_draft(be, cfg, tgt, serial, parallel,
                                 context_tokens=ctx, last_hidden=last_hidden[-1])
            res = pv.verify_tree(ctx, bundle)
            err = float(np.max(np.abs(res.target_log_probs - bundle.target_log_probs)))
            max_err = max(max_err, err)
            matches = matches and err < 1e-4

            result = ts.speculative.batch_verify(
                target_log_probs=bundle.target_log_probs,
                draft_log_probs=bundle.draft_log_probs, paths=bundle.paths,
                rng=np.random.default_rng(seed + steps + 1))
            j = result.accepted_prefix_length
            commit = [int(t) for t in bundle.paths[result.accepted_path_idx, 1:1 + j]]
            commit.append(bundle.bonus_token(result.accepted_path_idx, j))

            n_tree = res.kv_rows_per_verify
            naive += len(ctx) + n_tree              # naive recomputes context+tree
            shared += n_tree                        # shared recomputes only tree
            pv.commit(np.asarray(commit, np.int64))  # append accepted K/V (once)
            shared += len(commit)                   # the only new context K/V
            seq.extend(commit)
            generated += len(commit)
            steps += 1
            tokens += len(commit)

    saved = naive - shared
    return PrefixSharingSummary(
        num_prompts=len(prompts), steps=steps, tokens_generated=tokens,
        naive_kv_rows=naive, shared_kv_rows=shared, kv_rows_saved=saved,
        reduction_pct=100.0 * saved / naive if naive else 0.0,
        verify_matches_recompute=matches, max_target_logprob_err=max_err)
