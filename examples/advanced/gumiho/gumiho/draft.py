"""Draft tree construction + Full Tree Attention (FTA) verification.

The hybrid draft produces:
  * ``serial_tokens`` tokens from the serial head (deterministic top-1), and
  * ``parallel_heads`` independent distributions from the parallel heads.

FTA exploits the parallel heads' independence: any token from head *i* can be
connected to any token from head *j*, so we enumerate the
``fta_tokens_per_head ** parallel_heads`` combinations, score each candidate
path by its summed log-prob, and keep the ``fta_top_paths`` best — shorter
paths borrow tokens from longer ones for free because the per-position token
computations already exist.

The kept paths are folded into a **prefix trie** (shared prefixes share nodes),
the target runs *once* over ``context + trie nodes`` under a **tree-attention
mask** (each node attends only to the context and its own ancestors), and the
per-path target log-probs are gathered for the Leviathan acceptance check in
``tessera.speculative.batch_verify``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .config import GumihoConfig
from .model import NEG_INF, ParallelHeads, SerialHead, TargetModel


@dataclass(frozen=True)
class DraftBundle:
    paths: np.ndarray            # [P, total_draft+1] int64 — root + draft tokens
    draft_log_probs: np.ndarray  # [P, total_draft] float64
    target_log_probs: np.ndarray  # [P, total_draft] float64
    num_tree_nodes: int          # unique trie nodes verified in one target pass
    num_paths: int
    path_node_ids: np.ndarray    # [P, total_draft+1] int64 — trie node per position
    node_target_argmax: np.ndarray  # [num_nodes] int64 — target greedy token AT each node

    def bonus_token(self, path_idx: int, accepted_len: int) -> int:
        """The target's greedy next token after ``accepted_len`` accepted draft
        tokens on ``path_idx`` — free, from the same verification pass."""
        pred_node = int(self.path_node_ids[path_idx, accepted_len])
        return int(self.node_target_argmax[pred_node])


def _topk(logp: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(logp)[::-1][:k]
    return idx.astype(np.int64), logp[idx]


def build_draft(
    backend,
    cfg: GumihoConfig,
    target: TargetModel,
    serial: SerialHead,
    parallel: ParallelHeads,
    *,
    context_tokens: np.ndarray,
    last_hidden: np.ndarray,
) -> DraftBundle:
    context_tokens = np.asarray(context_tokens, np.int64)
    root_token = int(context_tokens[-1])

    # 1. Serial head — first `serial_tokens` draft tokens (deterministic).
    s_tokens, s_logp, s_hiddens = serial.generate(
        backend, target, last_hidden, root_token)

    # 2. Parallel heads — per-head log-probs, then top-k tokens per head.
    par_logp = parallel.predict(backend, s_hiddens)        # [heads, V]
    per_head = [_topk(par_logp[i], cfg.fta_tokens_per_head)
                for i in range(cfg.parallel_heads)]

    # 3. FTA: enumerate combinations, score, keep top paths.
    combos = []
    for choice in itertools.product(range(cfg.fta_tokens_per_head),
                                    repeat=cfg.parallel_heads):
        toks = [int(per_head[h][0][choice[h]]) for h in range(cfg.parallel_heads)]
        score = float(sum(per_head[h][1][choice[h]]
                          for h in range(cfg.parallel_heads)))
        lps = [float(per_head[h][1][choice[h]]) for h in range(cfg.parallel_heads)]
        combos.append((score, toks, lps))
    combos.sort(key=lambda c: c[0], reverse=True)
    combos = combos[: cfg.fta_top_paths]

    # 4. Assemble paths + draft log-probs. Each path = serial prefix + parallel.
    P = len(combos)
    depth = cfg.total_draft_tokens
    paths = np.empty((P, depth + 1), np.int64)
    draft_lp = np.empty((P, depth), np.float64)
    for i, (_score, toks, lps) in enumerate(combos):
        path_tokens = list(s_tokens) + toks
        paths[i, 0] = root_token
        paths[i, 1:] = path_tokens
        draft_lp[i, : cfg.serial_tokens] = [s_logp[t, path_tokens[t]]
                                            for t in range(cfg.serial_tokens)]
        draft_lp[i, cfg.serial_tokens:] = lps

    # 5. Prefix trie over the draft tokens (shared prefixes share nodes).
    #    node 0 is the virtual root (= last context position).
    children: dict[tuple[int, int], int] = {}
    node_token = [root_token]
    node_parent = [-1]
    node_depth = [0]
    path_node_ids = np.zeros((P, depth + 1), np.int64)  # trie node per path/pos
    for i in range(P):
        cur = 0
        for p in range(1, depth + 1):
            tok = int(paths[i, p])
            key = (cur, tok)
            if key not in children:
                children[key] = len(node_token)
                node_token.append(tok)
                node_parent.append(cur)
                node_depth.append(p)
            cur = children[key]
            path_node_ids[i, p] = cur
    num_nodes = len(node_token)

    # 6. Build the verification sequence + tree-attention mask.
    #    seq = context (causal) followed by trie nodes 1..num_nodes-1.
    C = context_tokens.shape[0]
    tree_tokens = np.asarray(node_token[1:], np.int64)     # exclude virtual root
    seq = np.concatenate([context_tokens, tree_tokens])
    S = seq.shape[0]
    mask = np.full((S, S), NEG_INF, np.float64)
    # context: causal among itself.
    for i in range(C):
        mask[i, : i + 1] = 0.0
    # trie node at seq index C + (n-1) attends to all context + its ancestors.
    seq_index = {n: C + (n - 1) for n in range(1, num_nodes)}
    for n in range(1, num_nodes):
        si = seq_index[n]
        mask[si, :C] = 0.0
        anc = n
        while anc >= 1:
            mask[si, seq_index[anc]] = 0.0
            anc = node_parent[anc]

    # 7. One target pass over the whole tree.
    _hidden, logits = target.forward(backend, seq, add_mask=mask)
    log_probs = logits - _logsumexp_rows(logits)           # [S, V]

    # Target's greedy token predicted at each trie node (node 0 = root = last
    # context position) — supplies the free "bonus" token after acceptance.
    node_argmax = np.empty(num_nodes, np.int64)
    node_argmax[0] = int(np.argmax(log_probs[C - 1]))
    for n in range(1, num_nodes):
        node_argmax[n] = int(np.argmax(log_probs[seq_index[n]]))

    # 8. Gather per-path target log-probs. Token t_p is predicted by the node
    #    holding t_{p-1} (or the last context position for p == 1).
    target_lp = np.empty((P, depth), np.float64)
    for i in range(P):
        for p in range(1, depth + 1):
            pred_node = path_node_ids[i, p - 1]            # 0 => virtual root
            si = (C - 1) if pred_node == 0 else seq_index[pred_node]
            target_lp[i, p - 1] = log_probs[si, int(paths[i, p])]

    return DraftBundle(
        paths=paths,
        draft_log_probs=draft_lp,
        target_log_probs=target_lp,
        num_tree_nodes=num_nodes,
        num_paths=P,
        path_node_ids=path_node_ids,
        node_target_argmax=node_argmax,
    )


def _logsumexp_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, np.float64)
    m = z.max(-1, keepdims=True)
    return m + np.log(np.exp(z - m).sum(-1, keepdims=True))
