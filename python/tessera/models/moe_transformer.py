"""Shared MoE-transformer contract layer for the model-class roadmap.

Kimi-K2, DeepSeek-V3.2, GLM-5.2, and MiniMax-M3 are the *same* architecture family —
MoE-FFN transformers with latent (MLA) or grouped (GQA) attention, optional
DeepSeek-style sparse attention (DSA), MiniMax Sparse Attention (MSA), and
low-precision (INT4 / FP8) weights.
Rather than near-duplicate graphs, this module is the shared, shape-only
contract: one :class:`MoETransformerConfig`, one config verifier, one
block-graph builder, and a parameter-budget estimator.  The per-model modules
(``deepseek_v32`` / ``glm5`` / ``kimi_k2`` / ``minimax_m3``) are thin config
factories on top.

This is the **M0 scaffolding**: it builds and verifies the op graph (the
vocabulary the M3/M4 attention kernels and M1/M2 quant+MoE lowering must
implement) without execution.  Reuses the :class:`GraphNode` / ``GraphNodeList``
representation style of :mod:`diffusion_gemma`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .diffusion_gemma import GraphNode


class MoETransformerDimError(ValueError):
    """Config / built-graph dimension contract violation."""


ATTENTION_KINDS = ("mla", "gqa")
SPARSE_KINDS = (None, "dsa", "lsa", "msa")
QUANT_DTYPES = (None, "int4", "fp8_e4m3", "fp8_e5m2")
MSA_SCORE_TYPES = ("max", "mean_reference")


@dataclass(frozen=True)
class SharedTopKIndexGroup:
    """DSA IndexShare contract for one producer layer and its consumers."""

    producer_layer: int
    consumer_layers: tuple[int, ...]
    top_k: int
    tie_break: str = "stable_lowest_index"
    storage_policy: str = "current_query_only"


@dataclass(frozen=True)
class MoETransformerConfig:
    """One MoE-transformer model's shape contract.

    Attention is ``"mla"`` (multi-head latent — compress to ``kv_lora_rank``,
    cache the latent, expand to K/V; a decoupled ``rope_head_dim`` carries
    position) or ``"gqa"`` (``num_attention_heads`` query heads share
    ``num_kv_heads``).  ``sparse="dsa"`` adds DeepSeek sparse attention (top-k
    block selection); ``sparse="lsa"`` adds Lookahead Sparse Attention (local
    window ∪ threshold-selected past blocks); ``sparse="msa"`` adds MiniMax
    Sparse Attention with per-layer enablement.  ``weight_dtype`` selects the
    packed quant scheme the expert/dense GEMMs lower through (``stdlib.quant``).
    ``first_k_dense`` leading layers use a plain FFN before the MoE layers begin
    (DeepSeek convention).
    """

    name: str = "moe_transformer"
    hidden_size: int = 2048
    num_layers: int = 4
    vocab_size: int = 32000
    context_length: int = 8192

    # ── attention ────────────────────────────────────────────────────────────
    attn_kind: str = "gqa"
    num_attention_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 128
    qk_head_dim: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    # MLA latent ranks (attn_kind == "mla")
    q_lora_rank: int = 0          # 0 → no query down-projection
    kv_lora_rank: int = 512
    rope_head_dim: int = 64       # decoupled-RoPE head dim (partial RoPE)
    rope_variant: str = "rope"
    rope_theta: float = 10000.0

    # ── sparse / hybrid attention ─────────────────────────────────────────────
    sparse: str | None = None     # None | "dsa" | "lsa" | "msa"
    dsa_top_k_blocks: int = 0
    dsa_block_size: int = 64       # block size for DSA *and* LSA
    index_n_heads: int = 0
    index_head_dim: int = 0
    index_topk: int = 0
    index_topk_freq: int = 0
    index_skip_topk_offset: int = 0
    indexer_types: tuple[str, ...] = ()
    indexer_rope_interleave: bool = False
    sliding_window: int = 0       # 0 → dense; >0 → sliding-window layers
    layer_types: tuple[str, ...] = ("full",)
    # Lookahead Sparse Attention (sparse == "lsa"): local window ∪ threshold-
    # selected strictly-past blocks (block size = dsa_block_size).
    lsa_window_size: int = 0
    lsa_threshold: float = 0.5
    # MiniMax Sparse Attention (sparse == "msa"): exact block-sparse attention
    # driven by an Index Branch. ``msa_sparse_layer_freq`` mirrors HF's per-layer
    # sparse_attention_freq: 0 = dense attention, 1 = MSA.
    msa_top_k_blocks: int = 0
    msa_block_size: int = 0
    msa_index_dim: int = 0
    msa_num_index_heads: int = 0
    msa_score_type: str = "mean_reference"
    msa_sparse_layer_freq: tuple[int, ...] = ()

    # ── MoE ────────────────────────────────────────────────────────────────────
    num_experts: int = 16
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1
    moe_intermediate_size: int = 1408
    shared_expert_intermediate_size: int = 1408
    first_k_dense: int = 0        # leading dense-FFN layers before MoE
    dense_intermediate_size: int = 0  # 0 → reuse shared_expert_intermediate_size

    # ── quant ──────────────────────────────────────────────────────────────────
    weight_dtype: str | None = None
    quant_group_size: int = 0     # 0 → per-channel; >0 → grouped along K

    # ── numerics / budget ──────────────────────────────────────────────────────
    rms_norm_eps: float = 1e-6
    dtype: str = "bf16"
    total_params_b: float = 0.0   # 0 → budget unchecked (placeholder configs)
    active_params_b: float = 0.0
    hf_model_size_b: float = 0.0
    rollout_kv_dtype: str = ""

    # ── MTP / speculative decode ──────────────────────────────────────────────
    mtp_num_steps: int = 0
    mtp_num_layers: int = 0
    mtp_share_parameters: bool = False
    mtp_index_share: bool = False
    mtp_kv_share: bool = False

    @property
    def attn_dim(self) -> int:
        return self.num_attention_heads * self.qk_per_head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.qk_per_head_dim

    @property
    def qk_per_head_dim(self) -> int:
        return self.qk_head_dim or self.head_dim

    @property
    def value_per_head_dim(self) -> int:
        return self.v_head_dim or self.head_dim

    @property
    def value_dim(self) -> int:
        return self.num_attention_heads * self.value_per_head_dim

    @property
    def kv_value_dim(self) -> int:
        return self.num_kv_heads * self.value_per_head_dim

    def is_moe_layer(self, layer_index: int) -> bool:
        return layer_index >= self.first_k_dense

    def attention_mode(self, layer_index: int) -> str:
        if not self.layer_types:
            return "full"
        return self.layer_types[layer_index % len(self.layer_types)]

    def indexer_mode(self, layer_index: int) -> str:
        if not self.indexer_types:
            return "full" if self.sparse == "dsa" else "none"
        return self.indexer_types[layer_index % len(self.indexer_types)]

    def uses_msa_layer(self, layer_index: int) -> bool:
        if self.sparse != "msa":
            return False
        if not self.msa_sparse_layer_freq:
            return True
        return bool(self.msa_sparse_layer_freq[layer_index % len(self.msa_sparse_layer_freq)])


def shared_topk_index_groups(config: MoETransformerConfig) -> tuple[SharedTopKIndexGroup, ...]:
    """Return the explicit DSA IndexShare groups implied by ``indexer_types``."""
    if not config.indexer_types:
        return ()
    groups: list[SharedTopKIndexGroup] = []
    producer: int | None = None
    consumers: list[int] = []
    for layer, mode in enumerate(config.indexer_types):
        if mode == "full":
            if producer is not None:
                groups.append(SharedTopKIndexGroup(
                    producer_layer=producer,
                    consumer_layers=tuple(consumers),
                    top_k=config.index_topk or config.dsa_top_k_blocks,
                ))
            producer = layer
            consumers = []
        elif mode == "shared":
            if producer is None:
                raise MoETransformerDimError(
                    f"indexer_types[{layer}]='shared' has no previous full producer")
            consumers.append(layer)
        else:
            raise MoETransformerDimError(
                f"indexer_types entries must be 'full' or 'shared'; got {mode!r}")
    if producer is not None:
        groups.append(SharedTopKIndexGroup(
            producer_layer=producer,
            consumer_layers=tuple(consumers),
            top_k=config.index_topk or config.dsa_top_k_blocks,
        ))
    return tuple(groups)


def deterministic_topk_indices(scores, top_k: int) -> np.ndarray:
    """Stable descending top-k indices with ties resolved by lower index."""
    s = np.asarray(scores)
    if top_k < 1 or top_k > s.shape[-1]:
        raise ValueError(f"top_k={top_k} out of [1, {s.shape[-1]}]")
    return np.argsort(-s, axis=-1, kind="stable")[..., :top_k].astype(np.int64)


def verify_config(config: MoETransformerConfig) -> None:
    """Reject an internally-inconsistent config before any graph is built."""
    if config.hidden_size <= 0 or config.head_dim <= 0:
        raise MoETransformerDimError("hidden_size and head_dim must be positive")
    if config.num_layers <= 0:
        raise MoETransformerDimError("num_layers must be positive")
    if config.attn_kind not in ATTENTION_KINDS:
        raise MoETransformerDimError(
            f"attn_kind must be one of {ATTENTION_KINDS}; got {config.attn_kind!r}")
    if config.sparse not in SPARSE_KINDS:
        raise MoETransformerDimError(
            f"sparse must be one of {SPARSE_KINDS}; got {config.sparse!r}")
    if config.weight_dtype not in QUANT_DTYPES:
        raise MoETransformerDimError(
            f"weight_dtype must be one of {QUANT_DTYPES}; got {config.weight_dtype!r}")

    if config.attn_kind == "gqa":
        if config.num_kv_heads <= 0 or config.num_attention_heads % config.num_kv_heads != 0:
            raise MoETransformerDimError(
                f"GQA head mismatch: num_attention_heads={config.num_attention_heads} "
                f"not a multiple of num_kv_heads={config.num_kv_heads}")
    else:  # mla
        if config.kv_lora_rank <= 0:
            raise MoETransformerDimError("MLA requires kv_lora_rank > 0")
        if config.rope_head_dim < 0 or config.rope_head_dim > config.qk_per_head_dim:
            raise MoETransformerDimError(
                f"rope_head_dim={config.rope_head_dim} must be in "
                f"[0, qk_head_dim={config.qk_per_head_dim}]")
        if config.qk_head_dim and config.qk_nope_head_dim + config.qk_rope_head_dim != config.qk_head_dim:
            raise MoETransformerDimError(
                "qk_nope_head_dim + qk_rope_head_dim must equal qk_head_dim")
        if config.qk_rope_head_dim and config.qk_rope_head_dim != config.rope_head_dim:
            raise MoETransformerDimError("qk_rope_head_dim must match rope_head_dim")

    if config.sparse == "dsa":
        if config.dsa_top_k_blocks <= 0 or config.dsa_block_size <= 0:
            raise MoETransformerDimError(
                "DSA requires dsa_top_k_blocks > 0 and dsa_block_size > 0")
        if config.indexer_types:
            if len(config.indexer_types) != config.num_layers:
                raise MoETransformerDimError(
                    f"indexer_types length {len(config.indexer_types)} must equal "
                    f"num_layers={config.num_layers}")
            if config.index_topk <= 0:
                raise MoETransformerDimError("DSA IndexShare requires index_topk > 0")
            if config.index_topk_freq <= 0:
                raise MoETransformerDimError("DSA IndexShare requires index_topk_freq > 0")
            shared_topk_index_groups(config)
    if config.sparse == "lsa":
        if config.lsa_window_size <= 0 or config.dsa_block_size <= 0:
            raise MoETransformerDimError(
                "LSA requires lsa_window_size > 0 and dsa_block_size > 0")
        if not (0.0 <= config.lsa_threshold <= 1.0):
            raise MoETransformerDimError("lsa_threshold must be in [0, 1]")
    if config.sparse == "msa":
        if config.msa_top_k_blocks <= 0 or config.msa_block_size <= 0:
            raise MoETransformerDimError(
                "MSA requires msa_top_k_blocks > 0 and msa_block_size > 0")
        if config.msa_score_type not in MSA_SCORE_TYPES:
            raise MoETransformerDimError(
                f"msa_score_type must be one of {MSA_SCORE_TYPES}; got "
                f"{config.msa_score_type!r}")
        if config.msa_sparse_layer_freq:
            if len(config.msa_sparse_layer_freq) != config.num_layers:
                raise MoETransformerDimError(
                    f"msa_sparse_layer_freq length {len(config.msa_sparse_layer_freq)} "
                    f"must equal num_layers={config.num_layers}")
            bad = tuple(v for v in config.msa_sparse_layer_freq if v not in (0, 1))
            if bad:
                raise MoETransformerDimError(
                    "msa_sparse_layer_freq entries must be 0 or 1")

    if config.num_experts < 1:
        raise MoETransformerDimError("num_experts must be >= 1")
    if not (1 <= config.num_experts_per_tok <= config.num_experts):
        raise MoETransformerDimError(
            f"num_experts_per_tok={config.num_experts_per_tok} must be in "
            f"[1, num_experts={config.num_experts}]")
    if config.moe_intermediate_size <= 0:
        raise MoETransformerDimError("moe_intermediate_size must be positive")
    if not (0 <= config.first_k_dense <= config.num_layers):
        raise MoETransformerDimError(
            f"first_k_dense={config.first_k_dense} out of [0, num_layers]")

    if config.weight_dtype is not None and config.quant_group_size:
        # The quantized GEMM groups along the contraction (K = hidden) axis.
        if config.hidden_size % config.quant_group_size != 0:
            raise MoETransformerDimError(
                f"hidden_size={config.hidden_size} must be divisible by "
                f"quant_group_size={config.quant_group_size}")


@dataclass(frozen=True)
class BlockGraph:
    """One decoder layer as an ordered, shape-checked node list."""

    nodes: tuple[GraphNode, ...]
    config: MoETransformerConfig
    layer_index: int
    is_moe: bool

    def op_sequence(self) -> tuple[str, ...]:
        return tuple(n.op for n in self.nodes)

    def find(self, op: str) -> GraphNode:
        for n in self.nodes:
            if n.op == op:
                return n
        raise KeyError(op)

    def find_all(self, op: str) -> tuple[GraphNode, ...]:
        return tuple(n for n in self.nodes if n.op == op)


def _add_attention(nodes: list[GraphNode], config: MoETransformerConfig,
                   layer_index: int, T: str) -> None:
    H = config.hidden_size
    mode = config.attention_mode(layer_index)
    sparse = config.sparse

    def add(op, inputs, output, **attrs):
        nodes.append(GraphNode(op=op, inputs=tuple(inputs), output=tuple(output), attrs=attrs))

    if config.attn_kind == "mla":
        R = config.kv_lora_rank
        qk_dim = config.attn_dim
        v_dim = config.value_dim
        # compress hidden → latent, cache latent, expand to K/V (the MLA chain
        # the M3 FlashMLA kernel will absorb).
        add("latent_kv_compress", [(T, H), (H, R)], (T, R), kv_lora_rank=R)
        add("q_proj", [(T, H), (H, qk_dim)], (T, qk_dim))
        add("latent_kv_expand_k", [(T, R), (R, qk_dim)], (T, qk_dim))
        add("latent_kv_expand_v", [(T, R), (R, v_dim)], (T, v_dim))
        kv_dim = qk_dim
        kv_value_dim = v_dim
        unified_kv = True
    else:  # gqa
        kv_dim = config.kv_dim
        kv_value_dim = config.kv_value_dim
        add("q_proj", [(T, H), (H, config.attn_dim)], (T, config.attn_dim))
        add("k_proj", [(T, H), (H, kv_dim)], (T, kv_dim))
        add("v_proj", [(T, H), (H, kv_value_dim)], (T, kv_value_dim))
        unified_kv = False

    # decoupled / partial RoPE on the rope_head_dim slice when configured.
    add("rope", [(T, config.attn_dim)], (T, config.attn_dim),
        applies_to="q", variant=config.rope_variant,
        rope_head_dim=(config.rope_head_dim or config.head_dim))

    attn_op = {"dsa": "deepseek_sparse_attention",
               "lsa": "lookahead_sparse_attention"}.get(sparse or "", "attention")
    if sparse == "msa" and config.uses_msa_layer(layer_index):
        attn_op = "msa_sparse_attention"
    attn_attrs = dict(
        mode=mode, num_heads=config.num_attention_heads,
        num_kv_heads=(config.num_attention_heads if unified_kv else config.num_kv_heads),
        head_dim=config.head_dim, attn_kind=config.attn_kind,
        sliding_window=(config.sliding_window if mode == "sliding" else None),
        unified_kv=unified_kv, rope_variant=config.rope_variant)
    if sparse == "dsa":
        index_mode = config.indexer_mode(layer_index)
        group = None
        if config.indexer_types:
            groups = shared_topk_index_groups(config)
            group = next((g for g in groups
                          if layer_index == g.producer_layer
                          or layer_index in g.consumer_layers), None)
            if index_mode == "full":
                add("dsa_topk_indexer", [(T, config.attn_dim), (T, kv_dim)], (T, config.index_topk),
                    index_n_heads=config.index_n_heads,
                    index_head_dim=config.index_head_dim,
                    index_topk=config.index_topk,
                    index_topk_freq=config.index_topk_freq,
                    index_skip_topk_offset=config.index_skip_topk_offset,
                    tie_break="stable_lowest_index",
                    storage_policy="current_query_only",
                    group_producer_layer=layer_index,
                    group_consumer_layers=(group.consumer_layers if group else ()))
            else:
                add("shared_topk_index", [], (T, config.index_topk),
                    source_layer=(group.producer_layer if group else None),
                    consumer_layer=layer_index,
                    index_topk=config.index_topk,
                    tie_break="stable_lowest_index",
                    storage_policy="current_query_only")
        attn_attrs.update(top_k_blocks=config.dsa_top_k_blocks,
                          block_size=config.dsa_block_size,
                          indexer_mode=index_mode,
                          index_topk=config.index_topk or None,
                          index_share_group=(
                              group.producer_layer if group is not None else None),
                          index_storage_policy=(
                              "current_query_only" if config.indexer_types else None))
    elif sparse == "lsa":
        attn_attrs.update(window_size=config.lsa_window_size,
                          block_size=config.dsa_block_size,
                          threshold=config.lsa_threshold)
    elif sparse == "msa" and config.uses_msa_layer(layer_index):
        attn_attrs.update(top_k_blocks=config.msa_top_k_blocks,
                          block_size=config.msa_block_size,
                          index_dim=config.msa_index_dim,
                          num_index_heads=config.msa_num_index_heads,
                          score_type=config.msa_score_type,
                          force_local_block=True,
                          causal=True)
    add(attn_op,
        [(T, config.attn_dim), (T, kv_dim), (T, kv_dim)],
        (T, config.value_dim), **attn_attrs)
    add("o_proj", [(T, config.value_dim), (config.value_dim, H)], (T, H))
    add("residual_add", [(T, H), (T, H)], (T, H), source="attn")


def _add_ffn(nodes: list[GraphNode], config: MoETransformerConfig,
             layer_index: int, T: str, is_moe: bool) -> None:
    H = config.hidden_size
    E = config.num_experts

    def add(op, inputs, output, **attrs):
        nodes.append(GraphNode(op=op, inputs=tuple(inputs), output=tuple(output), attrs=attrs))

    quant_attrs = ({} if config.weight_dtype is None
                   else {"weight_dtype": config.weight_dtype,
                         "quant_group_size": config.quant_group_size or None})

    if not is_moe:
        Fd = config.dense_intermediate_size or config.shared_expert_intermediate_size
        add("dense_ffn", [(T, H), (H, Fd), (H, Fd), (Fd, H)], (T, H),
            intermediate_size=Fd, **quant_attrs)
        add("residual_add", [(T, H), (T, H)], (T, H), source="ffn")
        return

    F = config.moe_intermediate_size
    add("router", [(T, H), (H, E)], (T, E),
        top_k=config.num_experts_per_tok, num_experts=E)
    add("moe_swiglu_block",
        [(T, H), (E, H, F), (E, H, F), (E, F, H), (E,)], (T, H),
        num_experts=E, moe_intermediate_size=F, **quant_attrs)
    if config.num_shared_experts > 0:
        Fs = config.shared_expert_intermediate_size
        add("shared_expert", [(T, H), (H, Fs), (H, Fs), (Fs, H)], (T, H),
            shared_intermediate_size=Fs, **quant_attrs)
        add("moe_combine", [(T, H), (T, H), (T, config.num_experts_per_tok)], (T, H))
    else:
        add("moe_combine", [(T, H), (T, config.num_experts_per_tok)], (T, H))
    add("residual_add", [(T, H), (T, H)], (T, H), source="moe")


def build_block(config: MoETransformerConfig, *, layer_index: int = 0) -> BlockGraph:
    """Build one decoder layer as a verified, shape-only graph."""
    verify_config(config)
    H = config.hidden_size
    T = "T"
    is_moe = config.is_moe_layer(layer_index)
    nodes: list[GraphNode] = []
    nodes.append(GraphNode("rmsnorm", ((T, H), (H,)), (T, H),
                           {"eps": config.rms_norm_eps, "position": "input"}))
    _add_attention(nodes, config, layer_index, T)
    nodes.append(GraphNode("rmsnorm", ((T, H), (H,)), (T, H),
                           {"eps": config.rms_norm_eps, "position": "post_attn"}))
    _add_ffn(nodes, config, layer_index, T, is_moe)
    graph = BlockGraph(nodes=tuple(nodes), config=config,
                       layer_index=layer_index, is_moe=is_moe)
    verify_block(graph, config)
    return graph


def verify_block(graph: BlockGraph, config: MoETransformerConfig) -> None:
    """Config-aware graph verifier — reject mismatched dims before runtime."""
    verify_config(config)
    H = config.hidden_size

    q = graph.find("q_proj")
    if q.output[-1] != config.attn_dim:
        raise MoETransformerDimError(
            f"q_proj out width {q.output[-1]} != attn_dim={config.attn_dim}")

    if config.attn_kind == "mla":
        comp = graph.find("latent_kv_compress")
        if comp.output[-1] != config.kv_lora_rank:
            raise MoETransformerDimError(
                f"latent_kv_compress out {comp.output[-1]} != kv_lora_rank="
                f"{config.kv_lora_rank}")
        v = graph.find("latent_kv_expand_v")
        if v.output[-1] != config.value_dim:
            raise MoETransformerDimError(
                f"latent_kv_expand_v out {v.output[-1]} != value_dim={config.value_dim}")
    else:
        k = graph.find("k_proj")
        if k.output[-1] != config.kv_dim:
            raise MoETransformerDimError(
                f"k_proj out width {k.output[-1]} != kv_dim={config.kv_dim}")
        v = graph.find("v_proj")
        if v.output[-1] != config.kv_value_dim:
            raise MoETransformerDimError(
                f"v_proj out width {v.output[-1]} != kv_value_dim={config.kv_value_dim}")

    o = graph.find("o_proj")
    if o.inputs[0][-1] != config.value_dim:
        raise MoETransformerDimError("o_proj input width must match value_dim")

    if config.sparse == "dsa":
        attn = graph.find("deepseek_sparse_attention")
        if int(attn.attrs.get("top_k_blocks", 0)) != config.dsa_top_k_blocks:
            raise MoETransformerDimError("DSA top_k_blocks attr mismatch")
        if config.indexer_types:
            mode = config.indexer_mode(graph.layer_index)
            if attn.attrs.get("indexer_mode") != mode:
                raise MoETransformerDimError("DSA indexer_mode attr mismatch")
            op = "dsa_topk_indexer" if mode == "full" else "shared_topk_index"
            graph.find(op)
    if config.sparse == "lsa":
        attn = graph.find("lookahead_sparse_attention")
        if int(attn.attrs.get("window_size", 0)) != config.lsa_window_size:
            raise MoETransformerDimError("LSA window_size attr mismatch")
    if config.sparse == "msa":
        if config.uses_msa_layer(graph.layer_index):
            attn = graph.find("msa_sparse_attention")
            if int(attn.attrs.get("top_k_blocks", 0)) != config.msa_top_k_blocks:
                raise MoETransformerDimError("MSA top_k_blocks attr mismatch")
            if int(attn.attrs.get("block_size", 0)) != config.msa_block_size:
                raise MoETransformerDimError("MSA block_size attr mismatch")
            if attn.attrs.get("score_type") != config.msa_score_type:
                raise MoETransformerDimError("MSA score_type attr mismatch")
        else:
            graph.find("attention")

    if graph.is_moe:
        router = graph.find("router")
        if router.output[-1] != config.num_experts:
            raise MoETransformerDimError(
                f"router out width {router.output[-1]} != num_experts={config.num_experts}")
        moe = graph.find("moe_swiglu_block")
        E, F = config.num_experts, config.moe_intermediate_size
        if moe.inputs[1] != (E, H, F):
            raise MoETransformerDimError(
                f"w_gate {moe.inputs[1]} != (E, H, F)=({E}, {H}, {F})")
        if moe.inputs[3] != (E, F, H):
            raise MoETransformerDimError(
                f"w_down {moe.inputs[3]} != (E, F, H)=({E}, {F}, {H})")

    if graph.nodes[0].inputs[0][-1] != H:
        raise MoETransformerDimError("block input width must be hidden_size")
    if graph.nodes[-1].output[-1] != H:
        raise MoETransformerDimError("block output width must be hidden_size")


def estimated_param_counts(config: MoETransformerConfig) -> dict:
    """Approximate total / per-token-active parameter counts (ignores biases)."""
    H, L, E = config.hidden_size, config.num_layers, config.num_experts
    F, Fs = config.moe_intermediate_size, config.shared_expert_intermediate_size
    k = config.num_experts_per_tok

    embed = config.vocab_size * H
    if config.attn_kind == "mla":
        R = config.kv_lora_rank
        attn = H * config.attn_dim + H * R + R * config.attn_dim + R * config.value_dim + config.value_dim * H
    else:
        attn = H * config.attn_dim + H * config.kv_dim + H * config.kv_value_dim + config.value_dim * H
    router = H * E
    expert_one = 3 * H * F
    shared = (3 * H * Fs) if config.num_shared_experts > 0 else 0
    norms = 2 * H
    dense_one = 3 * H * (config.dense_intermediate_size or Fs)

    moe_layers = max(0, L - config.first_k_dense)
    dense_layers = config.first_k_dense

    per_moe_total = attn + router + E * expert_one + shared + norms
    per_moe_active = attn + router + k * expert_one + shared + norms
    per_dense = attn + dense_one + norms

    total = embed + moe_layers * per_moe_total + dense_layers * per_dense + H
    active = embed + moe_layers * per_moe_active + dense_layers * per_dense + H

    bits = {"int4": 4, "fp8_e4m3": 8, "fp8_e5m2": 8}.get(config.weight_dtype or "", 16)
    return {
        "total": total, "active": active,
        "total_b": round(total / 1e9, 2), "active_b": round(active / 1e9, 2),
        "weight_bits": bits, "weight_gb": round(total * bits / 8 / 1e9, 1),
    }


def verify_param_budget(config: MoETransformerConfig, *, rel_tol: float = 0.15) -> None:
    """Reject a config whose estimated total/active miss its declared budget.

    Skipped when the config declares no budget (``total_params_b == 0``).
    """
    if config.total_params_b <= 0:
        return
    est = estimated_param_counts(config)
    for name, target in (("total_b", config.total_params_b),
                         ("active_b", config.active_params_b)):
        got = est[name]
        if target > 0 and abs(got - target) > rel_tol * target:
            raise MoETransformerDimError(
                f"{config.name} param budget miss: {name}={got}B vs target {target}B "
                f"(rel_tol={rel_tol:.0%})")


__all__ = [
    "MoETransformerConfig",
    "MoETransformerDimError",
    "SharedTopKIndexGroup",
    "BlockGraph",
    "build_block",
    "verify_config",
    "verify_block",
    "shared_topk_index_groups",
    "deterministic_topk_indices",
    "estimated_param_counts",
    "verify_param_budget",
    "ATTENTION_KINDS",
    "SPARSE_KINDS",
    "QUANT_DTYPES",
    "MSA_SCORE_TYPES",
]
