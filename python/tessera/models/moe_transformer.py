"""Shared MoE-transformer contract layer for the model-class roadmap.

Kimi-K2, DeepSeek-V3.2, and GLM-5 are the *same* architecture family —
MoE-FFN transformers with latent (MLA) or grouped (GQA) attention, optional
DeepSeek-style sparse attention (DSA), and low-precision (INT4 / FP8) weights.
Rather than three near-duplicate graphs, this module is the shared, shape-only
contract: one :class:`MoETransformerConfig`, one config verifier, one
block-graph builder, and a parameter-budget estimator.  The per-model modules
(``deepseek_v32`` / ``glm5`` / ``kimi_k2``) are thin config factories on top.

This is the **M0 scaffolding**: it builds and verifies the op graph (the
vocabulary the M3/M4 attention kernels and M1/M2 quant+MoE lowering must
implement) without execution.  Reuses the :class:`GraphNode` / ``GraphNodeList``
representation style of :mod:`diffusion_gemma`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .diffusion_gemma import GraphNode


class MoETransformerDimError(ValueError):
    """Config / built-graph dimension contract violation."""


ATTENTION_KINDS = ("mla", "gqa")
SPARSE_KINDS = (None, "dsa", "lsa")
QUANT_DTYPES = (None, "int4", "fp8_e4m3", "fp8_e5m2")


@dataclass(frozen=True)
class MoETransformerConfig:
    """One MoE-transformer model's shape contract.

    Attention is ``"mla"`` (multi-head latent — compress to ``kv_lora_rank``,
    cache the latent, expand to K/V; a decoupled ``rope_head_dim`` carries
    position) or ``"gqa"`` (``num_attention_heads`` query heads share
    ``num_kv_heads``).  ``sparse="dsa"`` adds DeepSeek sparse attention (top-k
    block selection); ``sparse="lsa"`` adds Lookahead Sparse Attention (local
    window ∪ threshold-selected past blocks).  ``weight_dtype`` selects the packed quant scheme the
    expert/dense GEMMs lower through (``stdlib.quant``).  ``first_k_dense`` leading
    layers use a plain FFN before the MoE layers begin (DeepSeek convention).
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
    # MLA latent ranks (attn_kind == "mla")
    q_lora_rank: int = 0          # 0 → no query down-projection
    kv_lora_rank: int = 512
    rope_head_dim: int = 64       # decoupled-RoPE head dim (partial RoPE)
    rope_variant: str = "rope"

    # ── sparse / hybrid attention ─────────────────────────────────────────────
    sparse: str | None = None     # None | "dsa" (DeepSeek block-sparse) | "lsa"
    dsa_top_k_blocks: int = 0
    dsa_block_size: int = 64       # block size for DSA *and* LSA
    sliding_window: int = 0       # 0 → dense; >0 → sliding-window layers
    layer_types: tuple[str, ...] = ("full",)
    # Lookahead Sparse Attention (sparse == "lsa"): local window ∪ threshold-
    # selected strictly-past blocks (block size = dsa_block_size).
    lsa_window_size: int = 0
    lsa_threshold: float = 0.5

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

    @property
    def attn_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def is_moe_layer(self, layer_index: int) -> bool:
        return layer_index >= self.first_k_dense

    def attention_mode(self, layer_index: int) -> str:
        if not self.layer_types:
            return "full"
        return self.layer_types[layer_index % len(self.layer_types)]


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
        if config.rope_head_dim < 0 or config.rope_head_dim > config.head_dim:
            raise MoETransformerDimError(
                f"rope_head_dim={config.rope_head_dim} must be in [0, head_dim={config.head_dim}]")

    if config.sparse == "dsa":
        if config.dsa_top_k_blocks <= 0 or config.dsa_block_size <= 0:
            raise MoETransformerDimError(
                "DSA requires dsa_top_k_blocks > 0 and dsa_block_size > 0")
    if config.sparse == "lsa":
        if config.lsa_window_size <= 0 or config.dsa_block_size <= 0:
            raise MoETransformerDimError(
                "LSA requires lsa_window_size > 0 and dsa_block_size > 0")
        if not (0.0 <= config.lsa_threshold <= 1.0):
            raise MoETransformerDimError("lsa_threshold must be in [0, 1]")

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
        # compress hidden → latent, cache latent, expand to K/V (the MLA chain
        # the M3 FlashMLA kernel will absorb).
        add("latent_kv_compress", [(T, H), (H, R)], (T, R), kv_lora_rank=R)
        add("q_proj", [(T, H), (H, config.attn_dim)], (T, config.attn_dim))
        add("latent_kv_expand_k", [(T, R), (R, config.attn_dim)], (T, config.attn_dim))
        add("latent_kv_expand_v", [(T, R), (R, config.attn_dim)], (T, config.attn_dim))
        kv_dim = config.attn_dim
        unified_kv = True
    else:  # gqa
        kv_dim = config.kv_dim
        add("q_proj", [(T, H), (H, config.attn_dim)], (T, config.attn_dim))
        add("k_proj", [(T, H), (H, kv_dim)], (T, kv_dim))
        add("v_proj", [(T, H), (H, kv_dim)], (T, kv_dim))
        unified_kv = False

    # decoupled / partial RoPE on the rope_head_dim slice (MLA) or full (GQA)
    add("rope", [(T, config.attn_dim)], (T, config.attn_dim),
        applies_to="q", variant=config.rope_variant,
        rope_head_dim=(config.rope_head_dim if config.attn_kind == "mla" else config.head_dim))

    attn_op = {"dsa": "deepseek_sparse_attention",
               "lsa": "lookahead_sparse_attention"}.get(sparse or "", "attention")
    attn_attrs = dict(
        mode=mode, num_heads=config.num_attention_heads,
        num_kv_heads=(config.num_attention_heads if unified_kv else config.num_kv_heads),
        head_dim=config.head_dim, attn_kind=config.attn_kind,
        sliding_window=(config.sliding_window if mode == "sliding" else None),
        unified_kv=unified_kv, rope_variant=config.rope_variant)
    if sparse == "dsa":
        attn_attrs.update(top_k_blocks=config.dsa_top_k_blocks,
                          block_size=config.dsa_block_size)
    elif sparse == "lsa":
        attn_attrs.update(window_size=config.lsa_window_size,
                          block_size=config.dsa_block_size,
                          threshold=config.lsa_threshold)
    add(attn_op,
        [(T, config.attn_dim), (T, kv_dim), (T, kv_dim)],
        (T, config.attn_dim), **attn_attrs)
    add("o_proj", [(T, config.attn_dim), (config.attn_dim, H)], (T, H))
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
    else:
        k = graph.find("k_proj")
        if k.output[-1] != config.kv_dim:
            raise MoETransformerDimError(
                f"k_proj out width {k.output[-1]} != kv_dim={config.kv_dim}")

    if config.sparse == "dsa":
        attn = graph.find("deepseek_sparse_attention")
        if int(attn.attrs.get("top_k_blocks", 0)) != config.dsa_top_k_blocks:
            raise MoETransformerDimError("DSA top_k_blocks attr mismatch")
    if config.sparse == "lsa":
        attn = graph.find("lookahead_sparse_attention")
        if int(attn.attrs.get("window_size", 0)) != config.lsa_window_size:
            raise MoETransformerDimError("LSA window_size attr mismatch")

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
        attn = H * config.attn_dim + H * R + 2 * R * config.attn_dim + config.attn_dim * H
    else:
        attn = H * config.attn_dim + 2 * H * config.kv_dim + config.attn_dim * H
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

    Skipped when the config declares no budget (``total_params_b == 0``), as the
    placeholder configs (e.g. GLM-5, whose dims are unconfirmed) do.
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
    "BlockGraph",
    "build_block",
    "verify_config",
    "verify_block",
    "estimated_param_counts",
    "verify_param_budget",
    "ATTENTION_KINDS",
    "SPARSE_KINDS",
    "QUANT_DTYPES",
]
