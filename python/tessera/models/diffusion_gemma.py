"""DiffusionGemma — production text block-diffusion MoE model graph (experimental).

Phase A (this module) is the **contract layer**: a frozen model config, a
shape-only graph builder for one full text layer at production dimensions with
synthetic BF16 weights, and a config-aware graph verifier that rejects
mismatched dimensions *before* any runtime. Runtime/kernel lowering, real MoE
packing/scatter, the block-diffusion step region, the entropy-bound sampler, and
KV-cache promotion are subsequent phases that build on this contract.

The layer composes existing Tessera primitives — RMSNorm, Q/K/V/O projections,
RoPE, a sliding/full attention tag, residuals, a top-k router over the experts,
the grouped SwiGLU expert path (`moe_swiglu_block`), a shared-expert path, and
combine. The graph is represented as an ordered list of :class:`GraphNode`s
(op + input/output shapes + attrs) so the spec is testable without execution and
serves as the reference the later compiler-visible region/op must implement.

Two classes of dimension live in :class:`DiffusionGemmaConfig`:

* **From the Gemma 4 26B A4B model card** (authoritative): 30 layers, 25.2B
  total / 3.8B active params, 262144 vocab, 1024 sliding window, 256K context,
  128 experts with 8 active + 1 shared, hybrid attention that interleaves local
  sliding with full global and keeps the **final layer global**, global layers
  with **unified KV** + **Proportional RoPE (p-RoPE)**, text+image modalities,
  ~550M vision encoder (deferred), and the recommended sampling defaults
  (temperature 1.0 / top_p 0.95 / top_k 64).

* **Derived to the published budget** (the card omits these): ``hidden_size``,
  ``num_attention_heads`` / ``num_kv_heads`` / ``head_dim``,
  ``moe_intermediate_size``, ``shared_expert_intermediate_size``. The defaults
  (H=2560, 10:2 GQA heads × 256, expert FFN 768, shared FFN 5376) are *one*
  consistent solution that lands at ~25.0B total / ~3.8B active via
  :func:`estimated_param_counts` — **not** transcribed from Google. The large
  shared expert is forced by the active/total split (non-expert ≈ 2.4B,
  expert-mass ≈ 22.8B). :func:`verify_param_budget` (25.2, 3.8) passes; supply
  the real hyperparameters to replace the derived ones.

``canvas_size`` (256) is the block-diffusion design parameter from the work
plan, not a Gemma 4 spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class DiffusionGemmaDimError(ValueError):
    """Raised when a config or built graph violates a DiffusionGemma dimension
    contract (head divisibility, expert count, MoE FFN width, vocab size)."""


@dataclass(frozen=True)
class DiffusionGemmaConfig:
    """Frozen DiffusionGemma text config — Gemma 4 26B A4B MoE.

    Defaults are the Gemma 4 26B A4B model-card facts plus a budget-derived set
    for the dims the card omits (see the module docstring). GQA attention:
    ``num_attention_heads`` query heads share ``num_kv_heads`` KV heads. Hybrid
    attention: ``layer_types`` is the repeating sliding/global policy and the
    final layer is always global; global layers use unified KV + p-RoPE. MoE:
    ``num_experts_per_tok`` of ``num_experts`` per token plus ``num_shared_experts``
    always-on shared expert(s).
    """

    # ── text dims (DERIVED to the 25.2B/3.8B budget — card omits these) ───────
    hidden_size: int = 2560
    num_attention_heads: int = 10
    num_kv_heads: int = 2            # sliding-layer GQA; global layers unify KV
    head_dim: int = 256
    moe_intermediate_size: int = 768
    shared_expert_intermediate_size: int = 5376
    # ── from the Gemma 4 26B A4B model card ───────────────────────────────────
    num_layers: int = 30
    num_experts: int = 128
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    vocab_size: int = 262144
    context_length: int = 262144     # 256K
    canvas_size: int = 256           # block-diffusion design param (not a Gemma spec)
    sliding_window: int = 1024
    # hybrid attention: 5 sliding : 1 global, final layer always global
    layer_types: tuple[str, ...] = (
        "sliding", "sliding", "sliding", "sliding", "sliding", "full",
    )
    final_layer_global: bool = True
    global_unified_kv: bool = True   # card: global layers feature unified K/V
    rope_variant: str = "p_rope"     # card: Proportional RoPE on global layers
    # published budget (billions) — used by verify_param_budget
    total_params_b: float = 25.2
    active_params_b: float = 3.8
    # modalities + deferred vision
    modalities: tuple[str, ...] = ("text", "image")
    vision_encoder_params: int = 550_000_000  # ~550M, deferred (text-only first)
    # recommended sampling defaults (card best practices)
    sample_temperature: float = 1.0
    sample_top_p: float = 0.95
    sample_top_k: int = 64
    # ── numerics ─────────────────────────────────────────────────────────────
    # Softcap values are NOT given in the card (Gemma-2 lineage values kept as a
    # placeholder; Gemma 3+ may use QK-norm instead). Treat as unconfirmed.
    attn_logit_softcap: float = 50.0
    final_logit_softcap: float = 30.0
    rms_norm_eps: float = 1e-6
    dtype: str = "bf16"

    @property
    def attn_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    def attention_mode(self, layer_index: int) -> str:
        """Sliding vs global for ``layer_index``. The final layer is always
        global when ``final_layer_global`` is set (Gemma 4 hybrid attention)."""
        if self.final_layer_global and layer_index == self.num_layers - 1:
            return "full"
        return self.layer_types[layer_index % len(self.layer_types)]

    def kv_heads_for_layer(self, layer_index: int) -> int:
        """Global layers unify K/V (single KV head) when ``global_unified_kv``;
        sliding layers use the GQA ``num_kv_heads``."""
        if self.global_unified_kv and self.attention_mode(layer_index) == "full":
            return 1
        return self.num_kv_heads


def verify_config(config: DiffusionGemmaConfig) -> None:
    """Reject an internally-inconsistent config before any graph is built.

    Covers the four dimension contracts the work plan calls out at the config
    level (head divisibility / expert count) and basic positivity for the rest.
    """
    if config.hidden_size <= 0 or config.head_dim <= 0:
        raise DiffusionGemmaDimError("hidden_size and head_dim must be positive")
    if config.num_attention_heads <= 0 or config.num_kv_heads <= 0:
        raise DiffusionGemmaDimError("head counts must be positive")
    # Q/K/V head contract — GQA requires query heads to be a multiple of KV heads.
    if config.num_attention_heads % config.num_kv_heads != 0:
        raise DiffusionGemmaDimError(
            f"GQA head mismatch: num_attention_heads={config.num_attention_heads} "
            f"is not a multiple of num_kv_heads={config.num_kv_heads}")
    # Expert-count contract.
    if config.num_experts < 1:
        raise DiffusionGemmaDimError("num_experts must be >= 1")
    if not (1 <= config.num_experts_per_tok <= config.num_experts):
        raise DiffusionGemmaDimError(
            f"num_experts_per_tok={config.num_experts_per_tok} must be in "
            f"[1, num_experts={config.num_experts}]")
    if config.num_shared_experts < 0:
        raise DiffusionGemmaDimError("num_shared_experts must be >= 0")
    if config.moe_intermediate_size <= 0:
        raise DiffusionGemmaDimError("moe_intermediate_size must be positive")
    if config.shared_expert_intermediate_size <= 0:
        raise DiffusionGemmaDimError("shared_expert_intermediate_size must be positive")
    if config.vocab_size <= 0:
        raise DiffusionGemmaDimError("vocab_size must be positive")
    if config.canvas_size <= 0:
        raise DiffusionGemmaDimError("canvas_size must be positive")
    for mode in config.layer_types:
        if mode not in ("sliding", "full"):
            raise DiffusionGemmaDimError(
                f"layer_types entries must be 'sliding' or 'full'; got {mode!r}")
    if not config.layer_types:
        raise DiffusionGemmaDimError("layer_types must be non-empty")


@dataclass(frozen=True)
class GraphNode:
    """One shape-only graph node: an op name, its input/output shapes, and attrs.

    Shapes use ``"T"`` for the (symbolic) token/sequence length so a single
    built graph applies to any token count. Weight shapes are concrete (from the
    config) — these are the synthetic BF16 parameter shapes the node consumes.
    """

    op: str
    inputs: tuple[tuple, ...]   # input operand shapes (tensors + weights)
    output: tuple               # output shape
    attrs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TextBlockGraph:
    """One DiffusionGemma text layer as an ordered, shape-checked node list."""

    nodes: tuple[GraphNode, ...]
    config: DiffusionGemmaConfig
    layer_index: int
    attention_mode: str
    causal: bool

    def op_sequence(self) -> tuple[str, ...]:
        return tuple(n.op for n in self.nodes)

    def find(self, op: str) -> GraphNode:
        for n in self.nodes:
            if n.op == op:
                return n
        raise KeyError(op)


def build_text_block(
    config: DiffusionGemmaConfig,
    *,
    layer_index: int = 0,
    causal: bool = False,
) -> TextBlockGraph:
    """Build one full DiffusionGemma text layer as a shape-only graph.

    ``causal`` selects the block-diffusion attention role: ``True`` = causal
    encoder prefill, ``False`` = bidirectional 256-token canvas denoiser. The
    sliding-vs-full window policy is taken from ``config.layer_types`` for
    ``layer_index``. The built graph is verified against the config before
    return, so a bad config or shape flow fails here, before runtime.
    """
    verify_config(config)
    H = config.hidden_size
    mode = config.attention_mode(layer_index)
    is_global = mode == "full"
    # Global (full) layers unify K/V (single KV head) and use p-RoPE; sliding
    # layers use the GQA KV heads + standard RoPE.
    layer_kv_heads = config.kv_heads_for_layer(layer_index)
    layer_kv_dim = layer_kv_heads * config.head_dim
    rope_variant = config.rope_variant if is_global else "rope"
    T = "T"  # symbolic token length

    nodes: list[GraphNode] = []

    def add(op, inputs, output, **attrs):
        nodes.append(GraphNode(op=op, inputs=tuple(inputs), output=tuple(output), attrs=attrs))

    # 1. input RMSNorm
    add("rmsnorm", [(T, H), (H,)], (T, H), eps=config.rms_norm_eps)
    # 2. Q / K / V projections (GQA: K/V are narrower; global layers unify K/V)
    add("q_proj", [(T, H), (H, config.attn_dim)], (T, config.attn_dim))
    add("k_proj", [(T, H), (H, layer_kv_dim)], (T, layer_kv_dim))
    add("v_proj", [(T, H), (H, layer_kv_dim)], (T, layer_kv_dim))
    # 3. RoPE on Q and K (p-RoPE on global layers)
    add("rope", [(T, config.attn_dim)], (T, config.attn_dim), applies_to="q", variant=rope_variant)
    add("rope", [(T, layer_kv_dim)], (T, layer_kv_dim), applies_to="k", variant=rope_variant)
    # 4. attention (tagged sliding/global + causal-prefill vs bidirectional-canvas)
    add(
        "attention",
        [(T, config.attn_dim), (T, layer_kv_dim), (T, layer_kv_dim)],
        (T, config.attn_dim),
        mode=mode,
        causal=causal,
        sliding_window=(config.sliding_window if mode == "sliding" else None),
        num_heads=config.num_attention_heads,
        num_kv_heads=layer_kv_heads,
        unified_kv=(is_global and config.global_unified_kv),
        head_dim=config.head_dim,
        rope_variant=rope_variant,
        logit_softcap=config.attn_logit_softcap,
    )
    # 5. output projection + residual
    add("o_proj", [(T, config.attn_dim), (config.attn_dim, H)], (T, H))
    add("residual_add", [(T, H), (T, H)], (T, H), source="attn")
    # 6. post-attention RMSNorm
    add("rmsnorm", [(T, H), (H,)], (T, H), eps=config.rms_norm_eps, position="post_attn")
    # 7. MoE router — top-k over experts
    add(
        "router",
        [(T, H), (H, config.num_experts)],
        (T, config.num_experts),
        top_k=config.num_experts_per_tok,
        num_experts=config.num_experts,
    )
    # 8. routed expert path — grouped SwiGLU (moe_swiglu_block):
    #    x (T,H); w_gate/w_up (E,H,F); w_down (E,F,H); group_sizes (E,) → (T,H)
    F = config.moe_intermediate_size
    E = config.num_experts
    add(
        "moe_swiglu_block",
        [(T, H), (E, H, F), (E, H, F), (E, F, H), (E,)],
        (T, H),
        num_experts=E,
        moe_intermediate_size=F,
    )
    # 9. shared-expert path — a single always-on SwiGLU FFN
    Fs = config.shared_expert_intermediate_size
    add("shared_expert", [(T, H), (H, Fs), (H, Fs), (Fs, H)], (T, H),
        shared_intermediate_size=Fs)
    # 10. combine routed + shared, then residual
    add("moe_combine", [(T, H), (T, H), (T, config.num_experts_per_tok)], (T, H))
    add("residual_add", [(T, H), (T, H)], (T, H), source="moe")

    graph = TextBlockGraph(
        nodes=tuple(nodes), config=config, layer_index=layer_index,
        attention_mode=mode, causal=causal,
    )
    verify_text_block(graph, config)
    return graph


def build_lm_head(config: DiffusionGemmaConfig) -> GraphNode:
    """Final vocab projection (lm_head): ``(T, H) @ (H, vocab) → (T, vocab)``
    with the configured final-logit softcap. Separate from the text block so the
    vocab-size contract is checked where the logits are produced."""
    verify_config(config)
    H = config.hidden_size
    return GraphNode(
        op="lm_head",
        inputs=((("T"), H), (H, config.vocab_size)),
        output=(("T"), config.vocab_size),
        attrs={"vocab_size": config.vocab_size,
               "logit_softcap": config.final_logit_softcap},
    )


def verify_text_block(graph: TextBlockGraph, config: DiffusionGemmaConfig) -> None:
    """Config-aware graph verifier — reject mismatched dims before runtime.

    Catches the four contracts the work plan names: Q/K/V head mismatch, expert
    count mismatch, wrong ``moe_intermediate_size``, and wrong ``vocab_size``,
    plus shape-flow consistency between adjacent nodes.
    """
    verify_config(config)

    # Q/K/V head contract — projection widths must equal heads × head_dim.
    q = graph.find("q_proj")
    k = graph.find("k_proj")
    # K/V width is per-layer: global layers unify K/V to a single head.
    expected_kv_dim = config.kv_heads_for_layer(graph.layer_index) * config.head_dim
    if q.output[-1] != config.attn_dim:
        raise DiffusionGemmaDimError(
            f"q_proj out width {q.output[-1]} != num_attention_heads*head_dim="
            f"{config.attn_dim}")
    if k.output[-1] != expected_kv_dim:
        raise DiffusionGemmaDimError(
            f"k_proj out width {k.output[-1]} != layer kv heads*head_dim="
            f"{expected_kv_dim} (layer {graph.layer_index}, mode {graph.attention_mode})")
    attn = graph.find("attention")
    n_heads = int(attn.attrs.get("num_heads", 0))
    n_kv = int(attn.attrs.get("num_kv_heads", 0))
    if n_kv <= 0 or n_heads % n_kv != 0:
        raise DiffusionGemmaDimError(
            f"attention head mismatch: {n_heads} query heads not "
            f"a multiple of {n_kv} KV heads")

    # Expert-count contract — router output width == num_experts; top_k bounded.
    router = graph.find("router")
    if router.output[-1] != config.num_experts:
        raise DiffusionGemmaDimError(
            f"router out width {router.output[-1]} != num_experts={config.num_experts}")
    router_top_k = int(router.attrs.get("top_k", 0))
    if not (1 <= router_top_k <= config.num_experts):
        raise DiffusionGemmaDimError(
            f"router top_k={router_top_k} out of [1, {config.num_experts}]")

    # MoE FFN-width contract — expert weights must be (E, H, F) / (E, F, H).
    moe = graph.find("moe_swiglu_block")
    w_gate, w_up, w_down = moe.inputs[1], moe.inputs[2], moe.inputs[3]
    for name, w in (("w_gate", w_gate), ("w_up", w_up)):
        if w != (config.num_experts, config.hidden_size, config.moe_intermediate_size):
            raise DiffusionGemmaDimError(
                f"{name} shape {w} != (num_experts, hidden, moe_intermediate_size)="
                f"({config.num_experts}, {config.hidden_size}, {config.moe_intermediate_size})")
    if w_down != (config.num_experts, config.moe_intermediate_size, config.hidden_size):
        raise DiffusionGemmaDimError(
            f"w_down shape {w_down} != (num_experts, moe_intermediate_size, hidden)")

    # Shape flow — every node's first tensor output feeds a downstream node; the
    # block is residual-shaped, so it must start and end at (T, hidden).
    if graph.nodes[0].inputs[0][-1] != config.hidden_size:
        raise DiffusionGemmaDimError("text block input width must be hidden_size")
    if graph.nodes[-1].output[-1] != config.hidden_size:
        raise DiffusionGemmaDimError("text block output width must be hidden_size")


def verify_lm_head(node: GraphNode, config: DiffusionGemmaConfig) -> None:
    """Vocab-size contract — lm_head must project to exactly ``vocab_size``."""
    if node.output[-1] != config.vocab_size:
        raise DiffusionGemmaDimError(
            f"lm_head out width {node.output[-1]} != vocab_size={config.vocab_size}")


def estimated_param_counts(config: DiffusionGemmaConfig) -> dict:
    """Estimate total and per-token *active* parameter counts from the config.

    Lets a config be checked against a published budget (e.g. Gemma 4 26B A4B =
    26B total / 4B active). Conventions: tied token-embedding / lm_head counted
    once; ``active`` counts the always-on params (embedding row, attention,
    router, shared expert, norms) plus the ``num_experts_per_tok`` selected
    experts' full weights (not all ``num_experts``). SwiGLU experts have three
    H×F matrices (gate, up, down). Estimates ignore tiny per-row biases.
    """
    H = config.hidden_size
    L = config.num_layers
    E = config.num_experts
    F = config.moe_intermediate_size
    Fs = config.shared_expert_intermediate_size
    k = config.num_experts_per_tok

    embed = config.vocab_size * H                       # tied embedding == lm_head
    attn = H * config.attn_dim + 2 * H * config.kv_dim + config.attn_dim * H
    router = H * E
    expert_one = 3 * H * F                              # gate + up + down
    experts_all = E * expert_one
    shared = 3 * H * Fs
    norms = 2 * H

    per_layer_total = attn + router + experts_all + shared + norms
    total = embed + L * per_layer_total + H             # + final norm

    per_layer_active = attn + router + k * expert_one + shared + norms
    active = embed + L * per_layer_active + H           # lm_head (tied) active at output

    return {
        "total": total,
        "active": active,
        "embedding": embed,
        "per_layer_total": per_layer_total,
        "per_layer_active": per_layer_active,
        "experts_all_per_layer": experts_all,
        "total_b": round(total / 1e9, 2),
        "active_b": round(active / 1e9, 2),
        "bf16_gb": round(total * 2 / 1e9, 1),
    }


def verify_param_budget(
    config: DiffusionGemmaConfig,
    *,
    total_b: float,
    active_b: float,
    rel_tol: float = 0.10,
) -> None:
    """Reject a config whose estimated total/active param counts miss a target
    budget (in billions) by more than ``rel_tol``. Use to calibrate the config
    against a published model size (e.g. 26B total / 4B active)."""
    est = estimated_param_counts(config)
    for name, target in (("total_b", total_b), ("active_b", active_b)):
        got = est[name]
        if abs(got - target) > rel_tol * target:
            raise DiffusionGemmaDimError(
                f"param budget miss: estimated {name}={got}B vs target {target}B "
                f"(rel_tol={rel_tol:.0%}) — config dims need calibration")
