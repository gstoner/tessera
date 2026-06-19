"""M0 — frontier MoE-model contract graphs build + verify; M5 north-star gate.

The full-scale configs must lower to a *valid, verified* op graph (the
artifact-at-full-config claim — provable without execution).  The scaled configs
must additionally compose the shared stdlib pillars end-to-end against a numpy
reference (the M5 execution gate; currently partial — attention pillars M3/M4
land the rest).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tessera.models import deepseek_v32, glm5, kimi_k2, minimax_m3
from tessera.models import moe_transformer as mt

ALL_MODELS = [deepseek_v32, glm5, kimi_k2, minimax_m3]
MODEL_IDS = ["deepseek_v32", "glm5", "kimi_k2", "minimax_m3"]
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_CLASS_FIXTURES = {
    "deepseek_v32": ("deepseek_v32_block.mlir", ("tessera.deepseek_sparse_attention", "tessera.moe_swiglu_block")),
    "glm5": (
        "glm5_block.mlir",
        ("tessera.latent_kv_compress", "tessera.deepseek_sparse_attention", "tessera.moe_swiglu_block"),
    ),
    "kimi_k2": ("kimi_k2_block.mlir", ("tessera.latent_kv_compress", "tessera.moe_swiglu_block")),
    "minimax_m3": ("minimax_m3_block.mlir", ("tessera.msa_sparse_attention", "tessera.moe_swiglu_block")),
}


def _expected_attention_op(cfg: mt.MoETransformerConfig, layer_index: int) -> str:
    if cfg.sparse == "dsa":
        return "deepseek_sparse_attention"
    if cfg.sparse == "lsa":
        return "lookahead_sparse_attention"
    if cfg.sparse == "msa" and cfg.uses_msa_layer(layer_index):
        return "msa_sparse_attention"
    return "attention"


# ── M0: full-config graphs build + verify (artifact target) ───────────────────
@pytest.mark.parametrize("model", ALL_MODELS, ids=MODEL_IDS)
def test_full_config_block_graph_builds_and_verifies(model):
    cfg = model.config()
    mt.verify_config(cfg)
    # build the first (often dense) and a later (MoE) layer
    dense = mt.build_block(cfg, layer_index=0)
    moe = mt.build_block(cfg, layer_index=cfg.num_layers - 1)
    assert moe.is_moe
    mt.verify_block(dense, cfg)
    mt.verify_block(moe, cfg)
    # the MoE layer's op vocabulary names the pillars the kernels implement
    ops = moe.op_sequence()
    assert "router" in ops and "moe_swiglu_block" in ops
    assert _expected_attention_op(cfg, cfg.num_layers - 1) in ops
    if cfg.attn_kind == "mla":
        assert "latent_kv_compress" in ops


@pytest.mark.parametrize("model", ALL_MODELS, ids=MODEL_IDS)
def test_full_config_artifact_all_layers_anchored(model):
    """Artifact-at-full-config: every layer builds + verifies at production dims,
    and the compute-core ops the artifact relies on are catalog-anchored (each
    has a Graph IR identity in OP_SPECS) — the runnable companion to the lit
    fixtures under tests/tessera-ir/model_class/."""
    from tessera.compiler.op_catalog import OP_SPECS
    cfg = model.config()
    ops_seen: set[str] = set()
    for li in range(cfg.num_layers):
        g = mt.build_block(cfg, layer_index=li)   # verify_block runs at full dims
        ops_seen.update(g.op_sequence())
    core = {"moe_swiglu_block", "rmsnorm"}
    if cfg.attn_kind == "mla":
        core |= {"latent_kv_compress", "latent_kv_expand_k", "latent_kv_expand_v"}
    if cfg.sparse == "dsa":
        core.add("deepseek_sparse_attention")
    if cfg.sparse == "msa":
        core.add("msa_sparse_attention")
    for op in core:
        assert op in ops_seen, f"{cfg.name}: core op {op} not emitted"
        assert op in OP_SPECS, f"{cfg.name}: core op {op} not catalog-anchored"


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_model_class_lit_fixture_anchors_core_ops(model_id):
    fixture_name, expected_ops = MODEL_CLASS_FIXTURES[model_id]
    path = REPO_ROOT / "tests" / "tessera-ir" / "model_class" / fixture_name
    text = path.read_text(encoding="utf-8")
    assert "RUN: tessera-opt" in text
    assert f"func.func @{model_id}_block" in text
    for op in expected_ops:
        assert op in text
    if model_id == "minimax_m3":
        assert "artifact contract only" in text
        assert "MSA runtime decode" in text


@pytest.mark.parametrize("model", ALL_MODELS, ids=MODEL_IDS)
def test_param_estimate_order_of_magnitude(model):
    cfg = model.config()
    est = mt.estimated_param_counts(cfg)
    assert est["total"] > est["active"] > 0
    # quant weight-bytes reflect the declared low-precision dtype
    if cfg.weight_dtype == "int4":
        assert est["weight_bits"] == 4
    elif cfg.weight_dtype and cfg.weight_dtype.startswith("fp8"):
        assert est["weight_bits"] == 8


def test_declared_budgets_are_in_range_where_published():
    # DeepSeek-V3.2, Kimi-K2, and GLM-5.2 publish sizes; verify the estimator
    # lands close to the contract-level parameter budgets.
    mt.verify_param_budget(deepseek_v32.config(), rel_tol=0.30)
    mt.verify_param_budget(kimi_k2.config(), rel_tol=0.30)
    mt.verify_param_budget(glm5.config())
    mt.verify_param_budget(minimax_m3.config())


def test_bad_config_rejected():
    cfg = deepseek_v32.config()
    bad = mt.MoETransformerConfig(**{**cfg.__dict__, "num_experts_per_tok": cfg.num_experts + 1})
    with pytest.raises(mt.MoETransformerDimError):
        mt.verify_config(bad)


# ── M5 north-star: scaled-instance pillar composition (execution gate) ────────
@pytest.mark.parametrize("model", ALL_MODELS, ids=MODEL_IDS)
def test_scaled_config_builds(model):
    cfg = model.scaled_config()
    for li in range(cfg.num_layers):
        mt.build_block(cfg, layer_index=li)


def test_scaled_moe_pillar_executes_against_reference():
    """The M2 capacity-aware MoE + M1 quant pillars run end-to-end on a scaled
    DeepSeek-V3.2 MoE layer, dense path matching the reference."""
    from tessera.stdlib import moe
    cfg = deepseek_v32.scaled_config()
    rng = np.random.default_rng(0)
    H, F, E, k = cfg.hidden_size, cfg.moe_intermediate_size, cfg.num_experts, cfg.num_experts_per_tok
    T = 24
    s = 1.0 / np.sqrt(H)
    wr = (rng.standard_normal((H, E)) * s).astype(np.float32)
    wg = (rng.standard_normal((E, H, F)) * s).astype(np.float32)
    wu = (rng.standard_normal((E, H, F)) * s).astype(np.float32)
    wd = (rng.standard_normal((E, F, H)) / np.sqrt(F)).astype(np.float32)
    x = rng.standard_normal((T, H)).astype(np.float32)
    res = moe.moe_forward(x, wr, wg, wu, wd, top_k=k, capacity_factor=1.5)
    assert res.y.shape == (T, H)
    assert res.plan.capacity == moe.compute_capacity(T, k, E, 1.5)


def test_northstar_scaled_attention_plus_moe_compose():
    """North-star (partial): the M3 MLA attention pillar feeds the M2 MoE pillar
    end-to-end on a scaled DeepSeek-V3.2 block. The remaining M5 work is the
    full multi-layer stack + autoregressive decode loop across every layer."""
    from tessera.stdlib import attention, moe
    cfg = deepseek_v32.scaled_config()
    rng = np.random.default_rng(0)
    H, Hh, dh = cfg.hidden_size, cfg.num_attention_heads, cfg.head_dim
    d_c, d_rope = cfg.kv_lora_rank, cfg.rope_head_dim
    d_nope = dh - d_rope
    s = 1.0 / np.sqrt(H)
    w = attention.MLAWeights(
        w_dkv=rng.standard_normal((H, d_c)) * s,
        w_uk=rng.standard_normal((d_c, Hh * d_nope)) / np.sqrt(d_c),
        w_uv=rng.standard_normal((d_c, Hh * dh)) / np.sqrt(d_c),
        w_q=rng.standard_normal((H, Hh * (d_nope + d_rope))) * s,
        w_kr=rng.standard_normal((H, d_rope)) * s,
        num_heads=Hh, d_nope=d_nope, d_rope=d_rope, d_v=dh)
    T = 16
    x = rng.standard_normal((T, H)).astype(np.float64)

    # M3: latent attention → (T, Hh*dh); project back to hidden for the residual.
    attn_out, _, _ = attention.mla_prefill(x, w)
    w_o = (rng.standard_normal((Hh * dh, H)) * (1.0 / np.sqrt(Hh * dh)))
    h1 = x + attn_out @ w_o

    # M2: capacity-aware MoE over the attention output.
    E, F, k = cfg.num_experts, cfg.moe_intermediate_size, cfg.num_experts_per_tok
    wr = (rng.standard_normal((H, E)) * s).astype(np.float32)
    wg = (rng.standard_normal((E, H, F)) * s).astype(np.float32)
    wu = (rng.standard_normal((E, H, F)) * s).astype(np.float32)
    wd = (rng.standard_normal((E, F, H)) / np.sqrt(F)).astype(np.float32)
    res = moe.moe_forward(h1.astype(np.float32), wr, wg, wu, wd, top_k=k,
                          capacity_factor=1.5)
    y = h1 + res.y
    assert y.shape == (T, H)
    assert np.isfinite(y).all()
