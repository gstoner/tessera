"""AR-DECODE-PROGRAM-1 slice S0 — the decode-step contract, fixture and oracle.

This module is the declared Decision #31(a) **oracle** for the compiled
autoregressive decode program planned in
``docs/audit/compiler/AR_DECODE_PROGRAM_PLAN.md``. It is not a production
route: it never lowers, never dispatches, and no backend may consult it to
decide what a decode step means. Each compiled slice (S1..S7) is accepted by
differential comparison against :func:`reference_decode`.

Three things live here:

* :class:`DecodeStepContract` — every *semantic* key a compiled decode step
  needs. All fields are required (no defaults, Decision #21a): a missing or
  invalid key raises :class:`ARDecodeContractError` with the registered code
  ``AR_DECODE_CONTRACT_INVALID`` and names the field. The envelope is the S0
  scope: dense GQA, full attention, greedy decoding, fp32 accumulation.
* The frozen tiny fixture — :data:`AR_DECODE_TINY`, :func:`tiny_fixture_weights`
  and :data:`TINY_PROMPT`. Weights come from a counter-based integer hash, not
  ``numpy.random``, so every fleet host (and every device test that uploads
  them) reproduces the same bytes regardless of numpy version; a digest pins it.
* :func:`reference_decode` / :func:`recompute_decode` — KV-cached greedy decode
  and full-recompute decode over the fp64 reference runtime in
  :mod:`tessera.models.moe_transformer_runtime`, plus the paged KV layout a
  device program must reproduce (:func:`paged_kv_pages`).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import math
from typing import Sequence

import numpy as np

from .moe_transformer import MoETransformerConfig, verify_config
from . import moe_transformer_runtime as _rt

CONTRACT_CODE = "AR_DECODE_CONTRACT_INVALID"

# S0 envelope. Widening any of these is a later slice's job, with its own proof.
STORAGE_DTYPES = ("fp32", "fp16", "bf16")
ACCUM_DTYPES = ("fp32",)
ROPE_VARIANTS = ("rope",)          # rotate-half RoPE, `stdlib.attention.apply_rope`
SAMPLING_MODES = ("greedy",)
TIE_BREAKS = ("first_occurrence",)  # numpy argmax semantics: lowest index wins

# Acceptance tolerance of a compiled step's logits against the fp64 oracle,
# keyed by storage dtype (accumulation is fp32 throughout). Token equality is
# the secondary gate; these bound the primary one.
LOGIT_TOLERANCE = {
    "fp32": {"rtol": 1e-4, "atol": 1e-4},
    "fp16": {"rtol": 2e-2, "atol": 2e-2},
    "bf16": {"rtol": 6e-2, "atol": 6e-2},
}


class ARDecodeContractError(ValueError):
    """A decode-step contract is missing a semantic key or leaves the envelope."""

    def __init__(self, field: str, reason: str):
        self.code = CONTRACT_CODE
        self.field = field
        self.reason = reason
        super().__init__(f"{CONTRACT_CODE}: {field}: {reason}")


@dataclass(frozen=True)
class DecodeStepContract:
    """Semantic keys of one compiled decode step. Every field is required.

    ``eos_token_id`` must be passed explicitly; ``None`` means "never stop on a
    token", which is a decision the caller states, not a default.
    """

    storage_dtype: str
    accum_dtype: str
    rope_variant: str
    rope_theta: float
    rms_norm_eps: float
    eos_token_id: int | None
    sampling: str
    tie_break: str
    page_size: int
    max_seq_len: int


def _fail(field: str, reason: str):
    raise ARDecodeContractError(field, reason)


def _positive_finite(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(field, f"must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        _fail(field, f"must be finite and > 0, got {value!r}")
    return float(value)


def _positive_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        _fail(field, f"must be an int, got {value!r}")
    if value <= 0:
        _fail(field, f"must be > 0, got {value!r}")
    return int(value)


def _require_contract(obj: object) -> None:
    # Runtime callers are not type-checked; a dict or namespace must not pass.
    if not isinstance(obj, DecodeStepContract):
        _fail("contract", f"expected DecodeStepContract, got {type(obj).__name__}")


def validate_decode_contract(config: MoETransformerConfig, contract: DecodeStepContract) -> None:
    """Fail closed unless ``contract`` is complete, in the S0 envelope, and agrees with ``config``."""
    _require_contract(contract)
    for f in fields(DecodeStepContract):
        if f.name != "eos_token_id" and getattr(contract, f.name) is None:
            _fail(f.name, "is a semantic key and may not be None (Decision #21a)")
    verify_config(config)

    # Model class: dense GQA, full attention, no MoE, no sparse policy.
    if config.attn_kind != "gqa":
        _fail("config.attn_kind", f"S0 envelope is GQA only, got {config.attn_kind!r}")
    if config.sparse is not None:
        _fail("config.sparse", f"S0 envelope has no sparse attention, got {config.sparse!r}")
    if config.sliding_window:
        _fail("config.sliding_window", "S0 envelope is full attention")
    if any(config.attention_mode(i) != "full" for i in range(config.num_layers)):
        _fail("config.layer_types", "S0 envelope is full attention on every layer")
    if any(config.is_moe_layer(i) for i in range(config.num_layers)):
        _fail("config.first_k_dense", "S0 envelope is dense FFN on every layer")

    if contract.storage_dtype == "tf32":
        _fail("storage_dtype", "tf32 is a math mode, not a storage dtype (Decision #15a)")
    if contract.storage_dtype not in STORAGE_DTYPES:
        _fail("storage_dtype", f"must be one of {STORAGE_DTYPES}, got {contract.storage_dtype!r}")
    if contract.accum_dtype not in ACCUM_DTYPES:
        _fail("accum_dtype", f"must be one of {ACCUM_DTYPES}, got {contract.accum_dtype!r}")
    if contract.rope_variant not in ROPE_VARIANTS:
        _fail("rope_variant", f"must be one of {ROPE_VARIANTS}, got {contract.rope_variant!r}")
    if contract.rope_variant != config.rope_variant:
        _fail("rope_variant", f"contract {contract.rope_variant!r} != model {config.rope_variant!r}")
    theta = _positive_finite(contract.rope_theta, "rope_theta")
    if theta != float(config.rope_theta):
        _fail("rope_theta", f"contract {theta!r} != model {config.rope_theta!r}")
    eps = _positive_finite(contract.rms_norm_eps, "rms_norm_eps")
    if eps != float(config.rms_norm_eps):
        _fail("rms_norm_eps", f"contract {eps!r} != model {config.rms_norm_eps!r}")
    if contract.sampling not in SAMPLING_MODES:
        _fail("sampling", f"must be one of {SAMPLING_MODES}, got {contract.sampling!r}")
    if contract.tie_break not in TIE_BREAKS:
        _fail("tie_break", f"must be one of {TIE_BREAKS}, got {contract.tie_break!r}")
    eos = contract.eos_token_id
    if eos is not None:
        if isinstance(eos, bool) or not isinstance(eos, (int, np.integer)):
            _fail("eos_token_id", f"must be an int or explicit None, got {eos!r}")
        if not 0 <= int(eos) < config.vocab_size:
            _fail("eos_token_id", f"must be in [0, {config.vocab_size}), got {eos!r}")
    page = _positive_int(contract.page_size, "page_size")
    cap = _positive_int(contract.max_seq_len, "max_seq_len")
    if cap % page:
        _fail("max_seq_len", f"must be a multiple of page_size={page}, got {cap}")
    if cap > config.context_length:
        _fail("max_seq_len", f"exceeds model context_length={config.context_length}")


# ── frozen tiny fixture ──────────────────────────────────────────────────────
AR_DECODE_TINY = MoETransformerConfig(
    name="ar_decode_tiny", hidden_size=64, num_layers=2, vocab_size=256,
    context_length=128, attn_kind="gqa", num_attention_heads=4, num_kv_heads=2,
    head_dim=16, first_k_dense=2, dense_intermediate_size=128,
    rope_variant="rope", rope_theta=10000.0, rms_norm_eps=1e-6,
)

TINY_PROMPT: tuple[int, ...] = (1, 17, 42, 99, 3)

# sha256 over the fp64 bytes of every fixture array, in `_fixture_arrays` order.
TINY_WEIGHTS_SHA256 = "bfa5018ff77ed6126711045f99df07c37a04cdae26e4f2c68758856e9c522d3a"


def _hash_uniform(shape: Sequence[int], stream: int) -> np.ndarray:
    """Deterministic U[-1, 1) from splitmix64 over (stream, index) — no numpy RNG."""
    n = int(np.prod(shape))
    with np.errstate(over="ignore"):
        z = np.arange(n, dtype=np.uint64) + np.uint64(stream) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = (z >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))
    return (2.0 * u - 1.0).reshape(tuple(shape))


def tiny_fixture_weights(config: MoETransformerConfig = AR_DECODE_TINY) -> _rt.ModelWeights:
    """Dense GQA weights for ``config`` from the counter-based hash.

    Norm weights are deliberately not all-ones, so a compiled program that drops
    an RMSNorm scale fails the oracle instead of passing by coincidence.
    """
    H, D = config.hidden_size, config.head_dim
    Hq, Hkv = config.num_attention_heads, config.num_kv_heads
    F = config.dense_intermediate_size
    stream = iter(range(1, 1_000_000))

    def w(*shape, scale):
        return _hash_uniform(shape, next(stream)) * scale

    def norm(n):
        return 1.0 + 0.25 * _hash_uniform((n,), next(stream))

    s = math.sqrt(3.0 / H)  # unit-variance rows for a U[-1,1) draw
    layers = []
    for _ in range(config.num_layers):
        attn = {"kind": "gqa", "w_q": w(H, Hq * D, scale=s), "w_k": w(H, Hkv * D, scale=s),
                "w_v": w(H, Hkv * D, scale=s),
                "w_o": w(Hq * D, H, scale=math.sqrt(3.0 / (Hq * D)))}
        ffn = {"w_gate": w(H, F, scale=s), "w_up": w(H, F, scale=s),
               "w_down": w(F, H, scale=math.sqrt(3.0 / F))}
        layers.append(_rt.LayerWeights(norm1=norm(H), norm2=norm(H), attn=attn,
                                       ffn=ffn, is_moe=False))
    return _rt.ModelWeights(embed=w(config.vocab_size, H, scale=math.sqrt(3.0)),
                            layers=layers, final_norm=norm(H),
                            lm_head=w(H, config.vocab_size, scale=s))


def _fixture_arrays(weights: _rt.ModelWeights) -> list[np.ndarray]:
    out = [weights.embed]
    for lw in weights.layers:
        out += [lw.norm1, lw.norm2, lw.attn["w_q"], lw.attn["w_k"], lw.attn["w_v"],
                lw.attn["w_o"], lw.ffn["w_gate"], lw.ffn["w_up"], lw.ffn["w_down"]]
    return out + [weights.final_norm, weights.lm_head]


def weights_digest(weights: _rt.ModelWeights) -> str:
    h = hashlib.sha256()
    for a in _fixture_arrays(weights):
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()


def tiny_fixture_contract(*, storage_dtype: str = "fp32",
                          eos_token_id: int | None = None) -> DecodeStepContract:
    """The S0 fixture contract (page 16, capacity 64). ``eos_token_id`` stays explicit."""
    return DecodeStepContract(
        storage_dtype=storage_dtype, accum_dtype="fp32", rope_variant="rope",
        rope_theta=AR_DECODE_TINY.rope_theta, rms_norm_eps=AR_DECODE_TINY.rms_norm_eps,
        eos_token_id=eos_token_id, sampling="greedy", tie_break="first_occurrence",
        page_size=16, max_seq_len=64)


# ── oracle ───────────────────────────────────────────────────────────────────
@dataclass
class OracleTrace:
    """One greedy decode: chosen tokens, the logits that chose them, top-1/top-2 margins."""

    prompt: tuple[int, ...]
    tokens: tuple[int, ...]
    logits: np.ndarray            # (steps, vocab) fp64
    margins: np.ndarray           # (steps,) top-1 minus top-2 logit
    stopped_on_eos: bool
    state: _rt.DecodeState | None  # final KV state (cached decode only)


def _check_request(config, contract, prompt, max_new_tokens) -> tuple[int, ...]:
    validate_decode_contract(config, contract)
    ids = tuple(int(t) for t in np.asarray(prompt).reshape(-1))
    if not ids:
        _fail("prompt", "must contain at least one token")
    if any(not 0 <= t < config.vocab_size for t in ids):
        _fail("prompt", f"token ids must be in [0, {config.vocab_size})")
    n = _positive_int(max_new_tokens, "max_new_tokens")
    if len(ids) + n > contract.max_seq_len:
        _fail("max_new_tokens", f"prompt {len(ids)} + {n} exceeds max_seq_len={contract.max_seq_len}")
    return ids


def _choose(logits: np.ndarray) -> tuple[int, float]:
    tok = int(np.argmax(logits))  # first occurrence on ties (TIE_BREAKS)
    top2 = np.partition(logits, -2)[-2:]
    return tok, float(top2[1] - top2[0])


def reference_decode(config: MoETransformerConfig, weights: _rt.ModelWeights,
                     contract: DecodeStepContract, prompt, max_new_tokens: int) -> OracleTrace:
    """KV-cached greedy decode through the fp64 reference runtime."""
    ids = _check_request(config, contract, prompt, max_new_tokens)
    logits, state = _rt.prefill(config, weights, list(ids), max_seq=contract.max_seq_len)
    toks, rows, margins, eos = [], [], [], False
    for step in range(max_new_tokens):
        tok, margin = _choose(logits)
        toks.append(tok); rows.append(logits); margins.append(margin)
        if contract.eos_token_id is not None and tok == contract.eos_token_id:
            eos = True
            break
        if step + 1 == max_new_tokens:
            break  # the last token's logits are never consumed; do not append it
        logits, state = _rt.decode_step(config, weights, state, tok)
    return OracleTrace(ids, tuple(toks), np.asarray(rows), np.asarray(margins), eos, state)


def recompute_decode(config: MoETransformerConfig, weights: _rt.ModelWeights,
                     contract: DecodeStepContract, prompt, max_new_tokens: int) -> OracleTrace:
    """Greedy decode by recomputing the full forward on the growing prefix (no cache)."""
    ids = _check_request(config, contract, prompt, max_new_tokens)
    seq, toks, rows, margins, eos = list(ids), [], [], [], False
    for _ in range(max_new_tokens):
        logits = _rt.forward(config, weights, seq)[-1]
        tok, margin = _choose(logits)
        toks.append(tok); rows.append(logits); margins.append(margin)
        if contract.eos_token_id is not None and tok == contract.eos_token_id:
            eos = True
            break
        seq.append(tok)
    return OracleTrace(ids, tuple(toks), np.asarray(rows), np.asarray(margins), eos, None)


def decisive_steps(trace: OracleTrace, storage_dtype: str) -> np.ndarray:
    """Boolean mask of steps where token equality is a meaningful gate.

    A step is decisive when the oracle's top-1/top-2 margin exceeds twice the
    declared logit tolerance at that step's scale, so no in-tolerance compiled
    result can pick a different token. Elsewhere logits parity is the only gate.
    """
    if storage_dtype not in LOGIT_TOLERANCE:
        _fail("storage_dtype", f"no declared tolerance for {storage_dtype!r}")
    tol = LOGIT_TOLERANCE[storage_dtype]
    scale = np.abs(trace.logits).max(axis=1)
    return trace.margins > 2.0 * (tol["atol"] + tol["rtol"] * scale)


def paged_kv_pages(config: MoETransformerConfig, contract: DecodeStepContract,
                   state: _rt.DecodeState) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-layer ``(k_pages, v_pages)`` in the device layout a compiled step must produce.

    Shape ``[max_seq_len // page_size, page_size, num_kv_heads, head_dim]``; token
    ``t`` lives at page ``t // page_size``, slot ``t % page_size``. Keys are stored
    after RoPE. Slots past the live length are zero.
    """
    validate_decode_contract(config, contract)
    pages = contract.max_seq_len // contract.page_size
    shape = (pages, contract.page_size, config.num_kv_heads, config.head_dim)
    out = []
    for cache in state.caches:
        if cache[0] != "gqa":
            _fail("state", f"expected a dense GQA cache, got {cache[0]!r}")
        k, v = np.asarray(cache[1]), np.asarray(cache[2])
        if k.shape[0] != state.position or v.shape[0] != state.position:
            _fail("state", "cache length disagrees with decode position")
        kp, vp = np.zeros(shape), np.zeros(shape)
        kp.reshape(-1, *shape[2:])[: state.position] = k
        vp.reshape(-1, *shape[2:])[: state.position] = v
        out.append((kp, vp))
    return out


__all__ = [
    "ACCUM_DTYPES", "AR_DECODE_TINY", "ARDecodeContractError", "CONTRACT_CODE",
    "DecodeStepContract", "LOGIT_TOLERANCE", "OracleTrace", "ROPE_VARIANTS",
    "SAMPLING_MODES", "STORAGE_DTYPES", "TIE_BREAKS", "TINY_PROMPT",
    "TINY_WEIGHTS_SHA256", "decisive_steps", "paged_kv_pages", "recompute_decode", "reference_decode",
    "tiny_fixture_contract", "tiny_fixture_weights", "validate_decode_contract",
    "weights_digest",
]
