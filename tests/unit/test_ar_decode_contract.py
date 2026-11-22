"""AR-DECODE-PROGRAM-1 S0: decode-step contract, frozen fixture and oracle.

Host-independent (pure numpy). Device slices S1+ compare against this oracle.
"""

from dataclasses import replace

import numpy as np
import pytest

from tessera.compiler.diagnostic_codes import code_lookup
from tessera.models import ar_decode as A
from tessera.models import moe_transformer_runtime as rt
from tessera.stdlib.attention import apply_rope

CFG = A.AR_DECODE_TINY


@pytest.fixture(scope="module")
def weights():
    return A.tiny_fixture_weights()


@pytest.fixture(scope="module")
def contract():
    return A.tiny_fixture_contract()


def test_fixture_weights_are_frozen_bytes(weights):
    # Counter-hash generation: identical on every host and numpy version.
    assert A.weights_digest(weights) == A.TINY_WEIGHTS_SHA256
    assert A.weights_digest(A.tiny_fixture_weights()) == A.TINY_WEIGHTS_SHA256
    # Norm scales are not all-ones, so dropping one cannot pass by coincidence.
    assert not np.allclose(weights.layers[0].norm1, 1.0)
    assert not np.allclose(weights.final_norm, 1.0)


def test_cached_decode_matches_full_recompute(weights, contract):
    cached = A.reference_decode(CFG, weights, contract, A.TINY_PROMPT, 32)
    full = A.recompute_decode(CFG, weights, contract, A.TINY_PROMPT, 32)
    assert cached.tokens == full.tokens
    # Per-step logits, not only argmax, so a tie cannot hide a cache error.
    np.testing.assert_allclose(cached.logits, full.logits, rtol=0, atol=1e-10)
    assert len(set(cached.tokens)) > 8  # the trajectory is not a fixed point


def test_every_fp32_step_is_decisive(weights, contract):
    trace = A.reference_decode(CFG, weights, contract, A.TINY_PROMPT, 32)
    assert A.decisive_steps(trace, "fp32").all()
    assert trace.margins.min() > 0.04


def test_eos_stops_decode(weights):
    free = A.reference_decode(CFG, weights, A.tiny_fixture_contract(), A.TINY_PROMPT, 16)
    eos = free.tokens[3]
    first = free.tokens.index(eos)
    stopped = A.reference_decode(CFG, weights, A.tiny_fixture_contract(eos_token_id=eos),
                                 A.TINY_PROMPT, 16)
    assert stopped.stopped_on_eos and stopped.tokens == free.tokens[: first + 1]
    full = A.recompute_decode(CFG, weights, A.tiny_fixture_contract(eos_token_id=eos),
                              A.TINY_PROMPT, 16)
    assert full.tokens == stopped.tokens


def test_paged_layout_holds_roped_keys_at_page_slots(weights, contract):
    trace = A.reference_decode(CFG, weights, contract, A.TINY_PROMPT, 20)
    pages = A.paged_kv_pages(CFG, contract, trace.state)
    P = trace.state.position
    assert P == len(A.TINY_PROMPT) + 19  # the last chosen token is not yet appended
    k_pages, v_pages = pages[0]
    assert k_pages.shape == (4, 16, CFG.num_kv_heads, CFG.head_dim)
    # Independent layer-0 key for prompt token 2, from weights and the contract's theta.
    ids = list(A.TINY_PROMPT)
    h = rt.rmsnorm(weights.embed[ids], weights.layers[0].norm1, contract.rms_norm_eps)
    k = (h @ weights.layers[0].attn["w_k"]).reshape(len(ids), CFG.num_kv_heads, CFG.head_dim)
    k2 = apply_rope(k.transpose(1, 0, 2), np.arange(len(ids)),
                    base=contract.rope_theta).transpose(1, 0, 2)[2]
    np.testing.assert_allclose(k_pages[2 // 16, 2 % 16], k2, rtol=0, atol=1e-12)
    # Token 17 sits on page 1, slot 1; slots past the live length are zero.
    assert np.any(k_pages[1, 1]) and not np.any(k_pages[P // 16, P % 16:])
    assert not np.any(v_pages[3])


def test_rope_theta_reaches_the_computation(weights):
    """The runtime used to ignore config.rope_theta (always base 10000)."""
    hi = replace(CFG, rope_theta=500000.0)
    c_hi = replace(A.tiny_fixture_contract(), rope_theta=500000.0)
    base = A.reference_decode(CFG, weights, A.tiny_fixture_contract(), A.TINY_PROMPT, 8)
    other = A.reference_decode(hi, weights, c_hi, A.TINY_PROMPT, 8)
    assert np.abs(base.logits - other.logits).max() > 1e-3
    full = A.recompute_decode(hi, weights, c_hi, A.TINY_PROMPT, 8)
    np.testing.assert_allclose(other.logits, full.logits, rtol=0, atol=1e-10)


def test_contract_code_is_registered():
    entry = code_lookup(A.CONTRACT_CODE)
    assert entry is not None and entry.language == "python" and entry.severity == "error"


@pytest.mark.parametrize("field,value,needle", [
    ("storage_dtype", None, "may not be None"),
    ("storage_dtype", "tf32", "math mode"),
    ("storage_dtype", "fp8_e4m3", "must be one of"),
    ("accum_dtype", None, "may not be None"),
    ("accum_dtype", "fp16", "must be one of"),
    ("rope_variant", "yarn", "must be one of"),
    ("rope_theta", None, "may not be None"),
    ("rope_theta", 0.0, "> 0"),
    ("rope_theta", float("inf"), "> 0"),
    ("rope_theta", 20000.0, "!= model"),
    ("rms_norm_eps", 1e-5, "!= model"),
    ("eos_token_id", 256, "must be in"),
    ("eos_token_id", True, "explicit None"),
    ("sampling", "top_k", "must be one of"),
    ("tie_break", "last_occurrence", "must be one of"),
    ("page_size", 0, "> 0"),
    ("page_size", 24, "multiple of page_size"),
    ("max_seq_len", 256, "context_length"),
    ("max_seq_len", None, "may not be None"),
])
def test_invalid_contract_fails_closed(field, value, needle):
    bad = replace(A.tiny_fixture_contract(), **{field: value})
    with pytest.raises(A.ARDecodeContractError) as exc:
        A.validate_decode_contract(CFG, bad)
    assert exc.value.code == A.CONTRACT_CODE
    assert str(exc.value).startswith(A.CONTRACT_CODE)
    assert needle in str(exc.value)


def test_contract_has_no_defaults():
    with pytest.raises(TypeError):
        A.DecodeStepContract(storage_dtype="fp32", accum_dtype="fp32")  # type: ignore[call-arg]


@pytest.mark.parametrize("change,field", [
    ({"attn_kind": "mla", "rope_head_dim": 8}, "config.attn_kind"),
    ({"first_k_dense": 1}, "config.first_k_dense"),
    ({"sliding_window": 8}, "config.sliding_window"),
])
def test_model_outside_envelope_fails_closed(change, field):
    with pytest.raises(A.ARDecodeContractError) as exc:
        A.validate_decode_contract(replace(CFG, **change), A.tiny_fixture_contract())
    assert exc.value.field == field


@pytest.mark.parametrize("prompt,steps,needle", [
    ((), 4, "at least one token"),
    ((1, 300), 4, "token ids"),
    (A.TINY_PROMPT, 0, "> 0"),
    (A.TINY_PROMPT, 60, "exceeds max_seq_len"),
])
def test_invalid_request_fails_closed(weights, contract, prompt, steps, needle):
    with pytest.raises(A.ARDecodeContractError, match=needle):
        A.reference_decode(CFG, weights, contract, prompt, steps)


def test_decisive_steps_refuses_undeclared_storage(weights, contract):
    trace = A.reference_decode(CFG, weights, contract, A.TINY_PROMPT, 4)
    with pytest.raises(A.ARDecodeContractError, match="no declared tolerance"):
        A.decisive_steps(trace, "fp8_e4m3")


def test_non_contract_object_is_refused():
    with pytest.raises(A.ARDecodeContractError, match="expected DecodeStepContract"):
        A.validate_decode_contract(CFG, {"storage_dtype": "fp32"})  # type: ignore[arg-type]
