"""Portable route-policy contracts for APPLE-ATTN-BWD-1."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import pytest

from tessera.compiler.apple_attention_backward import (
    ATOMIC,
    ROUTE_IDS,
    SERIAL_RECOMPUTE,
    SPLIT_REDUCED,
    AppleAttentionBackwardPolicyError,
    resolve_route,
    resolve_variant_route,
    selector_shape_key,
    workspace_bytes,
)
from tessera.compiler.apple_route_selector import (
    AppleRouteContext,
    STRICT_ROUTE_LEDGER_SCHEMA,
)


_CONTEXT = AppleRouteContext(
    device="apple7",
    physical_device="Apple M1 Max",
    os_version="26.5.2",
    sdk_version="26.4",
    compiler_fingerprint="sha256:compiler",
    runtime_fingerprint="sha256:runtime",
)
_NOW = datetime(2026, 7, 20, tzinfo=timezone.utc)


def _strict_ledger(tmp_path, *decisions):
    path = tmp_path / "attention-backward-ledger.json"
    path.write_text(json.dumps({
        "schema": STRICT_ROUTE_LEDGER_SCHEMA,
        "selection_scope": "runtime_route",
        "measured_at": "2026-07-18T12:00:00Z",
        "expires_at": "2026-08-18T12:00:00Z",
        "context": _CONTEXT.as_mapping(),
        "source_report_count": 2,
        "source_report_digests": ["sha256:" + "a" * 64,
                                  "sha256:" + "b" * 64],
        "decisions": [{
            "device": "apple7",
            "op": "flash_attn_bwd",
            "shape": shape,
            "dtype": "f32",
            "timing_domain": domain,
            "incumbent_route": SERIAL_RECOMPUTE,
            "selected_route": route,
            "status": "promote_candidate",
            "selected_evidence": {
                "provenance": "native_gpu",
                "correctness": True,
                "device": "apple7",
                "timing_domain": domain,
            },
        } for shape, route, domain in decisions],
    }), encoding="utf-8")
    return path


def test_auto_selects_proven_zero_workspace_deterministic_route():
    policy = resolve_route((1, 2, 7, 8), (1, 2, 7, 6))
    assert policy.route == SERIAL_RECOMPUTE
    assert policy.deterministic and policy.workspace_bytes == 0
    assert policy.accumulation_dtype == "f32" and policy.implemented


def test_route_ids_match_native_abi_contract():
    assert ROUTE_IDS == {
        SERIAL_RECOMPUTE: 0, ATOMIC: 1, SPLIT_REDUCED: 2}


def test_apple7_selector_applies_determinism_workspace_and_timing_domain(tmp_path):
    split_key = selector_shape_key(
        outer=1, q_heads=4, kv_heads=4, sq=16, sk=16, dim=16,
        causal=False, window=0, bias=False, softcap=0.0)
    assert split_key == "b1_hq4_hkv4_sq16_sk16_d16_c0_w0_bias0_sc0p0"
    atomic_key = selector_shape_key(
        outer=1, q_heads=4, kv_heads=4, sq=17, sk=19, dim=64,
        causal=True, window=0, bias=False, softcap=0.0)
    ledger = _strict_ledger(
        tmp_path,
        (split_key, SPLIT_REDUCED, "end_to_end"),
        (atomic_key, ATOMIC, "end_to_end"),
    )
    injected = {
        "ledger_path": ledger, "context": _CONTEXT, "now": _NOW,
    }

    assert resolve_route(
        (4, 16, 16), (4, 16, 16), selector_key=split_key,
        device="apple7").route == SERIAL_RECOMPUTE
    selected = resolve_route(
        (4, 16, 16), (4, 16, 16), selector_key=split_key,
        device="apple7", **injected)
    assert selected.route == SPLIT_REDUCED and selected.deterministic
    capped = resolve_route(
        (4, 16, 16), (4, 16, 16), selector_key=split_key,
        device="apple7", workspace_limit_bytes=0, **injected)
    assert capped.route == SERIAL_RECOMPUTE

    atomic = resolve_route(
        (4, 19, 64), (4, 19, 64), selector_key=atomic_key,
        device="apple7", **injected)
    assert atomic.route == ATOMIC and not atomic.deterministic
    deterministic = resolve_route(
        (4, 19, 64), (4, 19, 64), selector_key=atomic_key, device="apple7",
        deterministic=True, **injected)
    assert deterministic.route == SERIAL_RECOMPUTE
    device_domain = resolve_route(
        (4, 19, 64), (4, 19, 64), selector_key=atomic_key, device="apple7",
        timing_domain="device", **injected)
    assert device_domain.route == SERIAL_RECOMPUTE


def test_variant_resolver_builds_exact_key_and_workspace_shape(tmp_path):
    key = selector_shape_key(
        outer=1, q_heads=2, kv_heads=2, sq=4, sk=1025, dim=64,
        causal=True, window=0, bias=False, softcap=0.0)
    ledger = _strict_ledger(tmp_path, (key, SPLIT_REDUCED, "end_to_end"))
    selected = resolve_variant_route(
        outer=1, q_heads=2, kv_heads=2, sq=4, sk=1025, dim=64,
        causal=True, device="apple7", ledger_path=ledger, context=_CONTEXT,
        now=_NOW)
    assert selected.route == SPLIT_REDUCED
    assert selected.workspace_bytes == 4 * 2 * (2 * 1025 * 64)
    with pytest.raises(AppleAttentionBackwardPolicyError, match="geometry"):
        resolve_variant_route(
            outer=1, q_heads=3, kv_heads=2, sq=4, sk=8, dim=16,
            causal=False)


def test_split_workspace_is_exactly_one_extra_dkdv_partial():
    k = (1, 2, 7, 8)
    v = (1, 2, 7, 6)
    expected = 4 * ((1 * 2 * 7 * 8) + (1 * 2 * 7 * 6))
    assert workspace_bytes(k, v, route=SPLIT_REDUCED) == expected
    assert workspace_bytes(k, v, route=SERIAL_RECOMPUTE) == 0
    assert workspace_bytes(k, v, route=ATOMIC) == 0


def test_deterministic_atomic_rejects_before_runtime_loading():
    with pytest.raises(AppleAttentionBackwardPolicyError,
                       match="cannot use the atomic route"):
        resolve_route((7, 8), (7, 8), route=ATOMIC, deterministic=True)


def test_split_workspace_cap_rejects_before_implementation_gate():
    required = workspace_bytes((7, 8), (7, 8), route=SPLIT_REDUCED)
    with pytest.raises(AppleAttentionBackwardPolicyError, match="exceeding limit"):
        resolve_route((7, 8), (7, 8), route=SPLIT_REDUCED,
                      workspace_limit_bytes=required - 1)


def test_split_candidate_is_implemented_and_deterministic():
    policy = resolve_route((7, 8), (7, 8), route=SPLIT_REDUCED,
                           require_implemented=False)
    assert policy.deterministic and policy.implemented
    assert policy.workspace_bytes > 0


def test_atomic_candidate_is_implemented_and_nondeterministic():
    policy = resolve_route((7, 8), (7, 8), route=ATOMIC)
    assert policy.implemented and not policy.deterministic


@pytest.mark.parametrize("route", ["", "rocm_g6c", "cuda_atomic"])
def test_unknown_route_rejects_stably(route):
    with pytest.raises(AppleAttentionBackwardPolicyError, match="unknown"):
        resolve_route((7, 8), (7, 8), route=route)
