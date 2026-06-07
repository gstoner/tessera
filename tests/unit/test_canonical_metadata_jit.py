"""Canonical compile metadata reaches the user-facing `@jit` runtime artifact.

COMPILER_AUDIT Next Work #1: the component-aware compile metadata
(`component_ops`, `shape_envelope`, `effects`, `layout_contracts`,
`fusion_groups`) is derived by `CompileResult` but used to be discarded on the
`@jit().runtime_artifact()` path — every key was absent for real jitted
functions. This wires `CompileResult.descriptive_metadata()` through
`JitFn._build_runtime_artifact` (additive — never overrides the executability
decision the cpu/apple fast paths own) and deepens `fusion_groups` to recognize
the cross-family chains the Apple GPU runtime actually fuses.
"""

import numpy as np
import pytest

import tessera as ts

_CANON_KEYS = [
    "canonical_component_ops",
    "canonical_shape_envelope",
    "canonical_effects",
    "canonical_layout_contracts",
    "canonical_fusion_groups",
]


def _mha(q, k, v):
    s = ts.ops.matmul(q, k)
    a = ts.ops.softmax(s)
    return ts.ops.matmul(a, v)


def _meta(fn, target):
    return ts.jit(target=target)(fn).runtime_artifact().metadata


# ── metadata is present on the jit artifact (the bug this closes) ──────────── #
@pytest.mark.parametrize("target", ["cpu", "apple_gpu"])
def test_jit_artifact_carries_canonical_metadata(target):
    m = _meta(_mha, target)
    for key in _CANON_KEYS:
        assert key in m, f"{key} missing from {target} jit artifact metadata"
    assert m["canonical_component_ops"] == ["matmul", "softmax"]
    assert m["canonical_effects"]["summary"] == "pure"
    se = m["canonical_shape_envelope"]
    assert se.get("schema") == "tessera.compile.shape_envelope.v1"
    assert [f["name"] for f in se["functions"]] == ["_mha"]   # the fn's name
    lc = m["canonical_layout_contracts"]
    assert lc.get("schema") == "tessera.compile.layout_contracts.v1"


def test_metadata_does_not_override_executability():
    # apple_gpu MHA is executable; merging descriptive metadata must not flip it.
    art = ts.jit(target="apple_gpu")(_mha).runtime_artifact()
    assert art.metadata["execution_mode"] == "metal_runtime"
    assert art.metadata.get("executable") is True


# ── fusion_groups recognizes the real fused chains ─────────────────────────── #
def test_fusion_groups_attention_block():
    fg = _meta(_mha, "apple_gpu")["canonical_fusion_groups"]
    chains = [g for g in fg if g["kind"] == "known_chain"]
    assert any(g["fused_kernel"] == "matmul_softmax_matmul" for g in chains)
    g = next(g for g in chains if g["fused_kernel"] == "matmul_softmax_matmul")
    assert [o["op"] for o in g["ops"]] == ["matmul", "softmax", "matmul"]


def test_fusion_groups_matmul_gelu():
    # Literal op calls — the graph builder recognizes `ts.ops.<name>` from the
    # AST, so a dynamic getattr would NOT emit the op.
    def fn(x, w):
        return ts.ops.gelu(ts.ops.matmul(x, w))
    fg = _meta(fn, "apple_gpu")["canonical_fusion_groups"]
    assert any(g.get("fused_kernel") == "matmul_gelu"
               for g in fg if g["kind"] == "known_chain")


def test_fusion_groups_matmul_rmsnorm():
    def fn(x, w):
        return ts.ops.rmsnorm(ts.ops.matmul(x, w))
    fg = _meta(fn, "apple_gpu")["canonical_fusion_groups"]
    assert any(g.get("fused_kernel") == "matmul_rmsnorm"
               for g in fg if g["kind"] == "known_chain")


def test_fusion_groups_matmul_softmax_two_op():
    def fn(a, b):
        return ts.ops.softmax(ts.ops.matmul(a, b))
    fg = _meta(fn, "apple_gpu")["canonical_fusion_groups"]
    assert any(g.get("fused_kernel") == "matmul_softmax" for g in fg if g["kind"] == "known_chain")


def test_no_false_chain_for_unconnected_ops():
    # matmul then an independent softmax on a *different* arg — not a data-flow
    # chain, so no known_chain fusion group.
    def fn(a, b, c):
        _ = ts.ops.matmul(a, b)
        return ts.ops.softmax(c)
    fg = _meta(fn, "cpu")["canonical_fusion_groups"]
    assert not [g for g in fg if g["kind"] == "known_chain"]


def test_effects_reflect_random_op():
    def fn(x):
        return ts.ops.dropout(x, p=0.5)
    m = _meta(fn, "cpu")
    assert m["canonical_effects"]["summary"] in ("random", "memory", "io", "top")
