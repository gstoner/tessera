"""Python fusion-intent emitter (Decision #19 emit-half).

`stamp_fusion_intents(module)` tags the terminal op of each recognized linear
fusion chain with `tessera.fusion.intent = "<kernel>"`, so the Apple Target IR
passes consume the compiler's fusion decision (source="descriptor") instead of
re-discovering it. This closes the Python→IR half of the loop; the C++
consume-half is locked by tests/tessera-ir/phase8/apple_gpu_fusion_descriptor.mlir
and tests/unit/test_apple_fusion_descriptor.py.
"""

from __future__ import annotations

from pathlib import Path

import tessera as ts
from tessera.compiler.graph_ir import GraphIRBuilder
from tessera.compiler.canonical_compile import stamp_fusion_intents, _INTENT_KERNELS


def _module(fn):
    # The AST frontend lowers `def` functions (not lambdas), so callers pass a
    # real def.
    b = GraphIRBuilder()
    b.lower(fn)
    return b.module()


def test_stamps_matmul_softmax_matmul_terminal():
    def f(a, b, c):
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(a, b)), c)
    m = _module(f)
    assert stamp_fusion_intents(m) == 1
    body = m.functions[-1].body
    # the tail matmul (highest index) is where the C++ pass reads the intent.
    # Stamped into `attrs` (MLIR-only), never kwargs (the op's call arguments).
    assert 'tessera.fusion.intent = "matmul_softmax_matmul"' in (body[-1].attrs or "")
    assert "tessera.fusion.intent" not in (body[0].attrs or "")
    assert "tessera.fusion.intent" not in body[-1].kwargs


def test_stamps_two_op_chains():
    def gelu_chain(a, b):
        return ts.ops.gelu(ts.ops.matmul(a, b))

    def rmsnorm_chain(a, b):
        return ts.ops.rmsnorm(ts.ops.matmul(a, b))

    def softmax_chain(a, b):
        return ts.ops.softmax(ts.ops.matmul(a, b))

    for kernel, fn in (("matmul_gelu", gelu_chain),
                       ("matmul_rmsnorm", rmsnorm_chain),
                       ("matmul_softmax", softmax_chain)):
        m = _module(fn)
        assert stamp_fusion_intents(m) == 1, kernel
        assert f'tessera.fusion.intent = "{kernel}"' in (m.functions[-1].body[-1].attrs or "")


def test_rendered_mlir_carries_intent():
    def gelu_chain(a, b):
        return ts.ops.gelu(ts.ops.matmul(a, b))
    m = _module(gelu_chain)
    stamp_fusion_intents(m)
    mlir = m.functions[-1].to_mlir()
    assert 'tessera.fusion.intent = "matmul_gelu"' in mlir


def test_idempotent():
    def f(a, b, c):
        return ts.ops.matmul(ts.ops.softmax(ts.ops.matmul(a, b)), c)
    m = _module(f)
    assert stamp_fusion_intents(m) == 1
    assert stamp_fusion_intents(m) == 1   # re-stamping doesn't double-count


def test_no_chain_no_stamp():
    def single(a, b):
        return ts.ops.matmul(a, b)
    m = _module(single)
    assert stamp_fusion_intents(m) == 0


def test_emitter_intents_match_cpp_consumers():
    # Loop contract: every intent the emitter produces is an intent some Apple
    # C++ fusion pass reads. Guards against the two halves drifting apart.
    lowering = (Path(__file__).resolve().parents[2] / "src" / "compiler" /
                "codegen" / "Tessera_Apple_Backend" / "lib" / "Target" / "Apple" /
                "Lowering")
    consumed = set()
    for cpp in lowering.glob("Matmul*FusionToAppleGPU.cpp"):
        src = cpp.read_text()
        if "tessera.fusion.intent" not in src:
            continue
        for kernel in _INTENT_KERNELS:
            if f'== "{kernel}"' in src or f'(intent == "{kernel}")' in src:
                consumed.add(kernel)
    assert _INTENT_KERNELS <= consumed, (
        f"emitter produces intents the C++ doesn't consume: {_INTENT_KERNELS - consumed}")
