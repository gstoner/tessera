from __future__ import annotations

import numpy as np

import tessera as ts


def test_transformer_attention_block_compiles_and_executes_cpu_dataflow():
    @ts.jit
    def transformer_attention_block(x, wq, wk, wv, wo):
        q = ts.ops.matmul(x, wq)
        k = ts.ops.matmul(x, wk)
        v = ts.ops.matmul(x, wv)
        k_t = ts.ops.transpose(k)
        scores = ts.ops.matmul(q, k_t)
        probs = ts.ops.softmax(scores)
        ctx = ts.ops.matmul(probs, v)
        return ts.ops.matmul(ctx, wo)

    x = np.array(
        [
            [0.25, -0.5, 1.0],
            [1.5, 0.0, -0.25],
        ],
        dtype=np.float32,
    )
    wq = np.array(
        [
            [0.5, -0.25],
            [0.0, 0.75],
            [1.0, 0.5],
        ],
        dtype=np.float32,
    )
    wk = np.array(
        [
            [0.25, 0.5],
            [-0.5, 1.0],
            [0.75, -0.25],
        ],
        dtype=np.float32,
    )
    wv = np.array(
        [
            [1.0, 0.0],
            [0.5, -0.5],
            [-0.25, 0.75],
        ],
        dtype=np.float32,
    )
    wo = np.array(
        [
            [0.5, -1.0],
            [1.25, 0.25],
        ],
        dtype=np.float32,
    )

    q = x @ wq
    k = x @ wk
    v = x @ wv
    scores = q @ k.T
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    expected = (probs @ v) @ wo

    np.testing.assert_allclose(
        transformer_attention_block(x, wq, wk, wv, wo),
        expected,
        rtol=1e-6,
    )

    ir = transformer_attention_block.ir_text()
    assert ir.count("tessera.matmul") == 6
    assert "tessera.transpose" in ir
    assert "tessera.softmax" in ir

    assert transformer_attention_block.uses_compiled_path
    artifacts = {artifact.level: artifact.text for artifact in transformer_attention_block.lowering_artifacts()}
    assert set(artifacts) == {"graph", "schedule", "tile", "target"}
    assert artifacts["schedule"].count("schedule.tile") == 6
    assert "layout_transform" in artifacts["tile"]
    assert "stable_reduction" in artifacts["tile"]
    assert artifacts["target"].count("tessera.cpu.matmul") == 6
    explanation = transformer_attention_block.explain_lowering()
    assert "JIT_COMPILED_CPU" in explanation
