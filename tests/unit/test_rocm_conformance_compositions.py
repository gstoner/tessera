"""Direct ROCm execute/compare proof for matmul -> softmax composition.

Skip-clean without a built ``tessera-opt`` and live AMD GPU.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")


def _runtime_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def test_rocm_matmul_softmax_composition_matches_numpy():
    rt = _runtime_or_skip()
    rng = np.random.default_rng(20260712)
    a = (rng.standard_normal((16, 16)) * 0.25).astype(np.float16)
    b = (rng.standard_normal((16, 16)) * 0.25).astype(np.float16)

    matmul = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "scores",
        "ops": [{"op_name": "tessera.matmul", "result": "scores",
                 "operands": ["a", "b"], "kwargs": {}}],
    })
    scores = rt.launch(matmul, (a, b))["output"]

    softmax = rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_softmax_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["scores"], "output_name": "out",
        "ops": [{"op_name": "tessera.softmax", "result": "out",
                 "operands": ["scores"], "kwargs": {"axis": -1}}],
    })
    out = rt.launch(softmax, (scores,))["output"]

    ref_scores = a.astype(np.float32) @ b.astype(np.float32)
    exp = np.exp(ref_scores - ref_scores.max(axis=-1, keepdims=True))
    expected = exp / exp.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(out, expected, rtol=0, atol=2e-2)
