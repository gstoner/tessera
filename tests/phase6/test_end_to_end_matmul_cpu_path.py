from __future__ import annotations

import numpy as np

import tessera as ts


def test_jit_matmul_executes_through_cpu_lowering_path():
    @ts.jit
    def mm(A, B):
        return ts.ops.matmul(A, B)

    A = np.arange(6, dtype=np.float32).reshape(2, 3)
    B = np.arange(12, dtype=np.float32).reshape(3, 4)

    out = mm(A, B)

    np.testing.assert_allclose(out, A @ B)
    assert mm.cpu_plan is not None
    assert mm.cpu_plan.op_name == "tessera.matmul"


def test_jit_gemm_lowering_artifacts_cover_all_compiler_layers():
    @ts.jit
    def gemm(A, B):
        return ts.ops.gemm(A, B)

    artifacts = {artifact.level: artifact.text for artifact in gemm.lowering_artifacts()}

    assert set(artifacts) == {"graph", "schedule", "tile", "target"}
    assert "tessera.gemm" in artifacts["graph"]
    assert "tessera.schedule.tile" in artifacts["schedule"]
    assert "tile_m = 128" in artifacts["schedule"]
    assert "tessera.tile.matmul" in artifacts["tile"]
    assert "tessera.cpu.matmul" in artifacts["target"]
    assert gemm.schedule_ir == artifacts["schedule"]
    assert gemm.tile_ir == artifacts["tile"]
    assert gemm.target_ir == artifacts["target"]


def test_jit_matmul_cpu_plan_supports_kwargs():
    @ts.jit
    def mm(A, B):
        return ts.ops.matmul(A, B)

    A = np.eye(3, dtype=np.float32)
    B = np.ones((3, 2), dtype=np.float32)

    np.testing.assert_allclose(mm(B=B, A=A), A @ B)


def test_non_matmul_jit_keeps_existing_eager_behavior():
    @ts.jit
    def relu(x):
        return ts.ops.relu(x)

    x = np.array([-1.0, 2.0], dtype=np.float32)

    assert relu.cpu_plan is None
    np.testing.assert_allclose(relu(x), np.array([0.0, 2.0], dtype=np.float32))


def test_developer_frontend_docs_link_first_end_to_end_path():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    doc = root / "docs" / "guides" / "Tessera_Developer_Frontend_End_To_End.md"
    readme = (root / "docs" / "README.md").read_text(encoding="utf-8")
    api = (root / "docs" / "spec" / "PYTHON_API_SPEC.md").read_text(encoding="utf-8")

    text = doc.read_text(encoding="utf-8")
    assert "@jit matmul -> Graph IR -> Schedule IR -> Tile IR -> Target IR -> CPU execution" in text
    assert "mm.lowering_artifacts()" in text
    assert "docs/guides/Tessera_Developer_Frontend_End_To_End.md" in readme
    assert ".lowering_artifacts()" in api
