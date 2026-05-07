from __future__ import annotations

from tessera.compiler.legality import TensorContract, check_op_legality


def test_matmul_legality_accepts_static_rank2_f32_cpu():
    result = check_op_legality(
        "matmul",
        [
            TensorContract(shape=(16, 32), dtype="f32", layout="row_major"),
            TensorContract(shape=(32, 8), dtype="f32", layout="row_major"),
        ],
        target="cpu",
    )

    assert result.ok


def test_matmul_legality_reports_stable_shape_code():
    result = check_op_legality(
        "matmul",
        [
            TensorContract(shape=(16, 31), dtype="f32", layout="row_major"),
            TensorContract(shape=(32, 8), dtype="f32", layout="row_major"),
        ],
        target="cpu",
    )

    assert not result.ok
    assert [d.code for d in result.diagnostics] == ["LEGALITY_MATMUL_K_MISMATCH"]


def test_legality_reports_layout_and_target_capability():
    result = check_op_legality(
        "flash_attn",
        [
            TensorContract(shape=(2, 4, 8), dtype="int8", layout="diagonal"),
            TensorContract(shape=(2, 4, 8), dtype="int8"),
            TensorContract(shape=(2, 4, 8), dtype="int8"),
        ],
        target="apple_gpu",
    )

    codes = {d.code for d in result.diagnostics}
    assert "LEGALITY_LAYOUT_UNSUPPORTED" in codes
    assert "LEGALITY_TARGET_CAPABILITY" in codes


def test_collective_requires_communication_effect():
    result = check_op_legality(
        "tessera.all_reduce",
        [TensorContract(shape=(8, 8), dtype="f32")],
        target="cpu",
        effects=("read",),
    )

    assert not result.ok
    assert any(d.code == "LEGALITY_COLLECTIVE_EFFECT" for d in result.diagnostics)
