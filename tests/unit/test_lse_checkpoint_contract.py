"""Structural gates for the persistent attention LSE checkpoint contract."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTN_TD = REPO_ROOT / "src/compiler/tile_opt_fa4/include/tessera/Dialect/Attn/Attn.td"
LOWERING = REPO_ROOT / "src/transforms/lib/TileIRLoweringPass.cpp"
ROCM_ADAPTER = REPO_ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/TileToROCM.cpp"
ROCM_FORWARD = REPO_ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAFlashAttnKernel.cpp"
ROCM_BACKWARD = (
    REPO_ROOT / "src/compiler/codegen/Tessera_ROCM_Backend/lib/Conversion/GenerateWMMAFlashAttnBwdKernel.cpp"
)
ROCM_NATIVE = REPO_ROOT / "python/tessera/compiler/rocm_native.py"


def _op_body(source: str, op: str) -> str:
    start = source.index(f"def {op} :")
    depth = 0
    for end in range(source.index("{", start), len(source)):
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
            if depth == 0:
                return source[start : end + 1]
    raise AssertionError(f"unterminated ODS body for {op}")


def test_lse_save_has_destination_identity_and_write_effect() -> None:
    body = _op_body(ATTN_TD.read_text(encoding="utf-8"), "LseSaveOp")
    assert "MemoryEffects<[MemWrite]>" in body
    assert "Pure" not in body
    assert "AnyMemRef:$destination" in body
    for field in ("$row_offset", "$identity", "$memory_space", "$scope"):
        assert field in body


def test_lse_load_has_source_identity_and_read_effect() -> None:
    body = _op_body(ATTN_TD.read_text(encoding="utf-8"), "LseLoadOp")
    assert "MemoryEffects<[MemRead]>" in body
    assert "Pure" not in body
    assert "AnyMemRef:$source" in body
    for field in ("$row_offset", "$identity", "$memory_space", "$scope"):
        assert field in body


def test_default_attention_lowering_emits_no_destinationless_checkpoint() -> None:
    source = LOWERING.read_text(encoding="utf-8")
    assert '"tessera_attn.lse.save"' not in source
    assert '"tessera_attn.lse.load"' not in source


def test_rocm_training_package_selects_saved_or_recomputed_lse() -> None:
    adapter = ROCM_ADAPTER.read_text(encoding="utf-8")
    forward = ROCM_FORWARD.read_text(encoding="utf-8")
    backward = ROCM_BACKWARD.read_text(encoding="utf-8")
    native = ROCM_NATIVE.read_text(encoding="utf-8")
    assert '"tessera.lse_checkpoint"' in adapter
    assert '"save_lse"' in adapter and '"saved_lse"' in adapter
    assert 'getAttrOfType<BoolAttr>("save_lse")' in forward
    assert 'flag("saved_lse")' in backward
    assert '"lse_checkpoint", "auto"' in native
    assert '{"auto", "saved", "recompute"}' in native
    assert "max(sq, sk) >= 128" in native
