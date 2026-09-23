"""Typed frontend and route-receipt guards for folded gfx1201 matmul."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from tessera.compiler.rocm_mxfp4_folded_frontend import (
    FoldedScaledMatmulProgram,
    author_folded_scaled_matmul_graph,
    compile_folded_scaled_matmul,
)
from tessera.compiler.rocm_mxfp4_native import select_mxfp4_route


def _inputs() -> tuple[np.ndarray, np.ndarray, object]:
    a = np.full((65, 64), 0x38, dtype=np.uint8)
    a_scale = np.ones(65, dtype=np.float32)
    codes = np.ones((48, 64), dtype=np.uint8)
    scales = np.full((2, 48), 127, dtype=np.uint8)
    folded = prepare_folded_weights(
        mx.pack_e2m1_codes(codes), scales, allow_approximate=True,
    )
    return a, a_scale, folded


def test_authored_graph_uses_shapes_and_explicit_folded_contract() -> None:
    graph = author_folded_scaled_matmul_graph(256, 80, 128)
    assert "tessera.scaled_matmul" in graph
    assert "tensor<256x128xui8>" in graph
    assert "tensor<80x128xui8>" in graph
    assert "block = [1, 128]" in graph
    assert "folded_row_reference_explicit_approximate" in graph
    with pytest.raises(ValueError, match="K divisible by 64"):
        author_folded_scaled_matmul_graph(256, 80, 96)


def test_exact_selector_does_not_silently_choose_approximate_folded_layout() -> None:
    exact = select_mxfp4_route(256, 80, 128)
    assert exact.accepted
    assert exact.abi_id is not None and "approx" not in exact.abi_id
    folded = select_mxfp4_route(
        256, 80, 128, requested_layout=mx.MXFP4_FOLDED_ROW_LAYOUT_V1,
    )
    assert not folded.accepted
    assert folded.abi_id is None
    assert "not an executable gfx1201 MXFP4 ABI" in folded.reason


def test_frontend_refuses_implicit_policy_and_bad_physical_inputs() -> None:
    a, a_scale, folded = _inputs()
    tool = Path("/does/not/exist/tessera-opt")
    with pytest.raises(ValueError, match="explicit approximate"):
        compile_folded_scaled_matmul(a, a_scale, folded, tessera_opt=tool)
    with pytest.raises(ValueError, match="raw E4M3"):
        compile_folded_scaled_matmul(
            a.astype(np.float32), a_scale, folded, tessera_opt=tool,
            allow_approximate=True,
        )
    with pytest.raises(ValueError, match="fp32"):
        compile_folded_scaled_matmul(
            a, a_scale.astype(np.float16), folded, tessera_opt=tool,
            allow_approximate=True,
        )
    with pytest.raises(ValueError, match="weight K"):
        compile_folded_scaled_matmul(
            np.ascontiguousarray(a[:, :32]), a_scale, folded, tessera_opt=tool,
            allow_approximate=True,
        )
    with pytest.raises(FileNotFoundError, match="compiler not found"):
        compile_folded_scaled_matmul(
            a, a_scale, folded, tessera_opt=tool, allow_approximate=True,
        )


def test_receipt_distinguishes_hsaco_bytes_from_composite_image_identity() -> None:
    payload = b"one emitted HSACO payload"
    payload_digest = hashlib.sha256(payload).hexdigest()
    image_digest = hashlib.sha256(b"HSACO plus target and toolchain").hexdigest()
    assert payload_digest != image_digest
    image = SimpleNamespace(
        payload=payload, payload_digest=payload_digest, image_digest=image_digest,
    )
    provenance = {
        "route": "folded_row_reference_bm256_tm4",
        "physical_contract": "rocm_mxfp4_w4a8_folded_prefill_v1",
        "block_m": 256, "block_n": 64, "block_k": 64,
        "tile_m_per_wave": 4, "tile_n_per_wave": 2,
        "schedule_hash": "schedule", "tile_ir_sha256": "tile",
        "target_ir_sha256": "target",
        "numeric_policy": "folded_row_reference_explicit_approximate",
        "fold_lossless": True, "fold_inexact_value_count": 0,
    }
    descriptor = SimpleNamespace(
        provenance=provenance, abi_id="folded-abi", entry_symbol="folded-entry",
    )
    package = SimpleNamespace(image=image, descriptor=descriptor)
    program = FoldedScaledMatmulProgram(package=package, graph_ir="graph")
    receipt = program.route_receipt
    assert receipt["hsaco_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["artifact_image_digest"] == image_digest
    assert receipt["graph_ir_sha256"] == hashlib.sha256(b"graph").hexdigest()
