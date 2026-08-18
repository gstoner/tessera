from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "filename,target,architecture",
    [
        ("x86_zen5_w4_region_products_evidence.json", "x86", "zen5_avx512"),
        ("rocm_gfx1151_w4_region_products_evidence.json", "rocm_gfx1151", "gfx1151"),
    ],
)
def test_w4_region_product_packet_binds_and_consumes_residual_abi(
    filename: str, target: str, architecture: str
) -> None:
    packet = json.loads((ROOT / "benchmarks" / "baselines" / filename).read_text())
    assert packet["schema"] == "tessera.w4_region_products.evidence.v1"
    assert packet["work_item"] == "W4-PRODUCT-1"
    assert packet["target"] == target
    assert packet["architecture"] == architecture
    assert packet["promotion"]["correctness_eligible"] is True
    assert packet["promotion"]["performance_eligible"] is False
    rows = {row["name"]: row for row in packet["rows"]}
    assert set(rows) == {
        "dynamic_if_saved_activation_vjp",
        "hybrid_while_sparse_tape_vjp",
    }
    for row in rows.values():
        assert len(row["artifact_digest"]) == 64
        assert len(row["paired_ir_digest"]) == 64
        assert len(row["cfg_digest"]) == 64
        assert len(row["residual_abi_digest"]) == 64
        assert row["numerical"]["passed"] is True
        assert row["resources"]["graph_reentry"] is False
        assert row["resources"]["predicate_replay"] is False
        assert len(row["timing"]["samples_ns"]) >= 20
        assert row["timing"]["device_clock"]["valid"] is False
        assert row["timing"]["eligible_for_promotion"] is False
    branch = rows["dynamic_if_saved_activation_vjp"]
    assert branch["resources"]["inactive_residual_elements"] == 0
    loop = rows["hybrid_while_sparse_tape_vjp"]
    assert loop["resources"]["checkpoint_indices"] == [2, 4]
    assert loop["resources"]["replayed_steps"] == 2
