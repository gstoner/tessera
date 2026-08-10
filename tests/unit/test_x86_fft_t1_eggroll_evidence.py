"""Drift gates for the 2026-08-09 Zen 5 FFT, T1, and EGGROLL packets."""

from __future__ import annotations

import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_BASELINES = _ROOT / "benchmarks" / "baselines"


def _packet(name: str) -> dict:
    return json.loads((_BASELINES / name).read_text())


def test_mixed_radix_packet_covers_every_admitted_codelet() -> None:
    packet = _packet("x86_zen5_fft_mixed_codelets_2026_08_09.json")
    assert packet["schema"] == "tessera.x86.fft_mixed_codelets.v1"
    assert packet["architecture"] == "zen5-avx512"
    assert packet["codelet_radices"] == [2, 3, 4, 5, 7, 9, 11, 13, 15, 17]
    assert packet["all_correct"] is True
    assert packet["selector_changed"] is False
    assert packet["verdict"] == "retain specialized codelets; selector remains work-gated"
    assert {row["length"] for row in packet["rows"]} == {
        68, 225, 289, 768, 3072, 5120
    }
    assert all(row["correct"] for row in packet["rows"])
    assert len(packet["library_sha256"]) == 64


def test_t1_packet_preserves_the_negative_ranking_verdict() -> None:
    packet = _packet("x86_zen5_t1_cache_model_2026_08_09.json")
    assert packet["schema"] == "tessera.x86.t1_cache_model.v1"
    assert packet["architecture"] == "zen5-avx512"
    assert [level["name"] for level in packet["hardware_inputs"]["cache_levels"]] == [
        "L1D", "L2", "L3"
    ]
    assert packet["verdict"] == "reject_for_ranking"
    assert packet["median_spearman_rho"] < 0.0
    assert packet["winner_matches"] == 0
    assert packet["winner_trials"] == 3
    assert packet["selector_changed"] is False
    assert len(packet["library_sha256"]) == 64


def test_eggroll_packet_is_fp32_rank1_and_vnni_fail_closed() -> None:
    packet = _packet("x86_zen5_es_low_rank_2026_08_09.json")
    assert packet["schema"] == "tessera.eggroll.es_low_rank.zen5_avx512.v1"
    assert packet["architecture"] == "zen5-avx512"
    assert packet["rng"] == "splitmix64-philox4x32-boxmuller.v1"
    assert packet["dtype"] == {"storage": "fp32", "accum": "fp32"}
    assert packet["all_correct"] is True
    assert packet["verdict"]["correctness"] == "retain"
    assert packet["verdict"]["performance"] == "retain_avx512_f32"
    assert packet["verdict"]["selector_eligible"] is True
    assert packet["verdict"]["vnni"] == (
        "fail_closed_pending_quantization_scale_and_saturation_contract"
    )
    assert all(row["shape"]["rank"] == 1 for row in packet["rows"])
    assert all(row["correct"] for row in packet["rows"])
    assert len(packet["library_sha256"]) == 64
