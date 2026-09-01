"""Static coverage/schema pins for the gfx1151 compiler retune ratchets."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "benchmarks" / "rocm" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_f32_retune_covers_square_rectangular_and_ragged():
    bench = _load("benchmark_rocm_f32_retune")
    assert (256, 256, 256) in bench.SHAPES
    assert (128, 512, 256) in bench.SHAPES
    assert any(m % 16 or n % 16 or k % 16 for m, n, k in bench.SHAPES)
    assert {(2, 2), (4, 4), (6, 4)} <= set(bench.TILES)


def test_grouped_and_swiglu_retunes_cover_transition_and_model_rows():
    grouped = _load("benchmark_rocm_grouped_gemm_retune")
    names = {case[0] for case in grouped.CASES}
    assert {"balanced_small", "transition_64k", "transition_64k_high_k",
            "balanced_model", "ragged_model", "wide_ffn",
            "narrow_down"} <= names
    swiglu = _load("benchmark_rocm_swiglu_retune")
    assert len(swiglu.CASES) >= 2


def test_transport_and_attention_ratchets_pin_required_matrix():
    transport = _load("benchmark_rocm_transport_retune")
    assert {(16, 128), (128, 1024), (16, 4096)} <= set(transport.CASES)
    g6b = _load("benchmark_rocm_g6b_two_wave")
    assert (1, 16, 1024, 128, False) in g6b.CASES
    assert (1, 16, 1009, 128, True) in g6b.CASES
    g6c = _load("benchmark_rocm_g6c_split_reduced")
    assert any(h != g for _, h, g, _, _, _ in g6c.CASES)
    assert {causal for *_, causal in g6c.CASES} == {False, True}


def test_consolidated_retune_baseline_records_all_decisions():
    path = (ROOT / "benchmarks" / "baselines" /
            "rocm_gfx1151_compiler_retune_2026_07_15.json")
    data = json.loads(path.read_text())
    assert data["schema"] == "tessera.rocm.compiler_retune.v1"
    assert data["device"] == "gfx1151"
    assert {"f32_gemm", "grouped_gemm", "grouped_swiglu",
            "kv_moe_transport", "g6b", "g6c"} <= data.keys()
    assert "promote" in data["g6b"]["decision"]
    assert "reject" in data["g6c"]["decision"]
    assert data["g6b"]["resources"]["two_wave_d128"]["vgpr_spills"] == 0


def test_retune_decisions_agree_with_the_evidence_they_kept():
    """`method.promotion_gate` is prose, so nothing held a verdict to it.

    Apple's equivalent block carries thresholds; this one carries the sentence
    "correct oracle plus shape-specific repeated-median gate", which no loader
    can re-derive. What each family *does* carry is a per-row speedup and win
    rate, and that is enough to check the direction of its own decision.

    Both directions are checked deliberately: a rejection is a claim too. If a
    regressed recording showed the rejected g6c candidate winning outright, the
    committed verdict would be wrong and nothing today would notice.
    """
    from tests._support.rocm_evidence import promotion_verdict_violations

    root = Path(__file__).parents[2]
    payload = json.loads((root / "benchmarks/baselines"
                          / "rocm_gfx1151_compiler_retune_2026_07_15.json"
                          ).read_text())

    # The reading of each decision is stated, not parsed: a reworded decision
    # must fail here rather than quietly change what is enforced.
    expectations = {
        "grouped_swiglu": (True, "promote three grouped GEMMs"),
        "g6b": (True, "promote two-wave plain/causal D=128"),
        "g6c": (False, "reject production promotion"),
    }
    for name, (expect_promotion, prefix) in expectations.items():
        family = payload[name]
        assert family["decision"].startswith(prefix), (
            f"{name}'s decision was reworded; re-read it and update the "
            f"expectation rather than the prefix: {family['decision']!r}")
        assert promotion_verdict_violations(
            family, expect_promotion=expect_promotion) == [], name


def test_winner_only_families_are_named_as_uncheckable_not_as_passing():
    """Two families kept no losing measurement, so their gate is unverifiable.

    `f32_gemm` and `grouped_gemm` record one configuration per shape, and their
    decisions name thresholds -- "2x2 only for square sizes through 256",
    "tn=1 below 64k outputs" -- that have nothing to be compared against. A
    checker that skipped them silently would report full coverage over an
    artifact half of which cannot be checked, which is the failure this whole
    line of work is about.

    Closing it needs a re-record on gfx1151 keeping both candidates per shape;
    until then this test pins the gap so it cannot be mistaken for coverage.
    """
    from tests._support.rocm_evidence import winner_only_families

    root = Path(__file__).parents[2]
    payload = json.loads((root / "benchmarks/baselines"
                          / "rocm_gfx1151_compiler_retune_2026_07_15.json"
                          ).read_text())
    families = ("f32_gemm", "grouped_gemm", "grouped_swiglu",
                "kv_moe_transport", "g6b", "g6c")
    assert winner_only_families(payload, families) == [
        "f32_gemm", "grouped_gemm", "kv_moe_transport"]


def test_the_verdict_re_derivation_catches_a_contradicted_decision():
    """The positive test above asserts `== []`, which a broken checker also
    satisfies -- so on its own it proves the artifact is consistent, not that
    anything checked it. Stubbing `promotion_verdict_violations` to return
    nothing leaves that test green. This is the case that does not.
    """
    import copy

    from tests._support.rocm_evidence import promotion_verdict_violations

    root = Path(__file__).parents[2]
    payload = json.loads((root / "benchmarks/baselines"
                          / "rocm_gfx1151_compiler_retune_2026_07_15.json"
                          ).read_text())

    # A rejected family whose rows were regressed into winning everywhere:
    # the evidence now says promote and the verdict still says reject.
    forged_win = copy.deepcopy(payload["g6c"])
    for row in forged_win["rows"]:
        row["device_speedup"] = row["e2e_speedup"] = 3.0
        row["device_win_rate"] = 1.0
    assert promotion_verdict_violations(
        forged_win, expect_promotion=False) == ["rejected_although_every_row_wins"]

    # A promoted family with one shape that does not actually win.
    forged_loss = copy.deepcopy(payload["grouped_swiglu"])
    forged_loss["rows"][0]["device_speedup"] = 0.8
    assert any("promoted_without_a_win" in violation for violation in
               promotion_verdict_violations(forged_loss, expect_promotion=True))

    # A promoted family whose win rate is not unanimous.
    forged_rate = copy.deepcopy(payload["g6b"])
    forged_rate["rows"][1]["win_rate"] = 0.5
    assert any("promoted_without_a_win" in violation for violation in
               promotion_verdict_violations(forged_rate, expect_promotion=True))

    # Rows stripped: unverifiable is not the same as fine.
    stripped = copy.deepcopy(payload["g6b"])
    stripped["rows"] = []
    assert promotion_verdict_violations(
        stripped, expect_promotion=True) == ["missing_rows"]

    # A row carrying no comparative field at all is named, not skipped.
    blank = copy.deepcopy(payload["g6b"])
    blank["rows"] = [{"shape": [1, 2, 3, 4]}]
    assert promotion_verdict_violations(blank, expect_promotion=True) == [
        "row[0]:no_comparative_evidence"]
