"""Host-free: a calibration corpus's eligibility is stated, consistent, declared.

EVIDENCE-PACKET-1 slice (sync EVIDENCE-GOVERNANCE-GATES-2026-09-27).
`target_perf.apply_corpus` is the consumer that turns a calibration corpus into
selector authority (measured dram_bw / peak TFLOP/s that the schedule planner
prices against). It read eligibility as ``corpus.get("selector_eligible",
True)`` and never read ``ineligibility_reasons`` at all, so

* a corpus that simply omitted ``selector_eligible`` became selector authority;
* a corpus stating ``selector_eligible: true`` beside a list of reasons it was
  not eligible was believed;
* a reason tag nobody declared was carried as if it were no reason.

`load_pruning_corpus` (inspection only) read neither field. Both now go through
`corpus_selector_eligibility`, which refuses each case by a registered code.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tessera.compiler.target_perf import (
    CORPUS_VERSION,
    apply_corpus,
    corpus_selector_eligibility,
    load_pruning_corpus,
    perf_for_device,
    reset_registry,
)

ROOT = Path(__file__).resolve().parents[2]
_COMMITTED = ROOT / "benchmarks/baselines/rocm_gfx1151_calibration_2026_08_15.json"


def _committed() -> dict:
    return json.loads(_COMMITTED.read_text())


def _write(tmp_path: Path, corpus: dict) -> Path:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(corpus))
    return path


def _bare_metal_corpus(**overrides) -> dict:
    """A corpus apply_corpus accepts, so each refusal below is the new check."""
    corpus = {
        "kind": "calibration_corpus",
        "version": CORPUS_VERSION,
        "measured_on": "2026-07-28",
        "execution_environment": "bare_metal",
        "host": "test-host",
        "selector_eligible": True,
        "ineligibility_reasons": [],
        "devices": {"a100_sxm4_80gb": {"dram_bw_gbps": 1700.0}},
        "measurements": {"a100_sxm4_80gb": {"results": {"dram_bw_gbps": 1700.0},
                                            "execution_environment": "bare_metal"}},
    }
    corpus.update(overrides)
    return corpus


def test_the_committed_corpus_still_reads_and_still_cannot_promote(tmp_path: Path) -> None:
    corpus = _committed()
    assert corpus_selector_eligibility(corpus) is False
    assert load_pruning_corpus(_COMMITTED)  # inspection still works
    with pytest.raises(ValueError, match="pruning-only"):
        apply_corpus(corpus)


def test_a_complete_eligible_corpus_is_applied() -> None:
    """The control for every refusal below."""
    try:
        assert apply_corpus(_bare_metal_corpus()) == ["a100_sxm4_80gb"]
    finally:
        reset_registry()


@pytest.mark.parametrize("field", ["selector_eligible", "ineligibility_reasons"])
def test_an_eligible_corpus_missing_either_field_is_refused(field: str) -> None:
    """The fail-open case: before 2026-09-27 this corpus was applied."""
    corpus = _bare_metal_corpus()
    del corpus[field]
    before = perf_for_device("a100_sxm4_80gb").measured
    with pytest.raises(ValueError, match="CALIBRATION_CORPUS_ELIGIBILITY_INCOMPLETE"):
        apply_corpus(corpus)
    assert perf_for_device("a100_sxm4_80gb").measured == before


@pytest.mark.parametrize("value", ["true", 1, None])
def test_a_non_bool_eligibility_is_refused(value) -> None:
    with pytest.raises(ValueError, match="CALIBRATION_CORPUS_ELIGIBILITY_INCOMPLETE"):
        apply_corpus(_bare_metal_corpus(selector_eligible=value))


@pytest.mark.parametrize("reasons", ["SOURCE_WORKTREE_DIRTY", [""], [3]])
def test_malformed_reasons_are_refused(reasons) -> None:
    with pytest.raises(ValueError, match="CALIBRATION_CORPUS_ELIGIBILITY_INCOMPLETE"):
        apply_corpus(_bare_metal_corpus(ineligibility_reasons=reasons))


def test_eligible_beside_reasons_is_refused() -> None:
    with pytest.raises(ValueError, match="CALIBRATION_CORPUS_ELIGIBILITY_CONTRADICTED"):
        apply_corpus(_bare_metal_corpus(ineligibility_reasons=["SOURCE_WORKTREE_DIRTY"]))


def test_ineligible_without_a_reason_is_refused(tmp_path: Path) -> None:
    doctored = _committed()
    doctored["ineligibility_reasons"] = []
    with pytest.raises(ValueError, match="CALIBRATION_CORPUS_ELIGIBILITY_CONTRADICTED"):
        load_pruning_corpus(_write(tmp_path, doctored))


def test_an_undeclared_reason_is_refused(tmp_path: Path) -> None:
    doctored = _committed()
    doctored["ineligibility_reasons"].append("SOMETHING_NOBODY_DECLARED")
    with pytest.raises(ValueError, match="CALIBRATION_CORPUS_REASON_UNKNOWN"):
        load_pruning_corpus(_write(tmp_path, doctored))
    with pytest.raises(ValueError, match="CALIBRATION_CORPUS_REASON_UNKNOWN"):
        apply_corpus(doctored)


@pytest.mark.parametrize("field", ["selector_eligible", "ineligibility_reasons"])
def test_the_committed_corpus_missing_a_field_is_refused_by_inspection_too(
        tmp_path: Path, field: str) -> None:
    doctored = copy.deepcopy(_committed())
    del doctored[field]
    with pytest.raises(ValueError, match="CALIBRATION_CORPUS_ELIGIBILITY_INCOMPLETE"):
        load_pruning_corpus(_write(tmp_path, doctored))


def test_detail_suffixed_tags_are_read_by_their_tag() -> None:
    corpus = _committed()
    corpus["ineligibility_reasons"] = ["HIP_DEVICE_EVENT_INVALID:bw_copy",
                                       "ROCPROFILER_KERNEL_CORRELATION_MISSING:copy_bw"]
    assert corpus_selector_eligibility(corpus) is False
