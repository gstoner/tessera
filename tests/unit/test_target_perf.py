"""Gate for ``compiler/target_perf.py`` — calibrated performance parameters.

Two properties this file protects, both about *honesty* rather than accuracy:

* **No invented numbers.** A field we have not measured or derived must read back
  as absent, and asking for it must raise a diagnostic naming the gap — never a
  silent zero or a plausible-looking default (Decision #21).
* **Measured beats spec, visibly.** A calibration overlay must win over the spec
  value *and* must be reported as measured, so a roofline built on vendor sheets
  can never be mistaken for one built on silicon.

See ``docs/audit/compiler/TILESIGHT_ASSESSMENT.md`` §2/§4.
"""
from __future__ import annotations

import json

import pytest

from tessera.compiler.capabilities import TARGET_CAPABILITIES
from tessera.compiler.schedule_planner import SchedulePlanner
from tessera.compiler.target_perf import (
    CORPUS_VERSION,
    UNIT_MATRIX,
    UNIT_VECTOR,
    Provenance,
    TargetPerf,
    TargetPerfError,
    KIND_CORPUS,
    KIND_SNAPSHOT,
    SNAPSHOT_VERSION,
    apply_corpus,
    apply_registry_snapshot,
    devices_for_target,
    load_corpus,
    load_registry_snapshot,
    peak_key,
    registry_snapshot,
    perf_for_device,
    perf_for_target,
    register_perf,
    registered_devices,
    reset_registry,
)


# --- registry integrity ------------------------------------------------------

def test_every_device_targets_a_canonical_target_id() -> None:
    """A device pointing at a target id ``normalize_target`` does not know about
    is unreachable — it would silently never be selected."""
    for name in registered_devices():
        perf = perf_for_device(name)
        assert perf.target in TARGET_CAPABILITIES, (
            f"{name} targets {perf.target!r}, not a canonical target id")


def test_every_device_carries_a_source_citation() -> None:
    """An uncited performance number is indistinguishable from a guess."""
    for name in registered_devices():
        assert len(perf_for_device(name).source) > 20, f"{name} has a stub source"


def test_peak_keys_are_canonical_dtypes() -> None:
    for name in registered_devices():
        for key in perf_for_device(name).peak_tflops:
            dtype, _, unit = key.partition(":")
            assert peak_key(dtype, unit) == key


def test_peak_key_rejects_non_canonical_dtype_and_unit() -> None:
    with pytest.raises(ValueError, match="canonical"):
        peak_key("bfloat16", UNIT_MATRIX)          # alias, not canonical
    with pytest.raises(ValueError, match="canonical"):
        peak_key("tf32", UNIT_MATRIX)              # not a storage dtype at all
    with pytest.raises(ValueError, match="unit"):
        peak_key("bf16", "tensor_core")


def test_the_three_fleet_boxes_are_registered() -> None:
    """Theory §6's three systems each need a profile for W7 attainment."""
    for target in ("nvidia_sm120", "rocm_gfx1151", "apple_gpu"):
        assert devices_for_target(target), f"no device registered for {target}"


# --- the no-invented-numbers rule -------------------------------------------

def test_provenance_is_per_field_not_per_row() -> None:
    """Regression: provenance was stored per row, so a row labelled DERIVED
    reported `derived` for its spec-sheet core count *and* for a value its own
    source string says was confirmed on-silicon. Under-reporting a measured
    value is exactly the failure this module exists to prevent."""
    ti = perf_for_device("rtx_5070_ti")
    assert ti.spec_provenance is Provenance.DERIVED
    # The FLOP identity really is derived...
    assert ti.provenance_of("peak_tflops.fp32:vector") is Provenance.DERIVED
    # ...the core count is a published figure...
    assert ti.provenance_of("compute_units") is Provenance.SPEC
    # ...and the SMEM capacity was read off the silicon.
    assert ti.provenance_of("smem_bytes_per_cu") is Provenance.MEASURED


def test_apple_bandwidth_is_spec_not_derived() -> None:
    """The 512-bit x 6.4 GT/s identity gives 409.6 GB/s; Apple publishes 400. It
    is their number, so it must not claim to be one we rederived."""
    m1 = perf_for_device("m1_max")
    assert m1.value("dram_bw_gbps") == 400.0
    assert m1.provenance_of("dram_bw_gbps") is Provenance.SPEC
    assert m1.provenance_of("peak_tflops.fp32:vector") is Provenance.DERIVED


def test_field_provenance_rejects_bad_keys_and_unknown() -> None:
    with pytest.raises(ValueError, match="unknown field_provenance field"):
        TargetPerf(device="d", target="x86", source="a source long enough to pass",
                   field_provenance={"dram_bandwidth": Provenance.SPEC})
    with pytest.raises(ValueError, match="cannot be UNKNOWN"):
        TargetPerf(device="d", target="x86", source="a source long enough to pass",
                   field_provenance={"dram_bw_gbps": Provenance.UNKNOWN})


def test_measured_overlay_still_outranks_a_field_provenance_entry() -> None:
    ti = perf_for_device("rtx_5070_ti")
    assert ti.provenance_of("compute_units") is Provenance.SPEC
    swept = ti.with_measured({"compute_units": 70.0}, on="2026-07-28")
    assert swept.provenance_of("compute_units") is Provenance.MEASURED


def test_absent_field_reads_as_none_not_zero() -> None:
    zen5 = perf_for_device("ryzen_ai_max_395_cpu")
    assert zen5.peak("fp32", UNIT_VECTOR) is None
    assert zen5.provenance_of("peak_tflops.fp32:vector") is Provenance.UNKNOWN


def test_require_raises_naming_the_gap_and_the_fix() -> None:
    zen5 = perf_for_device("ryzen_ai_max_395_cpu")
    with pytest.raises(TargetPerfError) as exc:
        zen5.require("peak_tflops.fp32:vector")
    message = str(exc.value)
    assert "ryzen_ai_max_395_cpu" in message
    assert "calibration" in message


def test_derived_quantities_decline_rather_than_fabricate() -> None:
    """A ridge point or attainment computed from a missing peak would be a
    fabricated number wearing a real number's clothes."""
    zen5 = perf_for_device("ryzen_ai_max_395_cpu")
    assert zen5.ridge_point("fp32", UNIT_VECTOR) is None
    assert zen5.attainment(5.0, "fp32", UNIT_VECTOR) is None
    assert zen5.roofline_ms(1e12, 1e9, "fp32", UNIT_VECTOR) is None


def test_unknown_device_lists_what_is_registered() -> None:
    with pytest.raises(TargetPerfError, match="registered:"):
        perf_for_device("h200_sxm5")


def test_unknown_field_name_raises_the_documented_error_type() -> None:
    """Regression: `value()` raised a bare ValueError for a bad field name, so
    `require()` did too — contradicting its documented `TargetPerfError`
    contract and slipping past callers catching the specific type."""
    a100 = perf_for_device("a100_sxm4_80gb")
    with pytest.raises(TargetPerfError, match="unknown TargetPerf field"):
        a100.value("dram_bandwidth")
    with pytest.raises(TargetPerfError, match="unknown TargetPerf field"):
        a100.require("dram_bandwidth")
    with pytest.raises(TargetPerfError):
        a100.provenance_of("dram_bandwidth")


def test_ambiguous_target_refuses_to_pick_a_sibling(monkeypatch) -> None:
    """``nvidia_sm120`` covers a 5070 Ti and an RTX PRO 6000; guessing between
    them is a >2x error, so the lookup must refuse."""
    sibling = TargetPerf(
        device="_test_sm120_sibling", target="nvidia_sm120",
        source="synthetic row for the ambiguity test — not a real part",
        peak_tflops={"fp32:vector": 1.0})
    register_perf(sibling)
    try:
        with pytest.raises(TargetPerfError, match="disambiguate"):
            perf_for_target("nvidia_sm120")
        # ...but naming the device resolves it.
        assert perf_for_target("nvidia_sm120", device="rtx_5070_ti").device == "rtx_5070_ti"
    finally:
        reset_registry()


def test_unregistered_target_returns_none_not_a_default() -> None:
    assert perf_for_target("rocm_gfx90a") is None


# --- derived quantities are arithmetically right -----------------------------

def test_ridge_point_and_roofline_bound() -> None:
    a100 = perf_for_device("a100_sxm4_80gb")
    # 312 TFLOP/s over 2039 GB/s ~= 153 FLOP/byte.
    ridge = a100.ridge_point("bf16", UNIT_MATRIX)
    assert ridge is not None and 150.0 < ridge < 156.0

    # A kernel far below the ridge is memory-bound: the bound must come from the
    # bandwidth term, which the FLOPs-only estimator cannot express.
    memory_bound = a100.roofline_ms(flops=1e9, bytes_moved=2.039e9,
                                    dtype="bf16", unit=UNIT_MATRIX)
    assert memory_bound == pytest.approx(1.0, rel=1e-3)   # 2.039 GB / 2039 GB/s

    compute_bound = a100.roofline_ms(flops=3.12e11, bytes_moved=1e3,
                                     dtype="bf16", unit=UNIT_MATRIX)
    assert compute_bound == pytest.approx(1.0, rel=1e-3)  # 312 GFLOP / 312 TFLOP/s


def test_attainment_is_a_fraction_of_peak() -> None:
    a100 = perf_for_device("a100_sxm4_80gb")
    assert a100.attainment(156.0, "bf16", UNIT_MATRIX) == pytest.approx(0.5)


def test_derived_peaks_match_their_stated_identity() -> None:
    """The DERIVED rows claim an exact identity in their ``source``; check it,
    so a transcription slip cannot sit in the table looking authoritative."""
    for device, lanes in (("rtx_5070_ti", 8960), ("m1_max", 4096)):
        perf = perf_for_device(device)
        assert perf.clock_ghz is not None
        expected = lanes * 2 * perf.clock_ghz / 1e3  # TFLOP/s
        got = perf.peak("fp32", UNIT_VECTOR)
        assert got is not None
        assert got == pytest.approx(expected, rel=0.02), (
            f"{device}: stated {got} vs identity {expected}")

    # AMD: cus x 64 lanes x 2 flop/clk x clock
    amd = perf_for_device("radeon_8060s")
    assert amd.compute_units is not None and amd.clock_ghz is not None
    expected_amd = amd.compute_units * 64 * 2 * amd.clock_ghz / 1e3
    got_amd = amd.peak("fp32", UNIT_VECTOR)
    assert got_amd is not None
    assert got_amd == pytest.approx(expected_amd, rel=0.02)


# --- the measured overlay ----------------------------------------------------

def test_measured_wins_over_spec_and_is_reported_as_measured() -> None:
    a100 = perf_for_device("a100_sxm4_80gb")
    assert a100.provenance_of("dram_bw_gbps") is Provenance.SPEC

    calibrated = a100.with_measured({"dram_bw_gbps": 1700.0}, on="2026-07-28",
                                    host="test-host")
    assert calibrated.value("dram_bw_gbps") == 1700.0
    assert calibrated.provenance_of("dram_bw_gbps") is Provenance.MEASURED
    # ...and the ridge point moves with it, which is the whole point.
    assert calibrated.ridge_point("bf16", UNIT_MATRIX) != a100.ridge_point(
        "bf16", UNIT_MATRIX)
    # The original is untouched (frozen dataclass, copy-on-write).
    assert a100.value("dram_bw_gbps") == 2039.0


def test_measured_overlay_rejects_a_typoed_field() -> None:
    """A sweep whose key is misspelled would silently land nowhere while the spec
    value kept being reported."""
    a100 = perf_for_device("a100_sxm4_80gb")
    with pytest.raises(ValueError, match="unknown measured field"):
        a100.with_measured({"dram_bandwidth": 1700.0}, on="2026-07-28")
    with pytest.raises(ValueError, match="canonical"):
        a100.with_measured({"peak_tflops.bfloat16:matrix": 250.0}, on="2026-07-28")


def test_measured_overlay_requires_a_date() -> None:
    with pytest.raises(ValueError, match="measured_on"):
        TargetPerf(device="d", target="x86", source="a source long enough to pass",
                   measured={"dram_bw_gbps": 1.0})


# --- calibration corpus ------------------------------------------------------

def test_apply_corpus_merges_and_reports_updated_devices() -> None:
    try:
        updated = apply_corpus({
            "version": CORPUS_VERSION,
            "measured_on": "2026-07-28",
            "host": "test-host",
            "devices": {"a100_sxm4_80gb": {"dram_bw_gbps": 1700.0}},
        })
        assert updated == ["a100_sxm4_80gb"]
        after = perf_for_device("a100_sxm4_80gb")
        assert after.value("dram_bw_gbps") == 1700.0
        assert after.measured_host == "test-host"
    finally:
        reset_registry()


def test_reset_registry_undoes_a_corpus_merge() -> None:
    """The registry is process-global, so merging a corpus needs a defined way
    back — otherwise a corpus applied in one place silently colours every later
    lookup, and tests have to reach into private state to clean up."""
    original = perf_for_device("a100_sxm4_80gb").value("dram_bw_gbps")
    apply_corpus({
        "version": CORPUS_VERSION,
        "measured_on": "2026-07-28",
        "devices": {"a100_sxm4_80gb": {"dram_bw_gbps": 1700.0}},
    })
    assert perf_for_device("a100_sxm4_80gb").value("dram_bw_gbps") == 1700.0
    reset_registry()
    assert perf_for_device("a100_sxm4_80gb").value("dram_bw_gbps") == original
    assert perf_for_device("a100_sxm4_80gb").measured == {}


def test_reset_registry_drops_an_ad_hoc_registration() -> None:
    register_perf(TargetPerf(device="_scratch", target="x86",
                             source="a synthetic row long enough to pass"))
    assert "_scratch" in registered_devices()
    reset_registry()
    assert "_scratch" not in registered_devices()


def test_corpus_rejects_version_skew_unknown_device_and_missing_date() -> None:
    with pytest.raises(ValueError, match="version"):
        apply_corpus({"version": 999, "measured_on": "2026-07-28", "devices": {}})
    with pytest.raises(ValueError, match="measured_on"):
        apply_corpus({"version": CORPUS_VERSION, "devices": {}})
    with pytest.raises(TargetPerfError, match="registered:"):
        apply_corpus({"version": CORPUS_VERSION, "measured_on": "2026-07-28",
                      "devices": {"no_such_gpu": {"dram_bw_gbps": 1.0}}})


def test_load_corpus_names_a_missing_file(tmp_path) -> None:
    with pytest.raises(TargetPerfError, match="not found"):
        load_corpus(tmp_path / "absent.json")


def test_apply_corpus_is_atomic_across_devices() -> None:
    """Regression: the merge registered row-by-row, so a corpus whose later row
    named an unknown device left the earlier rows applied. The registry is
    process-global, so a caller that caught the error was left holding a
    half-applied calibration silently colouring every later lookup."""
    before = perf_for_device("a100_sxm4_80gb").value("dram_bw_gbps")
    try:
        with pytest.raises(TargetPerfError, match="registered:"):
            apply_corpus({
                "version": CORPUS_VERSION, "measured_on": "2026-07-28",
                "devices": {
                    "a100_sxm4_80gb": {"dram_bw_gbps": 1700.0},  # valid, first
                    "no_such_gpu": {"dram_bw_gbps": 1.0},        # invalid, later
                },
            })
        assert perf_for_device("a100_sxm4_80gb").value("dram_bw_gbps") == before
        assert perf_for_device("a100_sxm4_80gb").measured == {}
    finally:
        reset_registry()


def test_apply_corpus_is_atomic_across_bad_fields() -> None:
    """Same guarantee when the second row's *field* is bad rather than its
    device."""
    before = perf_for_device("a100_sxm4_80gb").value("dram_bw_gbps")
    try:
        with pytest.raises(ValueError, match="unknown measured field"):
            apply_corpus({
                "version": CORPUS_VERSION, "measured_on": "2026-07-28",
                "devices": {
                    "a100_sxm4_80gb": {"dram_bw_gbps": 1700.0},
                    "h100_sxm5": {"dram_bandwidth": 3000.0},
                },
            })
        assert perf_for_device("a100_sxm4_80gb").value("dram_bw_gbps") == before
    finally:
        reset_registry()


# --- registry snapshot (distinct from a calibration corpus) ------------------

def test_registry_snapshot_round_trips_through_its_own_loader() -> None:
    """Regression: the snapshot was shaped and versioned like a corpus but could
    not be loaded by one — it omitted the required `measured_on` and its rows are
    whole profiles, not measured-field overlays. Writing the documented
    fleet-shared snapshot and reading it straight back raised."""
    text = json.dumps(registry_snapshot())   # JSON-clean: no enums leaking
    reloaded = json.loads(text)
    assert reloaded["kind"] == KIND_SNAPSHOT
    assert reloaded["version"] == SNAPSHOT_VERSION

    try:
        restored = apply_registry_snapshot(reloaded)
        assert set(restored) == set(registered_devices())
        # Provenance and overlays survive the trip.
        ti = perf_for_device("rtx_5070_ti")
        assert ti.provenance_of("smem_bytes_per_cu") is Provenance.MEASURED
        assert ti.provenance_of("compute_units") is Provenance.SPEC
    finally:
        reset_registry()


def test_snapshot_survives_a_calibration_overlay() -> None:
    """A snapshot taken after a sweep must carry the measured values, else the
    fleet-shared artifact loses exactly the data worth sharing."""
    try:
        apply_corpus({"version": CORPUS_VERSION, "measured_on": "2026-07-28",
                      "host": "test-host",
                      "devices": {"a100_sxm4_80gb": {"dram_bw_gbps": 1700.0}}})
        snap = json.loads(json.dumps(registry_snapshot()))
        reset_registry()
        assert perf_for_device("a100_sxm4_80gb").value("dram_bw_gbps") == 2039.0

        apply_registry_snapshot(snap)
        after = perf_for_device("a100_sxm4_80gb")
        assert after.value("dram_bw_gbps") == 1700.0
        assert after.provenance_of("dram_bw_gbps") is Provenance.MEASURED
        assert after.measured_host == "test-host"
    finally:
        reset_registry()


def test_each_loader_refuses_the_other_artifact_by_name() -> None:
    """The two payloads are not interchangeable; feeding one to the other must
    say so rather than fail with a confusing downstream error."""
    snap = registry_snapshot()
    with pytest.raises(ValueError, match="apply_registry_snapshot"):
        apply_corpus(snap)

    corpus = {"kind": KIND_CORPUS, "version": CORPUS_VERSION,
              "measured_on": "2026-07-28", "devices": {}}
    with pytest.raises(ValueError, match="apply_corpus"):
        apply_registry_snapshot(corpus)


def test_snapshot_rejects_version_skew_and_malformed_rows() -> None:
    with pytest.raises(ValueError, match="version"):
        apply_registry_snapshot({"kind": KIND_SNAPSHOT, "version": 999,
                                 "devices": {}})
    with pytest.raises(ValueError, match="not a valid profile"):
        apply_registry_snapshot({"kind": KIND_SNAPSHOT,
                                 "version": SNAPSHOT_VERSION,
                                 "devices": {"x": {"device": "x"}}})
    with pytest.raises(ValueError, match="does not match"):
        apply_registry_snapshot({
            "kind": KIND_SNAPSHOT, "version": SNAPSHOT_VERSION,
            "devices": {"mislabelled": {
                "device": "other", "target": "x86",
                "source": "a synthetic row long enough to pass"}}})


def test_snapshot_is_atomic() -> None:
    """A snapshot with a valid first row and a broken second must register
    neither."""
    good = perf_for_device("a100_sxm4_80gb").to_dict()
    good["dram_bw_gbps"] = 1.0
    try:
        with pytest.raises(ValueError, match="not a valid profile"):
            apply_registry_snapshot({
                "kind": KIND_SNAPSHOT, "version": SNAPSHOT_VERSION,
                "devices": {"a100_sxm4_80gb": good, "broken": {"device": "broken"}}})
        assert perf_for_device("a100_sxm4_80gb").value("dram_bw_gbps") == 2039.0
    finally:
        reset_registry()


def test_load_registry_snapshot_reads_a_file(tmp_path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(registry_snapshot()))
    try:
        assert "m1_max" in load_registry_snapshot(path)
    finally:
        reset_registry()

    with pytest.raises(TargetPerfError, match="not found"):
        load_registry_snapshot(tmp_path / "absent.json")


# --- consumption: the schedule planner --------------------------------------

def test_planner_for_target_uses_the_right_hardware_peak() -> None:
    """The bare constructor's default is A100 dense bf16 applied to everything;
    ``for_target`` must source the actual device's numbers."""
    default = SchedulePlanner()
    assert default.peak_tflops == 312.0
    assert default.perf is None

    apple = SchedulePlanner.for_target("apple_gpu", dtype="fp32")
    assert apple.perf is not None and apple.perf.device == "m1_max"
    assert apple.peak_tflops == pytest.approx(10.62)
    assert apple.smem_budget_bytes == 32768


def test_planner_for_target_refuses_a_missing_peak_rather_than_defaulting() -> None:
    """bf16 matrix peaks on the fleet boxes are genuinely unknown — the planner
    must say so instead of silently reusing an A100 number."""
    with pytest.raises(TargetPerfError, match="calibration sweep"):
        SchedulePlanner.for_target("nvidia_sm120", dtype="bf16")


def test_planner_for_target_reports_an_unprofiled_target() -> None:
    with pytest.raises(TargetPerfError, match="no performance profile"):
        SchedulePlanner.for_target("rocm_gfx90a", dtype="fp32")


def test_planner_refuses_to_inherit_another_devices_smem_budget() -> None:
    """Regression: `for_target` silently fell back to 98_304 bytes when the
    profile had no shared-memory capacity — inside a constructor whose stated
    contract is refusing silent defaults. A CPU lane would have inherited an
    A100-shaped budget and every legality verdict would be wrong."""
    cpu = TargetPerf(device="_smemless", target="x86",
                     source="a synthetic row with a peak but no SMEM capacity",
                     peak_tflops={"fp32:vector": 2.0})
    register_perf(cpu)
    try:
        with pytest.raises(TargetPerfError, match="smem_bytes_per_cu"):
            SchedulePlanner.for_target("x86", dtype="fp32", device="_smemless")
        # ...but the caller may state the budget rather than inherit one.
        planner = SchedulePlanner.for_target(
            "x86", dtype="fp32", device="_smemless", smem_budget_bytes=32_768)
        assert planner.smem_budget_bytes == 32_768
        assert planner.peak_tflops == pytest.approx(2.0)
    finally:
        reset_registry()
