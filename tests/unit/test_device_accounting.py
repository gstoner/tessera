"""The hollow-lane gate must fire when it should and stay silent when it should.

A gate against unevaluated checks that is itself unevaluated would be the
joke telling itself. Both directions are asserted here: the negative cases
matter more than the positive one, because a gate that fires on an honest
Mac skip would be turned off within a day and then protect nothing.
"""

from __future__ import annotations

import pytest

from tests._support import device_accounting as da
from tests._support.policy import MARKERS


def _family(name: str = "nvidia", *, present: bool) -> da.DeviceFamily:
    return da.DeviceFamily(
        name=name,
        markers=frozenset({f"hardware_{name}"}),
        is_plausibly_present=lambda: present,
        remedy="source the env script",
    )


def _ledger(executed: int, skipped: int, *, name: str = "nvidia") -> da.DeviceLedger:
    ledger = da.DeviceLedger()
    for _ in range(executed):
        ledger.record([name], executed=True)
    for _ in range(skipped):
        ledger.record([name], executed=False)
    return ledger


def test_all_skipped_on_a_host_with_the_device_is_hollow():
    """The Super-Bear incident, reduced: skips only, hardware present."""
    families = (_family(present=True),)
    hollow = _ledger(executed=0, skipped=395).hollow_lanes(families)
    assert len(hollow) == 1
    family, tally = hollow[0]
    assert family.name == "nvidia"
    assert (tally.executed, tally.skipped) == (0, 395)


def test_absent_device_skipping_is_honest_not_hollow():
    """A Mac declining to make a CUDA claim is required behaviour, not a defect."""
    families = (_family(present=False),)
    assert _ledger(executed=0, skipped=395).hollow_lanes(families) == []


def test_one_real_execution_clears_the_lane():
    """Reachable hardware plus per-test gating is ordinary, not a misconfiguration."""
    families = (_family(present=True),)
    assert _ledger(executed=1, skipped=394).hollow_lanes(families) == []


def test_collecting_none_of_a_lane_makes_no_claim_about_it():
    """`pytest tests/unit/test_dtype.py` on the ROCm box must not fail over GPU lanes."""
    families = (_family(present=True),)
    assert da.DeviceLedger().hollow_lanes(families) == []
    assert _ledger(executed=0, skipped=0).hollow_lanes(families) == []


def test_a_failing_device_test_counts_as_executed():
    """The ledger asks whether the lane ran, not whether it was correct.

    A red device suite is already reporting a problem truthfully; adding a
    hollow-lane failure on top would obscure it.
    """
    families = (_family(present=True),)
    ledger = da.DeviceLedger()
    ledger.record(["nvidia"], executed=True)  # a failure still executed
    ledger.record(["nvidia"], executed=False)
    assert ledger.hollow_lanes(families) == []


@pytest.mark.parametrize(
    "keywords,expected",
    [
        ({"hardware_nvidia": True, "test_x": True}, {"nvidia"}),
        ({"hardware_rocm": True}, {"rocm"}),
        ({"hardware_apple_gpu": True}, {"apple_gpu"}),
        ({"metal4": True}, {"apple_gpu"}),
        ({"hardware_amx": True}, {"amx"}),
        ({"test_plain": True}, set()),
    ],
)
def test_marker_attribution(keywords, expected):
    assert set(da.families_for_keywords(keywords)) == expected


def test_every_hardware_marker_belongs_to_a_family():
    """A hardware marker outside every family is invisible to this gate.

    Decision #29 in spirit: the marker declares a hardware requirement, so
    something must consume it. Adding `hardware_<new>` to MARKERS without
    wiring a family here fails this test rather than silently creating a lane
    the ledger cannot see.
    """
    declared = {name for name in MARKERS if name.startswith("hardware_")} | {"metal4"}
    covered = set().union(*(f.markers for f in da.DEVICE_FAMILIES))
    assert declared == covered, (
        f"hardware markers with no device family: {sorted(declared - covered)}; "
        f"families naming an unknown marker: {sorted(covered - declared)}"
    )


def test_probes_are_callable_and_total_on_this_host():
    """Every probe must answer on any host rather than raising.

    A probe that throws would take down an unrelated session, which is how a
    safety gate earns its removal.
    """
    for family in da.DEVICE_FAMILIES:
        assert isinstance(family.is_plausibly_present(), bool)


def test_report_names_the_lane_and_the_remedy():
    """The failure text has to be actionable; the incident's cost was diagnosis."""
    families = (_family(present=True),)
    hollow = _ledger(executed=0, skipped=3).hollow_lanes(families)
    text = "\n".join(da.format_hollow_lane_report(hollow))
    assert "0 executed, 3 tests skipped" in text
    assert "source the env script" in text
    assert "proves nothing" in text
