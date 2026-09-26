"""Guard rails for functional-complete alpha (MASTER_AUDIT).

The scoreboard measures the release definition; these tests make its guard
rails binding. Guard rail 1: bypass surfaces only shrink. Guard rails 2-3:
the per-stage and per-lane native counts only grow. Both directions are
pinned exactly against ``alpha_ratchet_baseline.json`` -- a regression fails,
and an improvement fails until the baseline is tightened, so a gain cannot be
silently given back later.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tessera.compiler import alpha_scoreboard as board

_BASELINE_PATH = Path(__file__).with_name("alpha_ratchet_baseline.json")
_SHRINK = (
    "graph_input_packagers",
    "delegating_packagers",
    "source_emitters",
    "target_ops_without_required_contract",
)
_REGEN = (
    "regenerate the baseline with: PYTHONPATH=python python3 -c \"import json; "
    "from tessera.compiler import alpha_scoreboard as a; s=a.summary(); "
    "s.pop('cells'); open('tests/unit/alpha_ratchet_baseline.json','w')"
    ".write(json.dumps(s, indent=2, sort_keys=True)+'\\n')\""
)


@pytest.fixture(scope="module")
def current() -> dict:
    return board.summary()


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("metric", _SHRINK)
def test_bypass_surfaces_only_shrink(current, baseline, metric):
    now, pinned = current[metric], baseline[metric]
    assert now <= pinned, (
        f"guard rail 1: {metric} rose {pinned} -> {now}. A change may not add a "
        "bypass (Python package_* constructor, source emitter, delegating "
        "packager, or Target IR op with an optional contract).")
    assert now == pinned, (
        f"{metric} improved {pinned} -> {now}: tighten the baseline to lock the "
        f"gain -- {_REGEN}")


def _grow_pairs(current: dict, baseline: dict):
    yield "complete", current["complete"], baseline["complete"]
    for group in ("stage_native", "lane_native"):
        assert set(current[group]) == set(baseline[group]), (
            f"{group} keys changed; the lane/stage definition moved -- {_REGEN}")
        for key in current[group]:
            yield f"{group}.{key}", current[group][key], baseline[group][key]


def test_native_counts_never_regress(current, baseline):
    regressions = [f"{name}: {pinned} -> {now}"
                   for name, now, pinned in _grow_pairs(current, baseline)
                   if now < pinned]
    assert not regressions, (
        "guard rails 2-3: a stage or lane lost native coverage: "
        + "; ".join(regressions))


def test_native_gains_are_locked(current, baseline):
    gains = [f"{name}: {pinned} -> {now}"
             for name, now, pinned in _grow_pairs(current, baseline)
             if now > pinned]
    assert not gains, f"native coverage improved ({'; '.join(gains)}) -- {_REGEN}"


def test_the_lane_definition_is_the_owners_eight():
    """Eight lanes on four machines, CPU + GPU each (MASTER_AUDIT)."""
    assert len(board.LANES) == 8
    hosts = {lane.host for lane in board.LANES}
    assert hosts == {"Mac M1 Max", "Princess-Luna", "The-Super-Bear", "Tajasarus"}
    for host in hosts:
        assert sum(1 for lane in board.LANES if lane.host == host) == 2
    zen2 = next(lane for lane in board.LANES if lane.lane == "bear_cpu")
    assert "AVX-512" not in zen2.device, "Zen 2 has no AVX-512"


def test_the_board_is_not_vacuous():
    cells = board.cells()
    assert len(cells) == len(board.LANES) * len(board.ALPHA_FAMILIES)
    assert {c.stages[0] for c in cells}, "no frontend state derived"
    assert any(c.stages[2] == board.DONE for c in cells), (
        "no compiled Schedule->Tile route found anywhere -- the route source broke")


def test_a_cell_is_complete_only_when_every_stage_is_native():
    ok = board.Cell("x", "f", (board.DONE,) * len(board.STAGES))
    assert ok.complete
    for i in range(len(board.STAGES)):
        stages = [board.DONE] * len(board.STAGES)
        stages[i] = "unmeasured"
        assert not board.Cell("x", "f", tuple(stages)).complete


def test_a_stale_alias_fails_closed(monkeypatch):
    """An alias naming a family its module lacks must raise, never count."""
    monkeypatch.setitem(board.ALPHA_FAMILIES, "matmul",
                        {"apple_gpu": ("no_such_family",)})
    with pytest.raises(board.AlphaScoreboardError, match="no_such_family"):
        board.cells()


def test_a_lane_without_a_spine_target_fails_closed(monkeypatch):
    bad = board.Lane("ghost", "nowhere", "none", "x86", "no_such_target", None)
    monkeypatch.setattr(board, "LANES", (*board.LANES, bad))
    with pytest.raises(board.AlphaScoreboardError, match="no_such_target"):
        board.cells()


def test_unrouted_is_worse_than_bypass_for_a_multi_family_alias():
    routes = {("t", "a"): "native", ("t", "b"): "bypass"}
    lane = board.Lane("l", "h", "d", "t", "t", None)
    saved = board.ALPHA_FAMILIES.get("__probe__")
    board.ALPHA_FAMILIES["__probe__"] = {"t": ("a", "b")}
    try:
        assert board._schedule_tile(routes, lane, "__probe__") == "bypass"
        board.ALPHA_FAMILIES["__probe__"] = {"t": ("a", "missing")}
        assert board._schedule_tile(routes, lane, "__probe__") == "unrouted"
    finally:
        if saved is None:
            del board.ALPHA_FAMILIES["__probe__"]
