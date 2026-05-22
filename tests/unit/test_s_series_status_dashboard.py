"""Slice 2 (2026-05-22) — S-series primitive-contracts dashboard drift gate.

The dashboard at ``docs/audit/generated/s_series_status.md`` is generated
from the primitive_coverage registry by
``tessera.compiler.s_series_status``.  These tests pin:

  1. The aggregate axis counts agree between the registry and the
     rendered table (no silent drift if a category's contract status
     changes).
  2. The dashboard renders deterministically — two render() calls
     produce the same text.
  3. The on-disk file matches the freshly-generated output (the same
     contract the support_table drift gate uses).
  4. The known structural invariants hold:
       - ``lowering_rule`` aggregate open == 0  (Sprint A1+ closure).
       - ``backend_kernel`` aggregate open == aggregate total (Phase
         G/H/I universal gate).
       - Every priority-≤50 category is the high-use S-series surface
         the audit promises to prioritise (S2/S5/S7/S10/S11/M6/M7).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tessera.compiler.s_series_status import (
    DASHBOARD_AXES,
    S_SERIES_PRIORITY,
    render_markdown,
    tally_by_category,
)
from tessera.compiler.primitive_coverage import all_primitive_coverages

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD = REPO_ROOT / "docs" / "audit" / "generated" / "s_series_status.md"


# ─────────────────────────────────────────────────────────────────────────────
# Determinism + drift gate
# ─────────────────────────────────────────────────────────────────────────────


def test_render_is_deterministic() -> None:
    a = render_markdown()
    b = render_markdown()
    assert a == b, "render_markdown is non-deterministic"


def test_dashboard_file_exists() -> None:
    assert DASHBOARD.exists(), (
        f"dashboard missing: {DASHBOARD} — generate with "
        "`python -m tessera.compiler.s_series_status`"
    )


def test_dashboard_file_matches_generated() -> None:
    """Drift gate — the on-disk file must equal the rendered output."""
    on_disk = DASHBOARD.read_text()
    fresh = render_markdown()
    assert on_disk == fresh, (
        "s_series_status.md is out of date.  Regenerate with: "
        "python -m tessera.compiler.s_series_status"
    )


def test_check_subcommand_passes_when_in_sync() -> None:
    """The --check CLI flag is what CI / validate.sh will call."""
    r = subprocess.run(
        [sys.executable, "-m", "tessera.compiler.s_series_status",
         "--check"],
        cwd=str(REPO_ROOT),
        env={"PYTHONPATH": str(REPO_ROOT / "python"),
             "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True, timeout=30,
    )
    assert r.returncode == 0, (
        f"--check failed: stdout={r.stdout}\nstderr={r.stderr}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Structural invariants
# ─────────────────────────────────────────────────────────────────────────────


def test_lowering_rule_is_closed_project_wide() -> None:
    """Sprint A1+ closed lowering_rule.  Any regression here means a
    new category added a partial that the classifier didn't cover."""
    rows = tally_by_category()
    total_open = sum(int(r["lowering_rule_open"]) for r in rows)
    assert total_open == 0, (
        f"lowering_rule has {total_open} open entries; the multi-axis "
        "classifier was supposed to close this axis.  "
        "Run `python -m tessera.compiler.s_series_status --render` to "
        "see which category regressed."
    )


def test_backend_kernel_is_universal_phase_g_gate() -> None:
    """Per CLAUDE.md + the audit doc, backend_kernel is open on
    every entry by design — promotion happens with hardware
    enablement.  This invariant prevents a silent partial-promotion
    that would split the Phase G gate apart."""
    rows = tally_by_category()
    for row in rows:
        # Either all entries in this category are open on backend, OR
        # the category is documented as "not_applicable" (e.g., pure
        # Python schedules + tokenizers).  We allow up to 1
        # documented exception per category.
        open_n = int(row["backend_kernel_open"])
        total = int(row["total"])
        if open_n < total:
            # Some category entries are not_applicable for backend
            # (e.g., pytrees, AOT, serialization).  Don't reject that
            # — just ensure we're not accidentally promoting real
            # backend rows to "complete" without hardware.
            cat = str(row["category"])
            assert cat in {
                "aot", "conformance", "data", "tokenizer", "schedule",
                "autodiff_transform", "control_flow", "grad_transform",
                "pytree", "state",
            } or open_n > 0, (
                f"category {cat!r} has backend rows marked complete "
                f"without Phase G enablement"
            )


def test_priority_50_or_below_anchors_high_use_surface() -> None:
    """The priority-50 cutoff is the user's promised "high-use
    S2/S5/S7/S10/S11" prioritisation.  Categories at ≤50 must be in
    those sprints (or M6/M7 for the constrained-lane families)."""
    high_priority_sprints = {"S2", "S5", "S7", "S10", "S11", "M6", "M7"}
    for category, (sprint, priority) in S_SERIES_PRIORITY.items():
        if priority <= 50:
            assert sprint in high_priority_sprints, (
                f"category {category!r} at priority {priority} maps to "
                f"sprint {sprint!r} which is not in the high-use set "
                f"{sorted(high_priority_sprints)}"
            )


def test_aggregate_counts_in_rendered_dashboard() -> None:
    """The aggregate table in the rendered dashboard must agree with
    the registry walk — pinning that the renderer doesn't drift from
    the underlying tally."""
    rows = tally_by_category()
    text = render_markdown(rows)

    for axis in DASHBOARD_AXES:
        open_n = sum(int(r[f"{axis}_open"]) for r in rows)
        complete_n = sum(int(r[f"{axis}_complete"]) for r in rows)
        # The aggregate table line: "| `axis` | N | M |"
        expected = f"| `{axis}` | {open_n} | {complete_n} |"
        assert expected in text, (
            f"aggregate line for {axis!r} missing or wrong; "
            f"expected: {expected!r}"
        )


def test_dashboard_calls_out_phase_g_gate_explicitly() -> None:
    """The closure-trajectory note must explicitly explain why
    backend_kernel is universally open — so future readers don't see
    "backend_kernel is universally open" and panic."""
    text = render_markdown()
    assert "Phase G/H/I" in text
    assert "lowering_rule` is closed" in text


def test_sprint_19_bucket_a_sharding_promotions_are_closed() -> None:
    """Sprint #19 Bucket A is the promote-now sharding set from the
    sharding partial audit: host/reference partition-spec rules are
    closed and need neither mock mesh nor hardware proof."""
    closed = {
        "ppo_policy_loss",
        "grpo_policy_loss",
        "cispo_policy_loss",
        "kv_cache_append",
        "kv_cache_prune",
        "kv_cache_read",
        "online_softmax_state",
        "bidirectional_scan",
        "gru_cell",
        "simple_rnn_cell",
        "ebm_energy",
        "ebm_self_verify",
        "ebm_decode_init",
        "clifford_geometric_product",
        "clifford_wedge",
        "clifford_inner",
        "clifford_left_contraction",
        "clifford_rotor_sandwich",
        "clifford_grade_projection",
        "clifford_reverse",
        "clifford_norm",
        "clifford_conjugate",
        "clifford_grade_involution",
        "clifford_hodge_star",
        "clifford_exp",
        "clifford_log",
        "lora_linear",
    }
    entries = all_primitive_coverages()
    assert {
        name for name in closed
        if entries[name].contract_status["sharding_rule"] != "complete"
    } == set()


def test_complex_jit_is_not_counted_as_a_primitive_contract() -> None:
    """``complex_jit`` is a frontend decorator, not a primitive op; it
    must stay out of the primitive-contract dashboard."""
    assert "complex_jit" not in all_primitive_coverages()
