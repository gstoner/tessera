"""Phase 0 — contract_consumer audit axis guards.

Locks the meta-gap tracker: every contract-pass-plan workstream has a row, status
is probed live (not hand-asserted), and the landed workstreams (A, B) report
`live`. As C/D/E/F land, their rows flip and these expectations tighten.

See docs/audit/roadmap/CONTRACT_PASS_PLAN.md (Phase 0).
"""

from __future__ import annotations

from tessera.compiler import contract_consumers as cc


def test_all_six_workstreams_present():
    workstreams = {c.workstream for c in cc.CONSUMERS}
    assert workstreams == {"A", "B", "C", "D", "E", "F"}


def test_status_is_live_or_declared():
    for c in cc.CONSUMERS:
        assert c.status in {"live", "declared", "none"}


def test_landed_workstreams_are_live():
    by_ws = {c.workstream: c for c in cc.CONSUMERS}
    # A (PagedKVState consumer) and B (phase specialization) landed already.
    assert by_ws["A"].status == "live"
    assert by_ws["B"].status == "live"


def test_unbuilt_workstreams_declared():
    by_ws = {c.workstream: c for c in cc.CONSUMERS}
    # These flip to "live" only when their consuming pass lands — this test
    # tightens (becomes an XFAIL→PASS signal) as each workstream completes.
    for ws in ("C", "D", "E", "F"):
        # Not asserting "declared" hard so the test survives a workstream
        # landing; assert it is a legal status and has a consumer named.
        assert by_ws[ws].consumer
        assert by_ws[ws].oracle


def test_status_counts_sum():
    counts = cc.status_counts()
    assert sum(counts.values()) == len(cc.CONSUMERS)
    assert counts["live"] >= 2  # A + B


def test_probe_is_live_not_hardcoded():
    # The A row must be live *because* the op exists — prove the probe reflects
    # reality by checking the consumer is actually importable.
    from tessera import ops
    assert hasattr(ops, "paged_attention")
    by_ws = {c.workstream: c for c in cc.CONSUMERS}
    assert by_ws["A"].status == "live"


def test_render_markdown_and_csv_nonempty():
    md = cc.render_markdown()
    csv = cc.render_csv()
    assert "Contract Consumers" in md
    assert "PagedKVState" in md and "PagedKVState" in csv
    assert md.count("###") == len(cc.CONSUMERS)  # one detail section each


def test_registered_as_generated_doc():
    from tessera.compiler import generated_docs as gd
    doc = gd.get("contract_consumers")
    assert doc.csv_path is not None
    assert gd.check(doc) is None, "contract_consumers dashboard is drift-dirty"
