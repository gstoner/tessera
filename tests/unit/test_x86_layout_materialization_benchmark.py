from benchmarks.x86.benchmark_x86_layout_materialization import record


def test_layout_benchmark_packet_schema(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.x86.benchmark_x86_layout_materialization._measure",
        lambda call, warmup, reps: (1.0, 1.1),
    )
    payload = record(size=4, warmup=1, reps=1)
    assert payload["schema"] == "tessera.x86.layout-materialization.v1"
    assert payload["column_major_speedup_vs_repack"] == 1.0
    assert len(payload["rows"]) == 2
