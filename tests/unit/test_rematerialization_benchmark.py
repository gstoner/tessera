from benchmarks.rematerialization_cross_target import record
from tessera.compiler.rematerialization_cost import (
    measured_rematerialization_cost_ns,
)


def test_rematerialization_packet_schema(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.rematerialization_cross_target._row",
        lambda target, size, epilogue, warmup, reps: {
            "target": target,
            "shape": [size, size, size],
            "operation": epilogue,
            "recompute_cost_ns": 1,
        },
    )
    payload = record(sizes=(8,), epilogues=("relu",), warmup=1, reps=1)
    assert payload["schema"] == "tessera.rematerialization.cross-target.v1"
    assert [row["target"] for row in payload["rows"]] == ["x86", "rocm"]


def test_rematerialization_packet_covers_shape_and_operation_grid(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.rematerialization_cross_target._row",
        lambda target, size, epilogue, warmup, reps: {
            "target": target,
            "shape": [size, size, size],
            "operation": epilogue,
        },
    )
    payload = record(
        sizes=(64, 128), epilogues=("relu", "silu"), warmup=0, reps=1
    )
    assert len(payload["rows"]) == 8
    assert {
        (row["target"], tuple(row["shape"]), row["operation"])
        for row in payload["rows"]
    } == {
        (target, (size, size, size), epilogue)
        for target in ("x86", "rocm")
        for size in (64, 128)
        for epilogue in ("relu", "silu")
    }


def test_exact_measured_cost_is_available_to_graph_compilation():
    assert (
        measured_rematerialization_cost_ns(
            "x86", "tessera.matmul", (128, 128, 128),
            consumer="tessera.matmul_relu",
        )
        is not None
    )
    assert (
        measured_rematerialization_cost_ns(
            "rocm", "tessera.matmul", (128, 128, 128),
            consumer="tessera.matmul_relu",
        )
        is not None
    )
    assert (
        measured_rematerialization_cost_ns(
            "rocm", "tessera.matmul", (129, 128, 128),
            consumer="tessera.matmul_relu",
        )
        is None
    )
