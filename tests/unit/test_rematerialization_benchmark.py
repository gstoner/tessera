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


def test_workload_policy_uses_measured_cross_family_rows(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.rematerialization_cross_target._row",
        lambda target, size, epilogue, warmup, reps: {
            "target": target, "shape": [size] * 3, "operation": epilogue,
            "recompute_cost_ns": 1, "result_bytes": 1,
        },
    )
    monkeypatch.setattr(
        "benchmarks.rematerialization_cross_target._producer_family_row",
        lambda target, family, rows, width, warmup, reps, layer=0: {
            "target": target,
                "producer_family": family,
                "layer": layer,
                "shape": [rows, width],
            "operation": f"tessera.{family}_consumer",
            "recompute_cost_ns": {"softmax": 20, "rmsnorm": 10}[family],
            "result_bytes": 1024,
        },
    )
    payload = record(
        sizes=(), epilogues=(), producer_families=("softmax", "rmsnorm"),
        producer_shape=(8, 8), memory_budget_bytes=1024, warmup=0, reps=1,
    )
    assert len(payload["rows"]) == 4
    for decision in payload["workload_policy"]:
        assert decision["peak_before_bytes"] == 2048
        assert decision["peak_after_bytes"] == 1024
        assert decision["selected_operations"] == ["tessera.rmsnorm_consumer"]
        assert decision["selected_instances"][0]["layer"] == 0


def test_multilayer_workload_expands_exact_producer_instances(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.rematerialization_cross_target._producer_family_row",
        lambda target, family, rows, width, warmup, reps, layer=0: {
            "target": target, "producer_family": family, "layer": layer,
            "operation": f"tessera.{family}_consumer",
            "shape": [rows, width], "recompute_cost_ns": 10 + layer,
            "result_bytes": rows * width * 4,
        },
    )
    payload = record(
        sizes=(), epilogues=(), producer_families=("softmax", "rmsnorm"),
        producer_shape=(8, 16), layers=4, memory_budget_bytes=1024,
        warmup=0, reps=1,
    )
    assert len(payload["rows"]) == 2 * 4 * 2
    assert payload["workload"]["layers"] == 4
    assert {row["layer"] for row in payload["rows"]} == {0, 1, 2, 3}
    assert {tuple(row["shape"]) for row in payload["rows"]} == {
        (8, 16), (8, 144), (8, 272), (8, 400),
    }
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
