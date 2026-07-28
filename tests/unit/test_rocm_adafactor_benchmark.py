import json
from pathlib import Path

from benchmarks.rocm.benchmark_rocm_adafactor import artifact, backward_artifact


def test_adafactor_benchmark_pins_four_entry_runtime_route() -> None:
    metadata = artifact().metadata
    assert metadata["compiler_path"] == "rocm_adafactor_compiled"
    assert metadata["ops"][0]["op_name"] == "tessera.adafactor"
    assert metadata["ops"][0]["operands"] == [
        "parameter", "gradient", "row", "col"
    ]


def test_adafactor_backward_benchmark_pins_physical_adjoint_route() -> None:
    metadata = backward_artifact().metadata
    assert metadata["compiler_path"] == "rocm_adafactor_bwd_compiled"
    assert metadata["ops"][0]["out_cotangent"] == "dy"


def test_adafactor_gfx1151_baseline_is_operation_total_evidence() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (root / "benchmarks/baselines/rocm_gfx1151_adafactor.json").read_text()
    )
    assert payload["schema"] in {
        "tessera.rocm.adafactor.benchmark.v1",
        "tessera.rocm.adafactor.benchmark.v2",
    }
    assert payload["device"] == "gfx1151"
    assert payload["timing_scope"] == "operation_total_ms"
    assert payload["selector_eligible"] is False
    assert payload["median_ms"] > 0.0
    assert len(payload["samples_ms"]) == payload["iterations"]


def test_adafactor_backward_gfx1151_packet_is_operation_total_evidence() -> None:
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (
            root
            / "benchmarks/baselines/rocm_gfx1151_adafactor_backward.json"
        ).read_text()
    )
    assert payload["schema"] == "tessera.rocm.adafactor.benchmark.v2"
    assert payload["device"] == "gfx1151"
    assert payload["direction"] == "backward"
    assert payload["scope_detail"].endswith(
        "seven_launches+synchronize+result_copies"
    )
    assert payload["selector_eligible"] is False
    assert payload["median_ms"] > 0.0
