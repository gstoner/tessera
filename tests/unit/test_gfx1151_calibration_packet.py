from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from tessera.compiler.profiler_provider_trace import (
    build_provider_trace_artifact,
    records_from_raw,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/calibration/calibrate_gfx1151.py"
SPEC = importlib.util.spec_from_file_location("calibrate_gfx1151", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CALIBRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CALIBRATION)


def _raw(*, environment: str = "bare_metal", dirty: bool = False) -> dict:
    def timing(kernel: str, result: float) -> dict:
        authoritative_seconds = 0.100 if environment == "bare_metal" else 0.1005
        row = {
            "kernel_name": kernel,
            "image_sha256": ("a" if kernel == "copy_bw" else "b") * 64,
            "host_wall_seconds": [0.101, 0.100],
            "hip_event_seconds": [0.100, 0.100],
            "host_wall_median_seconds": 0.1005,
            "hip_event_median_seconds": 0.100,
            "hip_event_valid": True,
            "host_event_relative_difference": 0.005,
            "timing_source": (
                "hip_event" if environment == "bare_metal" else "host_wall"
            ),
        }
        if kernel == "copy_bw":
            row["bytes_read_write"] = int(result * 1e9 * authoritative_seconds)
        else:
            row.update(
                {
                    "waves": 1,
                    "iterations": 1,
                    "chains": 1,
                    "flop_per_wmma": int(
                        result * 1e12 * authoritative_seconds
                    ),
                }
            )
        return row

    return {
        "schema": "tessera.gfx1151_calibration_measurement.v1",
        "architecture": "gfx1151",
        "requested_architecture": "gfx1151",
        "device_identity": {
            "query": "hipGetDevicePropertiesR0600",
            "ordinal": 0,
            "architecture": "gfx1151",
            "name": "AMD Radeon 8060S",
        },
        "device": "radeon_8060s",
        "execution_environment": environment,
        "measured_on": "2026-08-16",
        "host": "bare-metal-strix-halo",
        "source": {"source_commit": "a" * 40, "worktree_dirty": dirty},
        "results": {
            "dram_bw_gbps": 200.0,
            "peak_tflops.fp16:matrix": 48.0,
            "peak_tflops.bf16:matrix": 49.0,
        },
        "timings": {
            "dram_bw_gbps": timing("copy_bw", 200.0),
            "peak_tflops.fp16:matrix": timing("wmma_peak", 48.0),
            "peak_tflops.bf16:matrix": timing("wmma_peak", 49.0),
        },
    }


def _capture() -> dict:
    raw_records = [
        {
            "record_type": "dispatch",
            "activity": "dispatch",
            "kernel_name": kernel,
            "begin_ns": begin,
            "end_ns": begin + 100_000_000,
            "correlation_id": index,
        }
        for index, (kernel, begin) in enumerate(
            (
                ("copy_bw", 1_000),
                ("wmma_peak", 200_000_000),
                ("wmma_peak", 400_000_000),
            ),
            start=1,
        )
    ]
    return {
        "schema": "tessera.profiler_rocm_native_capture.v1",
        "provider": "rocprofiler",
        "status": "collected",
        "reason": None,
        "fresh_process": True,
        "process": {"clean_exit": True},
        "proof": {
            "dispatch_activity_seen": True,
            "hip_callback_seen": True,
            "hsa_callback_seen": False,
            "counter_records_seen": False,
            "pc_samples_seen": False,
        },
        "requested": {"counters": [], "pc_sampling": False},
        "provider_trace": build_provider_trace_artifact(
            provider="rocprofiler",
            records=records_from_raw("rocprofiler", raw_records),
            source_status="native",
        ),
        "application": [
            ".venv/bin/python",
            "benchmarks/calibration/calibrate_gfx1151.py",
            "--measurement-only",
            "/tmp/gfx1151-raw.json",
        ],
        "output_documents": [{"path": "trace.json", "sha256": "b" * 64}],
        "eligible_for_promotion": False,
    }


def test_bare_metal_device_event_and_profiler_packet_is_selector_eligible() -> None:
    corpus = CALIBRATION._finalize(
        _raw(),
        _capture(),
        allow_provisional=False,
        measurement_path=Path("/tmp/gfx1151-raw.json"),
    )
    assert corpus["selector_eligible"] is True
    assert corpus["evidence_tier"] == "selector_grade_bare_metal"
    assert corpus["ineligibility_reasons"] == []
    assert corpus["profiler_capture"]["proof"]["dispatch_activity_seen"] is True


def test_wsl_packet_cannot_be_finalized_as_selector_authority() -> None:
    with pytest.raises(ValueError, match="BARE_METAL_REQUIRED"):
        CALIBRATION._finalize(_raw(environment="wsl2"), None, allow_provisional=False)
    corpus = CALIBRATION._finalize(_raw(environment="wsl2"), None, allow_provisional=True)
    assert corpus["selector_eligible"] is False
    assert "BARE_METAL_REQUIRED" in corpus["ineligibility_reasons"]
    assert "ROCPROFILER_ACTIVITY_NOT_COLLECTED" in corpus["ineligibility_reasons"]


def test_dirty_source_is_not_selector_eligible() -> None:
    with pytest.raises(ValueError, match="SOURCE_WORKTREE_DIRTY"):
        CALIBRATION._finalize(
            _raw(dirty=True),
            _capture(),
            allow_provisional=False,
            measurement_path=Path("/tmp/gfx1151-raw.json"),
        )


def test_profiler_capture_must_bind_raw_path_and_both_kernel_families() -> None:
    capture = _capture()
    capture["application"][-1] = "/tmp/different.json"
    with pytest.raises(ValueError, match="ROCPROFILER_MEASUREMENT_LINEAGE_MISSING"):
        CALIBRATION._finalize(
            _raw(),
            capture,
            allow_provisional=False,
            measurement_path=Path("/tmp/gfx1151-raw.json"),
        )


def test_metric_rows_and_results_are_bound_exactly() -> None:
    raw = _raw()
    raw["timings"] = {f"unrelated_{index}": row for index, row in enumerate(raw["timings"].values())}
    with pytest.raises(ValueError, match="exact bandwidth"):
        CALIBRATION._finalize(
            raw,
            _capture(),
            allow_provisional=False,
            measurement_path=Path("/tmp/gfx1151-raw.json"),
        )

    raw = _raw()
    raw["results"]["dram_bw_gbps"] = 201.0
    with pytest.raises(ValueError, match="disagrees"):
        CALIBRATION._finalize(
            raw,
            _capture(),
            allow_provisional=False,
            measurement_path=Path("/tmp/gfx1151-raw.json"),
        )


def test_device_architecture_must_come_from_hip_properties() -> None:
    raw = _raw()
    raw["device_identity"]["architecture"] = "gfx1200"
    with pytest.raises(ValueError, match="queried gfx1151 identity"):
        CALIBRATION._finalize(
            raw,
            _capture(),
            allow_provisional=False,
            measurement_path=Path("/tmp/gfx1151-raw.json"),
        )
