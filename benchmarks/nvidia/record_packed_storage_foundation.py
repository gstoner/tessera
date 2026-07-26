"""Record reproducible SM120 generic packed-view execution evidence.

The first end-to-end observation is discarded. Device-event observations each
amortize ten resident launches; the retained end-to-end median covers ten
allocation/copy/launch/copy-back submissions. WSL rows are evidence only and
are never selector eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
OUT = ROOT / "benchmarks/baselines/nvidia_sm120_packed_storage_foundation.json"
STABILITY_LIMIT_PCT = 4.0


def _case(logical: str) -> tuple[object, dict[str, object]]:
    from tessera.compiler.nvidia_native import package_packed_decode
    from tessera.runtime import RuntimeArtifact

    # Large enough that a ten-launch CUDA-event observation measures kernel
    # work rather than event quantization, while retaining odd ragged extents.
    rows, columns = 4097, 4099
    factor = 1 if logical.startswith("fp6_") else 2
    stride = (columns + factor - 1) // factor + 1
    source = np.zeros(2 + (rows + 1) * stride, dtype=np.uint8)
    scaled = logical != "int4"
    scale_stride, scale_offset = (columns + 4) // 4 + 1, 1
    scale = (
        np.full(
            scale_offset + (rows + 1) * scale_stride,
            np.uint8(7 << 3 if logical == "nvfp4" else 127),
            dtype=np.uint8,
        )
        if scaled
        else np.zeros(1, dtype=np.uint8)
    )
    output = np.zeros((rows, columns), dtype=np.float32)
    package = package_packed_decode(
        logical=logical,
        rows=rows,
        columns=columns,
        source_bytes=source.size,
        strides=(stride, 1),
        offset=2,
        scale_dtype=(
            "ue4m3" if logical == "nvfp4" else "ue8m0" if scaled else "none"
        ),
        scale_bytes=scale.size,
        scale_block_size=4 if scaled else 0,
        scale_axis=1 if scaled else -1,
        scale_layout="row_major" if scaled else "none",
        scale_stride=scale_stride if scaled else 0,
        scale_offset=scale_offset if scaled else 0,
    )
    artifact = RuntimeArtifact(
        tile_ir=package.tile_ir,
        target_ir=package.target_ir,
        metadata={"target": "nvidia_sm120"},
        native_image=package.image,
        launch_descriptor=package.descriptor,
    )
    args: dict[str, object] = {
        "source": source,
        "scale": scale,
        "output": output,
        "RowOrigin": 1,
        "ColumnOrigin": 1,
        "Rows": rows,
        "Columns": columns,
        "SourceBytes": source.size,
        "ScaleBytes": scale.size,
    }
    return package, {"artifact": artifact, "args": args}


def record(*, observations: int = 7, amortized_launches: int = 10) -> dict[str, object]:
    from tessera.runtime import (
        _nvidia_device_name,
        _nvidia_native_descriptor_device_latency,
        _nvidia_native_descriptor_resources,
        launch,
    )

    rows: list[dict[str, object]] = []
    for logical in ("int4", "nvfp4", "fp6_e2m3", "fp6_e3m2"):
        package, state = _case(logical)
        artifact = state["artifact"]
        args = state["args"]
        first = launch(artifact, args)
        if not first.get("ok"):
            raise RuntimeError(str(first.get("reason")))
        device_samples = [
            _nvidia_native_descriptor_device_latency(
                package.image,
                package.descriptor,
                args,
                warmup=1,
                reps=amortized_launches,
            )
            for _ in range(observations)
        ]
        e2e_samples: list[float] = []
        for _ in range(amortized_launches + 1):
            start = time.perf_counter_ns()
            result = launch(artifact, args)
            elapsed_ms = (time.perf_counter_ns() - start) / 1.0e6
            if not result.get("ok"):
                raise RuntimeError(str(result.get("reason")))
            e2e_samples.append(elapsed_ms)
        retained_e2e = e2e_samples[1:]
        resources = _nvidia_native_descriptor_resources(
            package.image, package.descriptor, block_size=128,
        )
        row: dict[str, object] = {
            "candidate": f"generic_packed_decode_{logical}",
            "logical_dtype": logical,
            "selected_route": "compiler_owned_tile_packed_load",
            "device_event_ms": statistics.median(device_samples),
            "device_event_samples_ms": device_samples,
            "end_to_end_ms": statistics.median(retained_e2e),
            "end_to_end_samples_ms": retained_e2e,
            "discarded_first_end_to_end_ms": e2e_samples[0],
            "amortized_launches_per_device_observation": amortized_launches,
            "resources": resources,
            "compile_state": package.image.compile_state,
            "image_digest": package.image.image_digest,
            "descriptor_digest": package.descriptor.descriptor_digest,
            "cache_fingerprint": package.descriptor.cache_fingerprint,
            "selector_eligible": False,
            "selector_disposition": "retain",
            "host_policy": "wsl_evidence_only",
        }
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        row["evidence_fingerprint"] = (
            "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        )
        rows.append(row)
    return {
        "schema": "tessera.nvidia.packed-storage-foundation.v1",
        "device": _nvidia_device_name(),
        "arch": "sm_120a",
        "timing_policy": {
            "discard_first_end_to_end": True,
            "amortized_launches": amortized_launches,
            "device_observations": observations,
            "aggregate": "median",
        },
        "selector_policy": "WSL evidence is selector-ineligible",
        "rows": rows,
    }


def attach_reference(
    payload: dict[str, object], reference: dict[str, object]
) -> dict[str, object]:
    """Attach a second independent cohort without making it selector evidence."""

    if reference.get("schema") != payload.get("schema"):
        raise ValueError("reference schema does not match the current packet")
    if reference.get("device") != payload.get("device"):
        raise ValueError("reference device does not match the current packet")
    reference_rows = {
        str(row["candidate"]): row for row in reference.get("rows", [])
    }
    stable_count = 0
    for row in payload.get("rows", []):
        candidate = str(row["candidate"])
        if candidate not in reference_rows:
            raise ValueError(f"reference is missing candidate {candidate}")
        prior = reference_rows[candidate]
        stability: dict[str, float] = {}
        for domain in ("device_event_ms", "end_to_end_ms"):
            current_value = float(row[domain])
            reference_value = float(prior[domain])
            denominator = max(abs(current_value), abs(reference_value), 1.0e-12)
            stability[domain] = (
                abs(current_value - reference_value) / denominator * 100.0
            )
        stable = all(value <= STABILITY_LIMIT_PCT for value in stability.values())
        stable_count += int(stable)
        row["cross_run"] = {
            "reference_medians_ms": {
                "device_event": prior["device_event_ms"],
                "end_to_end": prior["end_to_end_ms"],
            },
            "stability_pct": {
                "device_event": stability["device_event_ms"],
                "end_to_end": stability["end_to_end_ms"],
            },
            "stable_both_domains": stable,
            "resource_identity": row["resources"] == prior["resources"],
            "image_identity": row["image_digest"] == prior["image_digest"],
            "descriptor_identity": (
                row["descriptor_digest"] == prior["descriptor_digest"]
            ),
            "cache_identity": (
                row["cache_fingerprint"] == prior["cache_fingerprint"]
            ),
            "reference_compile_state": prior["compile_state"],
        }
        row.pop("evidence_fingerprint", None)
        canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
        row["evidence_fingerprint"] = (
            "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
        )
    payload["stability_policy"] = {
        "limit_pct": STABILITY_LIMIT_PCT,
        "required_domains": ["device_event", "end_to_end"],
        "stable_rows": stable_count,
        "total_rows": len(payload.get("rows", [])),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=int, default=7)
    parser.add_argument("--amortized-launches", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--reference",
        type=Path,
        help="independent packet used to calculate two-run stability",
    )
    args = parser.parse_args(argv)
    if args.observations < 3 or args.amortized_launches < 2:
        parser.error("require at least 3 observations and 2 amortized launches")
    payload = record(
        observations=args.observations,
        amortized_launches=args.amortized_launches,
    )
    if args.reference is not None:
        payload = attach_reference(
            payload, json.loads(args.reference.read_text(encoding="utf-8"))
        )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
