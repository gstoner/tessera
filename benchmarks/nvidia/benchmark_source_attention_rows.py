"""Matched compiler-owned four/five-member source-attention reverse benchmark."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import time

import ml_dtypes
import numpy as np

from tessera.compiler.native_source_attention_reverse import export_native_source_attention_reverse
from tessera.compiler.prepared_source_attention_reverse import PreparedSourceAttentionReverse
from tests.device.nvidia.test_source_attention_reverse import oracle, source

PROFILES = {
    "small": (1, 2, 1, 3, 5, 4, 3),
    "tall": (1, 2, 1, 68, 5, 35, 7),
    "wide": (1, 2, 1, 17, 19, 257, 13),
    "medium": (1, 4, 2, 16, 32, 128, 32),
    "long": (1, 4, 2, 16, 64, 256, 32),
}


def checked(actual, expected, storage):
    for value, wanted in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value.astype(np.float32), wanted.astype(np.float32),
                                   rtol=.03 if storage == "bf16" else .006, atol=.002)


def run_case(name, dims, storage, output, samples, repeats, order_offset):
    dtype = np.float16 if storage == "f16" else ml_dtypes.bfloat16
    b, hq, hkv, sq, sk, d, dv = dims
    shapes = ((b, hq, sq, d), (b, hkv, sk, d), (b, hkv, sk, dv),
              (b, hq, sq, dv), (b, hq, sq))
    rng = np.random.default_rng(20261018)
    values = tuple(rng.normal(0, .3, shape).astype(np.float32 if i == 4 else dtype)
                   for i, shape in enumerate(shapes))
    snapshots = tuple(value.copy() for value in values)
    expected = oracle(*values)
    records, pins, owners = {}, {}, {}
    with ExitStack() as stack:
        for label, rows in (("inline", False), ("rows", True)):
            start = time.perf_counter()
            owner = stack.enter_context(PreparedSourceAttentionReverse(
                export_native_source_attention_reverse(source(values[:3], "normal", (0, 1, 2), rows=rows))))
            compiled_ms = (time.perf_counter() - start) * 1000
            checked(owner(*values), expected, storage)
            manifest = owner.manifest()
            archive = f"{storage}-{name}-{label}.json.gz"
            payload = json.dumps(manifest, sort_keys=True).encode()
            (output / archive).write_bytes(gzip.compress(payload, mtime=0))
            pins[label] = dict(contract_digest=manifest["contract_digest"],
                               program_schema=manifest["schema"], archive=archive,
                               archive_sha256=hashlib.sha256((output / archive).read_bytes()).hexdigest(),
                               compile_ms=compiled_ms, native_prepare_ms=owner.receipt["native_prepare_ms"],
                               images=[m.image_digest for m in owner._members])
            owners[label], records[label] = owner, []
        for index in (0, 1):
            assert owners["inline"]._members[index].image == owners["rows"]._members[index].image
        for sample in range(samples):
            order = ("inline", "rows") if (sample + order_offset) % 2 == 0 else ("rows", "inline")
            for label in order:
                stage, device, host = [], [], []
                start = time.perf_counter()
                for _ in range(repeats):
                    actual = owners[label](*values)
                    receipt = owners[label].receipt
                    stage.append(receipt["stage_device_ms"])
                    device.append(receipt["program_device_ms"])
                    host.append(receipt["completed_host_ms"])
                elapsed = (time.perf_counter() - start) * 1000 / repeats
                checked(actual, expected, storage)
                records[label].append(dict(sample=sample, end_to_end_ms=elapsed,
                    completed_native_host_ms=median(host), program_device_window_ms=median(device),
                    stage_device_window_ms=[median(row[i] for row in stage) for i in range(len(stage[0]))]))
        for actual, before in zip(values, snapshots, strict=True):
            np.testing.assert_array_equal(actual, before)
    medians = {label: {key: median(row[key] for row in rows)
                      for key in ("end_to_end_ms", "completed_native_host_ms", "program_device_window_ms")}
               for label, rows in records.items()}
    return dict(profile=name, storage=storage, dims=list(dims), pins=pins, samples=records,
                medians=medians, prefix_and_forward_images_identical=True,
                correctness="independent float64 oracle before and after timed samples",
                timing_scope="prepared source owner; event windows may include enqueue gaps; no frontend cost")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--order-offset", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    if args.samples < 7 or args.repeats < 1:
        raise ValueError("at least seven samples and one repeat are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    gpu = subprocess.check_output(["/usr/lib/wsl/lib/nvidia-smi",
        "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader"], text=True).strip()
    if len(gpu.splitlines()) != 1 or "RTX 5070" not in gpu or gpu.split(",")[2].strip() != "12.0":
        raise RuntimeError("owning RTX5070/SM120 required")
    result = dict(scope="named static homogeneous-half original-source AD policy A/B",
                  gpu=gpu, compiler_sha256=hashlib.sha256(Path(os.environ["TESSERA_OPT"]).read_bytes()).hexdigest(),
                  provider_sha256=hashlib.sha256(Path(os.environ["TESSERA_NVIDIA_PTX_LAUNCH_LIB"]).read_bytes()).hexdigest(),
                  recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  order_offset=args.order_offset, profiles=[])
    for storage in ("f16", "bf16"):
        for name, dims in PROFILES.items():
            row = run_case(name, dims, storage, args.output_dir, args.samples, args.repeats, args.order_offset)
            result["profiles"].append(row)
            (args.output_dir / "packet.json").write_text(json.dumps(result, indent=2) + "\n")
            print(storage, name, row["medians"], flush=True)


if __name__ == "__main__":
    main()

