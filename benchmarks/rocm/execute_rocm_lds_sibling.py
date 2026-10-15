"""Execute compiler-owned gfx1151 LDS sibling packages on the owning device."""

from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import ml_dtypes
from tessera import runtime as rt
from tessera.compiler.native_artifact import NativeImageArtifact, LaunchDescriptor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert rt._rocm_live_arch() == rt._rocm_chip() == "gfx1151"
    packages = json.loads(args.packages.read_text())
    if packages.get("schema") != "tessera.rocm.lds_sibling_packages.v1":
        raise ValueError("requires the native gfx1151 LDS sibling package schema")
    rows = []
    for record in packages["rows"]:
        m, k, n = record["shape_mkn"]
        dtype = np.float16 if record["dtype"] == "fp16" else ml_dtypes.bfloat16
        rng = np.random.default_rng(9471151 + m)
        a = rng.uniform(-0.5, 0.5, (m, k)).astype(dtype)
        b = rng.uniform(-0.5, 0.5, (k, n)).astype(dtype)
        original_a, original_b = a.copy(), b.copy()
        a.flags.writeable = b.flags.writeable = False
        output = np.zeros((m, n), np.float32)
        image = NativeImageArtifact.from_dict(packages["images"][record["image_digest"]])
        descriptor = LaunchDescriptor.from_dict(record["descriptor"])
        assert image.architecture == "gfx1151" and descriptor.provenance["staging"] == "lds"
        artifact = rt.RuntimeArtifact(
            metadata={"target": image.target},
            native_image=image,
            launch_descriptor=descriptor,
            tile_ir=record["tile_ir"],
            target_ir=record["target_ir"],
        )
        receipt = rt.launch(artifact, {"buffers": {"a": a, "b": b, "o": output}, "scalars": {"M": m, "N": n, "K": k}})
        assert receipt["ok"] and receipt["execution_kind"] == "native_gpu", receipt
        expected = a.astype(np.float32) @ b.astype(np.float32)
        np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-3)
        np.testing.assert_array_equal(a, original_a)
        np.testing.assert_array_equal(b, original_b)
        row = {
            "dtype": record["dtype"],
            "shape_mkn": record["shape_mkn"],
            "image_digest": image.image_digest,
            "entry": descriptor.entry_symbol,
            "abi_id": descriptor.abi_id,
            "correctness": "passed",
            "execution_kind": receipt["execution_kind"],
            "max_abs_error": float(np.max(np.abs(output - expected))),
            "target_ir_sha256": hashlib.sha256(record["target_ir"].encode()).hexdigest(),
            "tile_ir_sha256": hashlib.sha256(record["tile_ir"].encode()).hexdigest(),
        }
        rows.append(row)
        print(row, flush=True)
    for dtype in ("fp16", "bf16"):
        assert len({r["image_digest"] for r in rows if r["dtype"] == dtype}) == 1
    bad_image = replace(image, target="rocm_gfx1201", architecture="gfx1201")
    bad_descriptor = replace(descriptor, image_digest=bad_image.image_digest)
    try:
        bad_artifact = rt.RuntimeArtifact(
            metadata={"target": bad_image.target},
            native_image=bad_image,
            launch_descriptor=bad_descriptor,
            tile_ir=record["tile_ir"],
            target_ir=record["target_ir"],
        )
        rejected = rt.launch(
            bad_artifact, {"buffers": {"a": a, "b": b, "o": output}, "scalars": {"M": m, "N": n, "K": k}}
        )
    except (ValueError, RuntimeError) as error:
        refusal = str(error)
    else:
        assert not rejected.get("ok"), rejected
        refusal = str(rejected)
    print("architecture refusal:", refusal, flush=True)
    result = {
        "architecture": rt._rocm_live_arch(),
        "status": "cross_compiled_exact_device_execution",
        "compiler_host": "Tajasaurus",
        "execution_host": subprocess.check_output(["hostname"], text=True).strip(),
        "gpu_report": subprocess.check_output(["/opt/rocm/bin/rocminfo"], text=True),
        "source_sha256": packages["source_sha256"],
        "compiler_binary_sha256": packages["binary_sha256"],
        "runtime_sha256": hashlib.sha256(Path(rt.__file__).read_bytes()).hexdigest(),
        "package_bundle_sha256": hashlib.sha256(args.packages.read_bytes()).hexdigest(),
        "runtime_source_path": str(Path(rt.__file__).resolve()),
        "architecture_refusal": refusal,
        "rows": rows,
        "timing_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
