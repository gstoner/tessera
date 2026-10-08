"""Measure cross-shape image reuse; package wall time is not GPU kernel time."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
from tessera import runtime as rt
from tessera.compiler import rocm_native, scheduled_attention
from tessera.compiler.attention_contract import reference_streaming_attention
from tests.unit.test_scheduled_attention_consumers import _module

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = (
    "python/tessera/compiler/rocm_native.py",
    "python/tessera/compiler/scheduled_attention.py",
    "python/tessera/runtime.py",
    "tests/unit/test_scheduled_attention_consumers.py",
    "benchmarks/rocm/measure_scheduled_attention_cache.py",
)


def record():
    arch = rt._rocm_live_arch()
    if arch not in {"gfx1151", "gfx1201"}:
        raise RuntimeError(f"requires exact gfx1151 or gfx1201, got {arch}")
    rocm_native._cache.clear()
    rocm_native._shape_free_targets.clear()
    original = rocm_native._run_opt
    binary_calls = 0
    def counted(tool, text, pipeline):
        nonlocal binary_calls
        binary_calls += int("output=binary" in pipeline)
        return original(tool, text, pipeline)
    rows = []
    rocm_native._run_opt = counted
    try:
        for index, sq in enumerate((17, 23, 31, 65)):
            module = _module(target="rocm", query_rows=sq)
            module.functions[0].name = f"attention_{sq}"
            start = time.perf_counter()
            artifact = scheduled_attention.lower_scheduled_attention(module, target=f"rocm_{arch}")
            graph_to_tile_ms = (time.perf_counter() - start) * 1e3
            before = binary_calls
            start = time.perf_counter()
            package = rocm_native.package_scheduled_attention(artifact, pipeline_name="tessera-lower-to-rocm")
            first_ms = (time.perf_counter() - start) * 1e3
            compiled = binary_calls - before
            samples = []
            for _ in range(7):
                start = time.perf_counter()
                rocm_native.package_scheduled_attention(artifact, pipeline_name="tessera-lower-to-rocm")
                samples.append((time.perf_counter() - start) * 1e3)
            rng = np.random.default_rng(120 + index)
            q,k,v = [(rng.normal(size=shape)*.2).astype(np.float16)
                     for shape in ((1,4,sq,64),(1,2,19,64),(1,2,19,64))]
            out = np.full((1,4,sq,64),np.nan,np.float32)
            native = rt.RuntimeArtifact(metadata={"target":f"rocm_{arch}"},native_image=package.image,
                launch_descriptor=package.descriptor,tile_ir=package.tile_ir,target_ir=package.target_ir)
            result = rt.launch(native,dict(q=q,k=k,v=v,o=out,Sq=sq,Sk=19,Scale=.125,Causal=1,Hq=4,KvRatio=2,Window=64))
            assert result["ok"] and result["execution_kind"] == "native_gpu", result
            expected = reference_streaming_attention(q,k,v,block_size=16,scale=.125,
                causal=True,window_left=64,window_right=0)
            np.testing.assert_allclose(out,expected,rtol=.03,atol=.03)
            rows.append(dict(query_rows=sq,graph_to_tile_ms=graph_to_tile_ms,first_package_ms=first_ms,
                repeat_package_ms=samples,repeat_median_ms=statistics.median(samples),
                binary_compiles=compiled,compile_state=package.image.compile_state,
                image_digest=package.image.image_digest,schedule_digest=artifact.schedule_digest,
                correct=True,max_abs_error=float(np.max(np.abs(out-expected)))))
    finally:
        rocm_native._run_opt = original
    assert binary_calls == 1, binary_calls
    assert len({r["image_digest"] for r in rows}) == 1
    source_hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in SOURCE_PATHS
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return dict(
        schema="tessera.scheduled_attention_cache.v1",
        host=platform.node(),
        architecture=arch,
        claim="host_compile_cost",
        binary_compiles=binary_calls,
        source_revision=revision,
        source_worktree_dirty=dirty,
        relevant_source_sha256=source_hashes,
        rows=rows,
    )

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(record(),indent=2)+"\n")
