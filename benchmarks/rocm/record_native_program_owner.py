"""Diagnostic attribution of existing public and retained-owner call costs."""

import argparse
import cProfile
import dataclasses
import hashlib
import json
import os
import pstats
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
import numpy as np
import tessera as ts
from tessera import runtime as rt
from benchmarks.rocm.benchmark_public_quantized_mlp import make_mlp, inputs
from benchmarks.rocm.quantized_mlp_reference import assert_compiled_chain

parser = argparse.ArgumentParser()
parser.add_argument("--expected-revision", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--shape", type=lambda s: tuple(map(int, s.split("x"))), required=True)
args = parser.parse_args()
assert len(args.shape) == 4 and min(args.shape) > 0
assert rt._rocm_live_arch() == "gfx1201"
source = lambda: subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
assert source() == args.expected_revision


def source_snapshot():
    paths = subprocess.check_output(
        ["git", "ls-files", "--modified", "--others", "--exclude-standard"], text=True
    ).splitlines()
    return {p: sha(Path(p)) for p in sorted(set(paths)) if Path(p).is_file() and not p.startswith("graphify-out/")}


provider = rt._load_rocm_native_movement_runtime()
assert provider is not None
info = subprocess.check_output(["/opt/rocm/bin/rocminfo"], text=True)
device = re.search(r"Name:\s+gfx1201\s+Uuid:\s+(GPU-\S+)\s+Marketing Name:\s+([^\n]+)", info)
assert device
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
snapshot = source_snapshot()
identity = dict(
    candidate_files=snapshot,
    recorder_sha256=sha(__file__),
    source_revision=source(),
    gpu_uuid=device[1],
    gpu_name=device[2].strip(),
    compiler_sha256={k: sha(os.environ[k]) for k in ("TESSERA_OPT", "TESSERA_ROCM_OPT")},
    provider_sha256=sha(provider._name),
    provider_path=provider._name,
)

import ctypes as C

stats = provider.tessera_rocm_program_cache_stats
stats.argtypes = [C.POINTER(C.c_uint64)] * 4


def cache_stats():
    counts = [C.c_uint64() for _ in range(4)]
    assert stats(*[C.byref(x) for x in counts]) == 0
    return dict(zip(("hits", "misses", "idle_entries", "idle_bytes"), (x.value for x in counts)))


cache_before = cache_stats()
roots = inputs(*args.shape)
fn = ts.jit(target="rocm_gfx1201", native_required=True)(make_mlp(*args.shape))
first = fn(*roots)
assert fn.execution_kind == "native_gpu"
package = fn._rocm_quantized_product_last_program.native
certificate = assert_compiled_chain(package, roots, first, provider._name)
profiler = cProfile.Profile()
profiled = profiler.runcall(fn, *roots)
np.testing.assert_array_equal(profiled, first)
profile_rows = []
for (file, line, function), (primitive, total, self_s, cumulative_s, _) in pstats.Stats(profiler).stats.items():
    profile_rows.append(
        dict(
            file=file,
            line=line,
            function=function,
            primitive_calls=primitive,
            calls=total,
            self_s=self_s,
            cumulative_s=cumulative_s,
        )
    )
profile_rows.sort(key=lambda x: x["cumulative_s"], reverse=True)
public = []
updates = []
invokes_reads = []
updated_calls = []
events = []
start = time.perf_counter()
with package.prepare(roots, runtime_library=provider._name) as owner:
    prepare_ms = (time.perf_counter() - start) * 1000
    for window in range(7):
        if window % 2 == 0:
            start = time.perf_counter()
            actual = fn(*roots)
            public.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        owner.update(roots)
        updated = time.perf_counter()
        generation, _ = owner.invoke()
        retained = owner.read(generation)[0]
        ended = time.perf_counter()
        updates.append((updated - start) * 1000)
        invokes_reads.append((ended - updated) * 1000)
        updated_calls.append((ended - start) * 1000)
        if window % 2:
            start = time.perf_counter()
            actual = fn(*roots)
            public.append((time.perf_counter() - start) * 1000)
        np.testing.assert_array_equal(actual, first)
        np.testing.assert_array_equal(retained, first)
    for _ in range(3):
        generation, event = owner.invoke(repeats=8, timed=True)
        events.append(event)
        np.testing.assert_array_equal(owner.read(generation)[0], first)
seen = set()


def footprint(value):
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    total = sys.getsizeof(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        total += sum(footprint(getattr(value, f.name)) for f in dataclasses.fields(value))
    elif isinstance(value, Mapping):
        total += sum(footprint(k) + footprint(v) for k, v in value.items())
    elif isinstance(value, (tuple, list, set, frozenset)):
        total += sum(footprint(v) for v in value)
    return total


host_metadata = dict(
    approximate_unique_python_package_bytes=footprint(package),
    program_json_bytes=len(package.program_json.encode()),
    source_graph_bytes=len(package.graph_ir.encode()),
    validation_cache=getattr(package.validate, "cache_info", lambda: None)(),
    abi_cache=getattr(owner._bind_abi, "cache_info", lambda: None)(),
    scope="Python reachable package footprint;excludes native GPU allocations",
)
record = dict(
    schema="tessera.rocm.public_owner_attribution.v1",
    shape_mhkn=list(args.shape),
    identity=identity,
    correctness="independent_stage_and_propagation_before_timing_and_bitwise_after_each_arm",
    numerical_certificate=certificate,
    host_metadata=host_metadata,
    cache_before=cache_before,
    cache_after=cache_stats(),
    prepare_ms=prepare_ms,
    public_completed_ms=public,
    update_ms=updates,
    invoke_read_ms=invokes_reads,
    updated_retained_call_ms=updated_calls,
    native_program_event_ms=events,
    profile=profile_rows[:100],
    member_image_digests=[p.image.image_digest for p in package.native_stages],
    scope="diagnostic public ownership versus explicit existing prepared update/invoke/read API;synthetic roots;not a shipping optimization ratio or model quality acceptance",
)
assert source() == args.expected_revision
assert source_snapshot() == snapshot
assert sha(__file__) == identity["recorder_sha256"]
assert sha(provider._name) == identity["provider_sha256"]
assert all(sha(os.environ[k]) == v for k, v in identity["compiler_sha256"].items())
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2) + "\n")
profiler.dump_stats(str(args.output) + ".prof")
fn.close_native_storage()
print(
    json.dumps(
        dict(
            shape=record["shape_mhkn"],
            prepare_ms=prepare_ms,
            public_median_ms=statistics.median(public),
            updated_retained_median_ms=statistics.median(updated_calls),
            update_median_ms=statistics.median(updates),
            invoke_read_median_ms=statistics.median(invokes_reads),
            native_event_median_ms=statistics.median(events),
            output=str(args.output),
        )
    )
)
