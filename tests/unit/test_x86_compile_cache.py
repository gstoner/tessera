"""The x86 package compile cache hits on an exact repeat and misses on any change.

``x86_compile_cache`` memoizes each ``tessera-opt`` run on (compiler binary
digest, pass option, complete source text), the ``--version`` probe on the
binary digest, and a shared-object payload on its stat signature. These tests
drive real package calls (``tessera-opt`` required; the AVX-512 shared object
is replaced by a temporary file so any host can run them) and count the
subprocesses the cache lets through.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tessera.compiler import x86_breadth, x86_compile_cache, x86_native
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt

pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason="native compiler required")
PIPELINE = "tessera-lower-to-x86"


def _module(op_name="tessera.sub", shape=(3, 17), kwargs=None) -> GraphIRModule:
    dims = "x".join(map(str, shape))
    ty = IRType(f"tensor<{dims}xf32>", tuple(map(str, shape)), "fp32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="cache_probe", args=[IRArg("a", ty), IRArg("b", ty)], result_types=[ty],
        body=[IROp(result="o", op_name=op_name, operands=["%a", "%b"], operand_types=[str(ty)] * 2,
                   result_type=str(ty), kwargs=dict(kwargs or {}))],
        return_values=["%o"],
    )])


@pytest.fixture
def counted(monkeypatch, tmp_path):
    """A fresh cache, a stand-in shared object, and a subprocess counter."""
    library = tmp_path / "libtessera_x86_elementwise.so"
    library.write_bytes(b"\x7fELF-stand-in-v1")
    monkeypatch.setattr(x86_native, "_library_path", lambda architecture=None: library)
    x86_compile_cache.clear()
    calls: list[list[str]] = []
    real = subprocess.run

    def counting(args, *a, **k):
        calls.append([str(x) for x in args])
        return real(args, *a, **k)

    monkeypatch.setattr(x86_compile_cache.subprocess, "run", counting)
    yield calls, library
    x86_compile_cache.clear()


def test_exact_repeat_reruns_no_subprocess_and_rebuilds_an_equal_package(counted):
    calls, _ = counted
    first = x86_native.package_elementwise(_module(), pipeline_name=PIPELINE)
    cold = len(calls)
    # Four compiler actions: Graph->Schedule, Schedule->Tile, Tile->Target,
    # and --version. A cold ELF identity may also inspect loaded dependencies;
    # those are accounted separately and must disappear on an exact repeat.
    tool=str(find_tessera_opt())
    assert sum(call[0]==tool for call in calls)==4
    assert all(call[0]==tool or Path(call[0]).name in {"ldd","readelf"} for call in calls)
    second = x86_native.package_elementwise(_module(), pipeline_name=PIPELINE)
    assert len(calls) == cold
    assert first.descriptor == second.descriptor and first.image == second.image
    assert first is not second
    assert x86_compile_cache.stats()["hits"] >= 4


@pytest.mark.parametrize("variant", [
    _module("tessera.add"),                      # op identity
    _module(shape=(3, 18)),                      # shape
    _module("tessera.maximum"),                  # kind within one ABI family
])
def test_a_changed_graph_input_misses(counted, variant):
    calls, _ = counted
    base = x86_native.package_elementwise(_module(), pipeline_name=PIPELINE)
    cold = len(calls)
    changed = x86_native.package_elementwise(variant, pipeline_name=PIPELINE)
    assert len(calls) > cold
    assert changed.descriptor != base.descriptor


def test_a_changed_attribute_misses(counted):
    calls, _ = counted

    def loss(delta):
        ty = IRType("tensor<8xf32>", ("8",), "fp32")
        return GraphIRModule(functions=[GraphIRFunction(
            name="loss", args=[IRArg("p", ty), IRArg("t", ty)], result_types=[ty],
            body=[IROp(result="o", op_name="tessera.huber_loss", operands=["%p", "%t"],
                       operand_types=[str(ty)] * 2, result_type=str(ty),
                       kwargs={"reduction": "none", "delta": delta})],
            return_values=["%o"],
        )])

    half = x86_breadth.package_graph_breadth(loss(0.5), pipeline_name=PIPELINE)
    cold = len(calls)
    two = x86_breadth.package_graph_breadth(loss(2.0), pipeline_name=PIPELINE)
    assert len(calls) > cold
    assert half.descriptor.provenance["parameter"] == 0.5
    assert two.descriptor.provenance["parameter"] == 2.0


def test_a_changed_compiler_misses(counted, monkeypatch):
    calls, _ = counted
    x86_native.package_elementwise(_module(), pipeline_name=PIPELINE)
    cold = len(calls)
    tool=str(find_tessera_opt())
    compiler_calls=sum(call[0]==tool for call in calls)
    monkeypatch.setattr(x86_compile_cache, "_tool_digest", lambda tool: "0" * 64)
    x86_native.package_elementwise(_module(), pipeline_name=PIPELINE)
    assert sum(call[0]==tool for call in calls[cold:])>=compiler_calls
    # Every compile action and the version probe rerun; shared dependency
    # discovery is not itself a compiler action and may already be memoized.


def test_a_rebuilt_shared_object_misses(counted):
    calls, library = counted
    first = x86_native.package_elementwise(_module(), pipeline_name=PIPELINE)
    library.write_bytes(b"\x7fELF-stand-in-v2-rebuilt")
    second = x86_native.package_elementwise(_module(), pipeline_name=PIPELINE)
    assert second.image.payload == b"\x7fELF-stand-in-v2-rebuilt"
    assert first.image.image_digest != second.image.image_digest
    assert first.image.toolchain_fingerprint != second.image.toolchain_fingerprint


def test_a_failing_run_is_not_cached(counted):
    calls, _ = counted
    tool = find_tessera_opt()
    with pytest.raises(RuntimeError):
        x86_compile_cache.run(tool, "module { this is not mlir }", "--tessera-graph-to-schedule")
    first = len(calls)
    with pytest.raises(RuntimeError):
        x86_compile_cache.run(tool, "module { this is not mlir }", "--tessera-graph-to-schedule")
    assert len(calls) == first + 1


def test_the_key_holds_compiler_option_and_source(counted, monkeypatch):
    tool = find_tessera_opt()
    base = x86_compile_cache.run_key(tool, "module {}", "--canonicalize")
    assert base != x86_compile_cache.run_key(tool, "module {} ", "--canonicalize")
    assert base != x86_compile_cache.run_key(tool, "module {}", "--cse")
    monkeypatch.setattr(x86_compile_cache, "_tool_digest", lambda tool: "f" * 64)
    assert base != x86_compile_cache.run_key(tool, "module {}", "--canonicalize")


def test_a_forged_artifact_still_fails_replay_with_a_warm_cache(counted):
    from dataclasses import replace

    from tessera.compiler import scheduled_kernel

    artifact = scheduled_kernel.lower_scheduled_kernel(_module(), target="x86")
    x86_native.package_scheduled_kernel(artifact, pipeline_name=PIPELINE)  # warm every boundary
    forged = replace(artifact, tile_ir=artifact.tile_ir.replace('kind = "sub"', 'kind = "add"'))
    with pytest.raises(ValueError):
        x86_native.package_scheduled_kernel(forged, pipeline_name=PIPELINE)


def test_library_identity_is_the_stat_signature(tmp_path):
    x86_compile_cache.clear()
    path = tmp_path / "lib.so"
    path.write_bytes(b"one")
    data, digest = x86_compile_cache.payload(path)
    assert data == b"one"
    path.write_bytes(b"two!")
    assert x86_compile_cache.payload(path)[0] == b"two!"
    assert x86_compile_cache.payload(Path(path))[1] != digest
