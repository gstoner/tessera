"""E2E-REAL-6, SM120 low-precision matmuls: retired Graph-authored packaging vs the Schedule projection.

Sync ``E2E-REAL-6-sm120-lowp-matmul-2026-11-04``. NVFP4 and signed INT4
matmuls are packaged by typed consumers of a replayed ScheduledMatmulArtifact:
``nvidia_native.package_scheduled_nvfp4_matmul`` and
``package_scheduled_int4_matmul``, both reached from
``package_scheduled_matmul``. The frozen NVFP4 packager and the Graph-side
INT4 launch contract in ``tests/_support/nvidia_lowp_matmul_baseline.py`` are
the declared Decision #31(a) oracle; these tests are its differential.

Host-free half: projected descriptors equal the retired Graph-authored ones on
every launch field, the consumers never re-enter Graph admission, and a
tampered artifact or a changed semantic key (storage, scale layout, scale
format) fails before target compilation. The device half lives in
``tests/device/nvidia/test_lowp_matmul_migration.py``. ``mx_matmul`` has its
own differential in ``test_nvidia_mx_matmul_migration.py``.
"""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from tessera.compiler import bootstrap_prune_audit as audit
from tessera.compiler import nvidia_native as native
from tessera.compiler import scheduled_matmul as schedule
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type
from tests._support import nvidia_lowp_matmul_baseline as baseline

PIPELINE = "tessera-nvidia-pipeline-sm120"
pytestmark = pytest.mark.skipif(schedule.find_tessera_opt() is None, reason="requires production tessera-opt")

LAUNCH = ("abi_id", "entry_symbol", "buffers", "scalars", "shape_guards", "geometry", "workspace", "ordering")
NVFP4_SHAPES = ((16, 8, 64), (7, 5, 31), (33, 19, 129), (1, 1, 1))
INT4_SHAPES = ((8, 32, 32), (11, 19, 37), (16, 8, 64))


def _recording_compile(calls: list[str]):
    def compile_(tile_ir: str, entry: str):
        calls.append(entry)
        return tile_ir, "// PTX", {}, "compiler", "toolchain", (), "cold"
    return compile_


@pytest.fixture
def compiles(monkeypatch) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(native, "_compile_tile_ir", _recording_compile(calls))
    return calls


def nvfp4_module(m: int, n: int, k: int, *, scaled: bool = False, names=("a", "b", "scale_a", "scale_b"),
                 permuted: bool = False) -> GraphIRModule:
    sk = (k + 15) // 16
    types = [tensor_ir_type((m, k), "nvfp4"), tensor_ir_type((k, n), "nvfp4"),
             tensor_ir_type((m, sk), "uint8"), tensor_ir_type((sk, n), "uint8")]
    out = tensor_ir_type((m, n), "fp32")
    args = [IRArg(name, ty) for name, ty in zip(names, types)]
    kwargs = ({"scale_layout": dict(schedule.NVFP4_SCALE_LAYOUT),
               "numeric_policy": dict(schedule.NVFP4_NUMERIC_POLICY),
               "physical_contract": schedule.NVFP4_PHYSICAL_CONTRACT} if scaled else {})
    op = IROp(result="c", op_name="tessera.scaled_matmul" if scaled else "tessera.matmul",
              operands=["%" + name for name in names], operand_types=list(map(str, types)),
              result_type=str(out), kwargs=kwargs)
    if permuted:
        args.reverse()
    return GraphIRModule(functions=[GraphIRFunction(
        name="lowp_nvfp4", args=args, result_types=[out], body=[op], return_values=["%c"])])


def int4_module(m: int, n: int, k: int, *, names=("a", "b")) -> GraphIRModule:
    types = [tensor_ir_type((m, k), "int4"), tensor_ir_type((k, n), "int4")]
    out = tensor_ir_type((m, n), "int32")
    return GraphIRModule(functions=[GraphIRFunction(
        name="lowp_int4", args=[IRArg(name, ty) for name, ty in zip(names, types)], result_types=[out],
        body=[IROp(result="c", op_name="tessera.matmul", operands=["%" + n_ for n_ in names],
                   operand_types=list(map(str, types)), result_type=str(out))],
        return_values=["%c"])])


def batched_cases():
    from tests.unit.test_native_nested_nvfp4_vmap import case

    for mode in ("shared_rhs_rows", "independent_rhs", "shared_lhs"):
        for ta, tb in ((False, False), (True, False), (False, True), (True, True)):
            yield mode, ta, tb, case


def _same_launch(old, new) -> None:
    for field in LAUNCH:
        assert getattr(old.descriptor, field) == getattr(new.descriptor, field), field
    assert old.descriptor.provenance == new.descriptor.provenance
    assert old.tile_ir == new.tile_ir


@pytest.mark.parametrize("shape", NVFP4_SHAPES)
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
def test_nvfp4_projection_reproduces_retired_graph_authored_packager(compiles, shape, scaled, permuted):
    module = nvfp4_module(*shape, scaled=scaled, permuted=permuted)
    before = copy.deepcopy(module)
    assert native.native_package_kind(module) == "nvfp4_matmul"
    old = baseline.baseline_nvfp4_matmul(module, pipeline_name=PIPELINE)
    new = native.package_native(module, pipeline_name=PIPELINE)
    assert module == before
    _same_launch(old, new)
    assert compiles == ["tessera_tile_matmul_nvfp4"] * 2


@pytest.mark.parametrize("mode,ta,tb,case", list(batched_cases()))
def test_batched_nvfp4_projection_reproduces_retired_graph_authored_packager(compiles, mode, ta, tb, case):
    _, owner, values, _ = case(mode, ta, tb, (2, 3))
    graph = owner._trace_frontend_capture(values, {})[0]
    old = baseline.baseline_nvfp4_matmul(graph, pipeline_name=PIPELINE)
    new = native.package_nvfp4_matmul(graph, pipeline_name=PIPELINE)
    _same_launch(old, new)
    assert new.descriptor.provenance["batching"] == mode
    assert new.descriptor.provenance["logical_batch_shape"] == [2, 3]


def test_scheduled_matmul_dispatches_nvfp4_and_never_reenters_graph_admission(compiles, monkeypatch):
    artifact = schedule.lower_scheduled_matmul(
        schedule.canonical_nvfp4_graph(nvfp4_module(7, 5, 31)), target="nvidia_sm120")
    for name in ("supports_nvfp4_matmul", "requests_nvfp4_matmul", "_shape", "_static_shape"):
        monkeypatch.setattr(native, name, lambda *a, **k: pytest.fail("consumer re-entered Graph admission"))
    package = native.package_scheduled_matmul(artifact, pipeline_name=PIPELINE)
    assert package.descriptor.abi_id == native.SM120_NVFP4_ABI
    assert [b.name for b in package.descriptor.buffers] == ["a", "b", "scale_a", "scale_b", "c"]
    assert compiles == ["tessera_tile_matmul_nvfp4"]


def _nvfp4_artifact():
    return schedule.lower_scheduled_matmul(
        schedule.canonical_nvfp4_graph(nvfp4_module(16, 8, 64)), target="nvidia_sm120")


@pytest.mark.parametrize("edit", [
    # Relabelled artifact fields.
    lambda a: replace(a, a_name="b", b_name="a"),
    lambda a: replace(a, output_name="d"),
    lambda a: replace(a, m=32),
    lambda a: replace(a, k=32),
    lambda a: replace(a, storage="int4"),
    lambda a: replace(a, function_name="tessera_tile_matmul_nvfp4_batched"),
    # Semantic keys in the Schedule (Decision #21a): storage, block, format.
    lambda a: replace(a, schedule_ir=a.schedule_ir.replace("scale_k = 16 : i64", "scale_k = 32 : i64")),
    lambda a: replace(a, schedule_ir=a.schedule_ir.replace('scale_format = "ue4m3"', 'scale_format = "ue8m0"')),
    lambda a: replace(a, schedule_ir=a.schedule_ir.replace('format = "ue4m3", granularity', 'format = "ue8m0", granularity')),
    lambda a: replace(a, schedule_ir=a.schedule_ir.replace('storage = "nvfp4", storage_b', 'storage = "fp4_e2m1", storage_b')),
    lambda a: replace(a, schedule_ir=a.schedule_ir.replace('execution_mode = "exact_per_block"', 'execution_mode = "approximate"')),
    # Tile and retained Graph text.
    lambda a: replace(a, tile_ir=a.tile_ir.replace("tessera.scale_vector_size = 16", "tessera.scale_vector_size = 32")),
    lambda a: replace(a, tile_ir=a.tile_ir.replace("elements_per_container = 2", "elements_per_container = 1")),
    # Renaming one alias occurrence only (a consistent rename is a legitimate relabel).
    lambda a: replace(a, graph_ir=a.graph_ir.replace("%scale_a: ", "%scale_x: ", 1)),
    lambda a: replace(a, graph_ir=a.graph_ir.replace('format = "ue4m3"', 'format = "ue8m0"')),
])
def test_tampered_nvfp4_artifact_fails_before_compilation(compiles, edit):
    artifact = edit(_nvfp4_artifact())
    with pytest.raises((ValueError, RuntimeError)):
        native.package_scheduled_nvfp4_matmul(artifact, pipeline_name=PIPELINE)
    assert compiles == []


@pytest.mark.parametrize("key,value", [
    ("scale_layout", {"granularity": "block", "block": [1, 32], "format": "ue8m0"}),
    ("numeric_policy", {"accum": "fp16", "execution_mode": "exact_per_block"}),
    ("physical_contract", "nvidia_sm120_mx_blockscale_v1"),
])
def test_nvfp4_semantic_keys_fail_closed_on_conflict(compiles, key, value):
    for scaled in (False, True):
        module = nvfp4_module(16, 8, 64, scaled=scaled)
        module.functions[0].body[0].kwargs[key] = value
        with pytest.raises(ValueError, match=key):
            native.package_nvfp4_matmul(module, pipeline_name=PIPELINE)
        assert not schedule.supports_scheduled_nvfp4_matmul(module)
    assert compiles == []


@pytest.mark.parametrize("shape", INT4_SHAPES)
def test_int4_projection_reproduces_graph_authored_launch(compiles, shape):
    module = int4_module(*shape, names=("lhs", "rhs"))
    assert native.native_package_kind(module) == "int4_matmul"
    assert schedule.supports_scheduled_int4_matmul(module)
    old = baseline.graph_authored_int4_launch(module)
    new = native.package_native(module, pipeline_name=PIPELINE).descriptor
    assert new.buffers == old["buffers"]
    assert frozenset(new.shape_guards) == old["guards"]
    assert tuple(s.name for s in new.scalars) == old["scalars"]
    assert new.provenance["shape"] == old["shape"]
    assert compiles == ["tessera_tile_matmul_int4"]


def _int4_artifact():
    return schedule.lower_scheduled_matmul(int4_module(11, 19, 37), target="nvidia_sm120")


@pytest.mark.parametrize("edit", [
    lambda a: replace(a, a_name="b", b_name="a"),
    lambda a: replace(a, n=20),
    lambda a: replace(a, k=38),
    lambda a: replace(a, accum="f32"),
    lambda a: replace(a, schedule_ir=a.schedule_ir.replace('storage = "int4"', 'storage = "int8"')),
    lambda a: replace(a, tile_ir=a.tile_ir.replace("tessera.packed_shapes = array<i64: 11, 19, 19, 19>",
                                                   "tessera.packed_shapes = array<i64: 11, 37, 37, 19>")),
    lambda a: replace(a, tile_ir=a.tile_ir.replace('signedness = "signed_twos_complement"', 'signedness = "unsigned"')),
])
def test_tampered_int4_artifact_fails_before_compilation(compiles, edit):
    artifact = edit(_int4_artifact())
    with pytest.raises((ValueError, RuntimeError)):
        native.package_scheduled_int4_matmul(artifact, pipeline_name=PIPELINE)
    assert compiles == []


def test_int4_tile_carries_the_packed_shapes_the_consumer_reads():
    assert "tessera.packed_shapes = array<i64: 11, 19, 19, 19>" in _int4_artifact().tile_ir


def test_prune_rows_name_the_projecting_routes():
    rows = {(t, f): (route, status) for t, f, route, status in audit.family_rows()}
    assert rows[("nvidia_sm120", "nvfp4_matmul")] == (
        "scheduled_matmul.supports_scheduled_nvfp4_matmul", "compiled")
    assert rows[("nvidia_sm120", "int4_matmul")] == (
        "scheduled_matmul.supports_scheduled_int4_matmul", "compiled")
    assert rows[("nvidia_sm120", "mx_matmul")] == (
        "scheduled_matmul.supports_scheduled_mx_matmul", "compiled")


def test_admission_predicates_are_sm120_only():
    for target in ("rocm_gfx1151", "rocm_gfx1201", "x86", "apple_gpu"):
        assert not schedule.supports_scheduled_nvfp4_matmul(nvfp4_module(16, 8, 64), target=target)
        assert not schedule.supports_scheduled_int4_matmul(int4_module(16, 8, 64), target=target)
