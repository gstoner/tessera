"""E2E-REAL-6, SM120 OCP MX matmul: retired Graph-authored packaging vs the Schedule projection.

Sync ``E2E-REAL-6-sm120-mx-matmul-2026-11-04``. FP6 (E2M3/E3M2) and MXFP4
(E2M1) matmuls with UE8M0 K32 block scales now lower Graph -> Schedule -> Tile
through the C++ ``nvidia_sm120_mx_blockscale_v1`` contract, and
``nvidia_native.package_scheduled_mx_matmul`` reads every launch field from
the replayed ``ScheduledMatmulArtifact`` (``verify_mx_projection``). The
retired packager, which authored its Tile IR in Python and read the Graph, is
``tests/_support/nvidia_lowp_matmul_baseline.baseline_mx_matmul`` -- the
declared Decision #31(a) oracle; these tests are its host-free differential.
The device half is ``tests/device/nvidia/test_mx_matmul_migration.py``.
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
STORAGES = ("fp6_e2m3", "fp6_e3m2", "fp4_e2m1")
SHAPES = ((16, 8, 64), (19, 13, 69), (7, 5, 33), (1, 1, 1))


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


def mx_module(m: int, n: int, k: int, dtype: str, *, names=("a", "b", "scale_a", "scale_b"),
              permuted: bool = False) -> GraphIRModule:
    sk = (k + 31) // 32
    types = [tensor_ir_type((m, k), dtype), tensor_ir_type((k, n), dtype),
             tensor_ir_type((m, sk), "uint8"), tensor_ir_type((sk, n), "uint8")]
    out = tensor_ir_type((m, n), "fp32")
    args = [IRArg(name, ty) for name, ty in zip(names, types)]
    op = IROp(result="c", op_name="tessera.matmul", operands=["%" + name for name in names],
              operand_types=list(map(str, types)), result_type=str(out))
    if permuted:
        args.reverse()
    return GraphIRModule(functions=[GraphIRFunction(
        name="lowp_mx", args=args, result_types=[out], body=[op], return_values=["%c"])])


def _entry(dtype: str) -> str:
    return "tessera_tile_matmul_mx_" + schedule.MX_STORAGES[dtype][0]


def test_fp6_graph_types_are_mlir_builtins():
    text = schedule.canonical_mx_graph(mx_module(16, 8, 64, "fp6_e3m2")).to_mlir(
        target="nvidia_sm120", canonical=True)
    assert "f6E3M2FN" in text and "!tessera.fp6" not in text
    assert 'physical_contract = "nvidia_sm120_mx_blockscale_v1"' in text


@pytest.mark.parametrize("dtype", STORAGES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("permuted", [False, True])
def test_mx_projection_reproduces_retired_graph_authored_packager(compiles, dtype, shape, permuted):
    module = mx_module(*shape, dtype, permuted=permuted)
    before = copy.deepcopy(module)
    assert native.native_package_kind(module) == "mx_matmul"
    assert schedule.supports_scheduled_mx_matmul(module)
    old = baseline.baseline_mx_matmul(module, pipeline_name=PIPELINE)
    new = native.package_native(module, pipeline_name=PIPELINE)
    assert module == before
    for field in LAUNCH:
        assert getattr(old.descriptor, field) == getattr(new.descriptor, field), field
    # The Tile text is now the C++ Schedule->Tile product, so its digest
    # differs; every other provenance field is unchanged, and the new package
    # adds the schedule identity it was replayed from.
    old_provenance = dict(old.descriptor.provenance)
    new_provenance = dict(new.descriptor.provenance)
    assert new_provenance.pop("schedule_digest")
    old_provenance.pop("tile_ir_digest"), new_provenance.pop("tile_ir_digest")
    assert old_provenance == new_provenance
    assert "tessera.schedule_hash" in new.tile_ir and "tessera.schedule_hash" not in old.tile_ir
    assert compiles == [_entry(dtype)] * 2


def test_scheduled_matmul_dispatches_mx_and_never_reenters_graph_admission(compiles, monkeypatch):
    artifact = schedule.lower_scheduled_matmul(
        schedule.canonical_mx_graph(mx_module(19, 13, 69, "fp6_e2m3")), target="nvidia_sm120")
    for name in ("supports_mx_matmul", "requests_mx_matmul", "_scale_names", "_matrix_names",
                 "_shape", "_static_shape"):
        monkeypatch.setattr(native, name, lambda *a, **k: pytest.fail("consumer re-entered Graph admission"))
    package = native.package_scheduled_matmul(artifact, pipeline_name=PIPELINE)
    assert package.descriptor.abi_id == native.SM120_FP6_E2M3_ABI
    assert [b.name for b in package.descriptor.buffers] == ["a", "b", "scale_a", "scale_b", "c"]
    assert compiles == ["tessera_tile_matmul_mx_e2m3"]


def _mx_artifact(dtype: str = "fp4_e2m1"):
    return schedule.lower_scheduled_matmul(
        schedule.canonical_mx_graph(mx_module(19, 13, 69, dtype)), target="nvidia_sm120")


@pytest.mark.parametrize("edit", [
    # Relabelled artifact fields.
    lambda a: replace(a, a_name="b", b_name="a"),
    lambda a: replace(a, output_name="d"),
    lambda a: replace(a, m=32),
    lambda a: replace(a, k=64),
    lambda a: replace(a, storage="e2m3"),
    lambda a: replace(a, a_dtype="fp6_e2m3", b_dtype="fp6_e2m3"),
    lambda a: replace(a, function_name="tessera_tile_matmul_mx_e2m3"),
    # Semantic keys in the Schedule (Decision #21a): storage, block, format, policy.
    lambda a: replace(a, schedule_ir=a.schedule_ir.replace("scale_k = 32 : i64", "scale_k = 16 : i64")),
    lambda a: replace(a, schedule_ir=a.schedule_ir.replace('scale_format = "ue8m0"', 'scale_format = "ue4m3"')),
    lambda a: replace(a, schedule_ir=a.schedule_ir.replace('format = "ue8m0", granularity', 'format = "ue4m3", granularity')),
    lambda a: replace(a, schedule_ir=a.schedule_ir.replace('storage = "fp4_e2m1", storage_b', 'storage = "nvfp4", storage_b')),
    lambda a: replace(a, schedule_ir=a.schedule_ir.replace('execution_mode = "exact_per_block"', 'execution_mode = "approximate"')),
    # Tile and retained Graph text.
    lambda a: replace(a, tile_ir=a.tile_ir.replace("tessera.scale_vector_size = 32", "tessera.scale_vector_size = 16")),
    lambda a: replace(a, tile_ir=a.tile_ir.replace('tessera.scale_format = "ue8m0"', 'tessera.scale_format = "ue4m3"')),
    lambda a: replace(a, tile_ir=a.tile_ir.replace("elements_per_container = 2", "elements_per_container = 1")),
    lambda a: replace(a, tile_ir=a.tile_ir.replace('physical_contract = "nvidia_sm120_mx_blockscale_v1"', "")),
    # Renaming one alias occurrence only (a consistent rename is a legitimate relabel).
    lambda a: replace(a, graph_ir=a.graph_ir.replace("%scale_a: ", "%scale_x: ", 1)),
    lambda a: replace(a, graph_ir=a.graph_ir.replace('block = [1, 32]', 'block = [1, 16]')),
    lambda a: replace(a, graph_ir=a.graph_ir.replace("f4E2M1FN", "f6E2M3FN")),
])
def test_tampered_mx_artifact_fails_before_compilation(compiles, edit):
    artifact = edit(_mx_artifact())
    with pytest.raises((ValueError, RuntimeError)):
        native.package_scheduled_mx_matmul(artifact, pipeline_name=PIPELINE)
    assert compiles == []


@pytest.mark.parametrize("key,value", [
    ("scale_layout", {"granularity": "block", "block": [1, 16], "format": "ue4m3"}),
    ("numeric_policy", {"accum": "fp16", "execution_mode": "exact_per_block"}),
    ("physical_contract", "nvidia_sm120_nvfp4_blockscale_v1"),
])
def test_mx_semantic_keys_fail_closed_on_conflict(compiles, key, value):
    module = mx_module(16, 8, 64, "fp6_e2m3")
    module.functions[0].body[0].kwargs[key] = value
    with pytest.raises(ValueError, match=key):
        native.package_mx_matmul(module, pipeline_name=PIPELINE)
    assert not schedule.supports_scheduled_mx_matmul(module)
    assert compiles == []


def test_mx_schedule_refuses_a_scale_view_off_the_k32_grid(compiles):
    module = mx_module(16, 8, 64, "fp4_e2m1")
    fn = module.functions[0]
    fn.args[2].ir_type = tensor_ir_type((16, 3), "uint8")
    fn.body[0].operand_types[2] = str(fn.args[2].ir_type)
    assert not schedule.supports_scheduled_mx_matmul(module)
    with pytest.raises(ValueError):
        native.package_mx_matmul(module, pipeline_name=PIPELINE)
    assert compiles == []


def test_admission_predicate_is_sm120_only():
    for target in ("rocm_gfx1151", "rocm_gfx1201", "x86", "apple_gpu"):
        assert not schedule.supports_scheduled_mx_matmul(mx_module(16, 8, 64, "fp6_e2m3"), target=target)


def test_prune_row_names_the_projecting_route():
    rows = {(t, f): (route, status) for t, f, route, status in audit.family_rows()}
    assert rows[("nvidia_sm120", "mx_matmul")] == (
        "scheduled_matmul.supports_scheduled_mx_matmul", "compiled")


def test_production_has_no_python_mx_tile_author():
    assert not hasattr(native, "emit_mx_matmul_tile_ir")
