"""E2E-REAL-6, Apple CPU matmul/gemm/softmax: retired Graph-owned packaging vs
the compiled route.

``apple_cpu_native.package_native`` used to read the Python Graph object and
build the Accelerate launch descriptor from it. The static f32 ``matmul`` /
``gemm`` and rank-2 ``softmax`` envelopes now lower through the native Graph ->
Schedule -> Tile boundaries and package the replayed artifact, reaching the
Accelerate delegate through a verified ``tessera_apple.cpu.accelerate_gemm`` /
``vector_reduce`` Target IR op. The frozen constructor in
``tests/_support/apple_cpu_baseline.py`` is the declared oracle (Decision
#31(a)); these tests are its differential.

Host-free half: every descriptor/ABI field the old route produced is reproduced
over the envelope it admitted, its refusals stay refused, and a tampered
artifact fails closed before the runtime is touched. Device half (Darwin with
the real Accelerate runtime dylib, the M1 Max): both images run on the same
inputs and must agree bit-for-bit, and with a float64 oracle.
"""

from __future__ import annotations

import re
from dataclasses import replace

import numpy as np
import pytest

from tessera.compiler import apple_cpu_native, scheduled_kernel, scheduled_matmul
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests._support import apple_cpu_baseline as baseline

PIPELINE = "tessera-lower-to-apple_cpu"
_ELEMENT = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}


def _apple_dialect_registered() -> bool:
    tool = find_tessera_opt()
    if tool is None:
        return False
    probe = ('module {\n  tessera_apple.cpu.accelerate_gemm {source = "s", '
             'result = "r", ordinal = 0}\n}\n')
    try:
        run_tessera_opt(tool, probe, "--verify-each")
    except RuntimeError:
        return False
    return True


#: The Apple CPU Target IR op is verified by the production compiler, which
#: registers ``tessera_apple`` only where the Apple backend is configured.
_needs_compiler = pytest.mark.skipif(
    not _apple_dialect_registered(),
    reason="requires production tessera-opt with the tessera_apple dialect",
)


def _type(shape, dtype: str = "fp32") -> IRType:
    return IRType(f"tensor<{'x'.join(map(str, shape))}x{_ELEMENT[dtype]}>",
                  tuple(map(str, shape)), dtype)


def _module(op_name, shapes, out_shape, *, dtype="fp32", out_dtype=None, kwargs=None) -> GraphIRModule:
    args = [IRArg(f"a{i}", _type(shape, dtype)) for i, shape in enumerate(shapes)]
    out_type = _type(out_shape, out_dtype or dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name="apple_cpu_migration", args=args, result_types=[out_type],
        body=[IROp(result="out", op_name=op_name, operands=[f"%{a.name}" for a in args],
                   operand_types=[str(a.ir_type) for a in args], result_type=str(out_type),
                   kwargs=kwargs or {})],
        return_values=["%out"],
    )])


def _matmul(m, k, n, op_name="tessera.matmul", **kw):
    return _module(op_name, [(m, k), (k, n)], (m, n), **kw)


def _softmax(rows, columns, **kw):
    return _module("tessera.softmax", [(rows, columns)], (rows, columns), **kw)


_ADMITTED = [
    _matmul(1, 1, 1), _matmul(7, 9, 5), _matmul(64, 33, 17),
    _matmul(7, 9, 5, "tessera.gemm"), _matmul(16, 16, 16, "tessera.gemm"),
    _softmax(1, 1), _softmax(6, 11), _softmax(6, 11, kwargs={"axis": 1}),
    _softmax(6, 11, kwargs={"axis": -1}), _softmax(64, 129),
]
_ADMITTED_IDS = ["mm-1", "mm-7x9x5", "mm-64x33x17", "gemm-7x9x5", "gemm-16",
                 "sm-1", "sm-6x11", "sm-axis1", "sm-axis-1", "sm-64x129"]


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    library = tmp_path / "libtessera_apple_cpu_runtime.dylib"
    library.write_bytes(b"apple-cpu-migration-fixture")
    monkeypatch.setattr(apple_cpu_native, "_runtime_library_path", lambda: library)
    return library


# ---------------------------------------------------------------------------
# Differential: the compiled route reproduces the retired descriptor
# ---------------------------------------------------------------------------


@_needs_compiler
@pytest.mark.parametrize("module", _ADMITTED, ids=_ADMITTED_IDS)
def test_compiled_route_reproduces_retired_descriptor(module, fake_runtime):
    assert baseline.retired_contract(module)
    old = baseline.package(module, pipeline_name=PIPELINE)
    new = apple_cpu_native.package_native(module, pipeline_name=PIPELINE)
    for field in ("entry_symbol", "abi_id", "buffers", "shape_guards", "scalars",
                  "geometry", "ordering"):
        assert getattr(new.descriptor, field) == getattr(old.descriptor, field), field
    for key in ("route", "op_kind", "result_count"):
        assert new.descriptor.provenance[key] == old.descriptor.provenance[key], key
    old_call, new_call = old.descriptor.provenance["value_call"], new.descriptor.provenance["value_call"]
    for key in ("op", "op_kind", "symbol", "status"):
        assert new_call[key] == old_call[key], key
    assert new.image.entry_points == old.image.entry_points
    assert new.image.payload == old.image.payload
    assert (new.image.target, new.image.architecture, new.image.binary_format) == (
        old.image.target, old.image.architecture, old.image.binary_format)
    # The new route is the scheduled one, and says so.
    provenance = new.descriptor.provenance
    assert provenance["work_item"] == "E2E-REAL-6"
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["schedule_digest"])
    assert "tessera.matmul" not in new.tile_ir and "tessera.softmax" not in new.tile_ir
    if provenance["op_kind"] == "softmax":
        assert "tile.softmax_kernel" in new.tile_ir
        assert new.target_ir.startswith("tessera_apple.cpu.vector_reduce {")
        assert new_call["axis"] == -1
    else:
        assert "tile.matmul_kernel" in new.tile_ir
        # Tier-3 delegate behind the declared Target IR op, its tiling named as dropped.
        assert new.target_ir.startswith("tessera_apple.cpu.accelerate_gemm {")
        assert provenance["dropped_reason"] == "delegated_to_accelerate_cblas_sgemm"
    assert f'tessera.schedule_hash = "{provenance["schedule_digest"]}"' in new.target_ir


@_needs_compiler
@pytest.mark.parametrize("op_name,kind", [("tessera.matmul", "matmul"), ("tessera.gemm", "gemm"),
                                          ("tessera.softmax", "softmax")])
def test_driver_routes_f32_families_through_the_scheduled_package(op_name, kind, fake_runtime, monkeypatch):
    from tessera.compiler.driver import compile_graph_module

    monkeypatch.setenv("TESSERA_APPLE_CPU_RUNTIME_LIB", str(fake_runtime))
    module = _softmax(6, 11) if kind == "softmax" else _matmul(7, 9, 5, op_name)
    bundle = compile_graph_module(
        module, source_origin="apple-cpu-migration-driver", target="apple_cpu",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.abi_id == f"tessera.apple.cpu.value.{kind}.fp32.v1"
    assert bundle.launch_descriptor.provenance["work_item"] == "E2E-REAL-6"
    assert bundle.launch_descriptor.provenance["schedule_digest"]


_REFUSED = [
    _softmax(6, 11, kwargs={"axis": 0}),
    _module("tessera.softmax", [(2, 3, 4)], (2, 3, 4)),
    _softmax(6, 11, dtype="fp16"),
    _matmul(7, 9, 5, kwargs={"transposeA": True}),
    _matmul(7, 9, 5, out_dtype="fp16"),
    _module("tessera.matmul", [(2, 7, 9), (2, 9, 5)], (2, 7, 5)),
]
_REFUSED_IDS = ["softmax-axis0", "softmax-rank3", "softmax-f16", "transposeA",
                "mixed-output", "rank3-matmul"]


@_needs_compiler
@pytest.mark.parametrize("module", _REFUSED, ids=_REFUSED_IDS)
def test_scheduled_route_refuses_outside_the_f32_envelope(module, fake_runtime):
    assert not scheduled_matmul.supports_scheduled_matmul(module, target="apple_cpu")
    with pytest.raises((ValueError, RuntimeError)):
        if module.functions[0].body[0].op_name == "tessera.softmax":
            scheduled_kernel.lower_scheduled_kernel(module, target="apple_cpu")
        else:
            scheduled_matmul.lower_scheduled_matmul(module, target="apple_cpu")


def test_retired_softmax_axis_admission_was_wider_and_is_now_closed():
    """The retired ``axis % 2 == 1`` test admitted out-of-range axes (3, -3)
    for a rank-two softmax. The scheduled route names only the last axis."""
    module = _softmax(6, 11, kwargs={"axis": 3})
    assert baseline.retired_contract(module)
    assert apple_cpu_native.native_package_kind(module) is None
    with pytest.raises(ValueError):
        scheduled_kernel.lower_scheduled_kernel(module, target="apple_cpu")


@pytest.mark.parametrize("module", [
    _matmul(7, 9, 5, dtype="fp16"),
], ids=["bnns-f16"])
def test_owed_families_are_not_claimed_by_the_scheduled_route(module):
    # Batched GEMM and the LAPACK families moved to their own native contract
    # (2026-11-04, tests/unit/test_apple_cpu_kernel_migration.py).
    assert apple_cpu_native.native_package_kind(module) is not None
    assert apple_cpu_native.scheduled_package_kind(module) is None


def test_scheduled_admission_names_apple_cpu_only():
    """``tessera.gemm`` is admitted as matmul's alias on Apple CPU alone."""
    gemm = _matmul(7, 9, 5, "tessera.gemm")
    assert scheduled_matmul.supports_scheduled_matmul(gemm, target="apple_cpu")
    for target in ("x86", "apple_gpu", "rocm_gfx1151"):
        assert not scheduled_matmul.supports_scheduled_matmul(gemm, target=target)
    assert not scheduled_kernel.supports_scheduled_kernel(
        _module("tessera.reduce", [(6, 11)], (6,), kwargs={"kind": "sum"}), target="apple_cpu")


# ---------------------------------------------------------------------------
# Tampering fails closed before the runtime is touched
# ---------------------------------------------------------------------------


def _no_runtime(monkeypatch):
    monkeypatch.setattr(apple_cpu_native, "_runtime_library_path",
                        lambda: pytest.fail("runtime touched before rejection"))


@_needs_compiler
@pytest.mark.parametrize("field,value", [
    ("m", 8), ("n", 6), ("k", 10), ("storage", "f16"), ("accum", "f16"),
    ("a_dtype", "fp16"), ("macro_tile_m", 32), ("architecture", "apple_m2"),
    ("target", "apple_gpu"), ("function_name", "other"), ("a_name", "out"),
])
def test_matmul_descriptor_field_tampering_fails_closed(field, value, monkeypatch):
    artifact = scheduled_matmul.lower_scheduled_matmul(_matmul(7, 9, 5), target="apple_cpu")
    _no_runtime(monkeypatch)
    with pytest.raises(ValueError):
        apple_cpu_native.package_scheduled_matmul(replace(artifact, **{field: value}),
                                                  pipeline_name=PIPELINE)


@_needs_compiler
@pytest.mark.parametrize("field,value", [
    ("rows", 7), ("columns", 12), ("input_shape", (6, 12)), ("output_shape", (6, 12)),
    ("dtype", "fp16"), ("storage", "f16"), ("kind", "relu"), ("workgroup_size", 128),
    ("input_name", "out"), ("function_name", "other"), ("architecture", "apple7"),
])
def test_softmax_descriptor_field_tampering_fails_closed(field, value, monkeypatch):
    artifact = scheduled_kernel.lower_scheduled_kernel(_softmax(6, 11), target="apple_cpu")
    _no_runtime(monkeypatch)
    with pytest.raises(ValueError):
        apple_cpu_native.package_scheduled_kernel(replace(artifact, **{field: value}),
                                                  pipeline_name=PIPELINE)


@_needs_compiler
@pytest.mark.parametrize("old,new", [
    ('arch = "apple_silicon_cpu"', 'arch = "apple_m2"'),
    ('accum = "f32"', 'accum = "f16"'),
    ('macro_tile_m = 16', 'macro_tile_m = 32'),
    ('activation = "none"', 'activation = "relu"'),
])
def test_matmul_schedule_tampering_fails_closed(old, new, monkeypatch):
    artifact = scheduled_matmul.lower_scheduled_matmul(_matmul(7, 9, 5), target="apple_cpu")
    line = next(l for l in artifact.schedule_ir.splitlines() if "= schedule.matmul" in l)
    assert old in line
    _no_runtime(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        apple_cpu_native.package_scheduled_matmul(
            replace(artifact, schedule_ir=artifact.schedule_ir.replace(line, line.replace(old, new))),
            pipeline_name=PIPELINE)


@_needs_compiler
@pytest.mark.parametrize("old,new", [
    ('arch = "apple_silicon_cpu"', 'arch = "apple_m2"'),
    ('exp_mode = "accurate"', 'exp_mode = "approx_exp2"'),
    ('ftz = false', 'ftz = true'),
    ('workgroup_size = 1 : i64', 'workgroup_size = 2 : i64'),
])
def test_softmax_schedule_tampering_fails_closed(old, new, monkeypatch):
    artifact = scheduled_kernel.lower_scheduled_kernel(_softmax(6, 11), target="apple_cpu")
    line = next(l for l in artifact.schedule_ir.splitlines() if "= schedule.softmax" in l)
    assert old in line
    _no_runtime(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        apple_cpu_native.package_scheduled_kernel(
            replace(artifact, schedule_ir=artifact.schedule_ir.replace(line, line.replace(old, new))),
            pipeline_name=PIPELINE)


@_needs_compiler
def test_tile_tampering_fails_closed(monkeypatch):
    artifact = scheduled_matmul.lower_scheduled_matmul(_matmul(7, 9, 5), target="apple_cpu")
    tampered = artifact.tile_ir.replace("%c9_i64 = arith.constant 9", "%c9_i64 = arith.constant 10")
    assert tampered != artifact.tile_ir
    _no_runtime(monkeypatch)
    with pytest.raises(ValueError, match="replay"):
        apple_cpu_native.package_scheduled_matmul(replace(artifact, tile_ir=tampered),
                                                  pipeline_name=PIPELINE)


@_needs_compiler
def test_target_ir_is_verified_by_the_dialect():
    with pytest.raises(RuntimeError, match="requires attribute 'source'"):
        apple_cpu_native._verified_target_ir(
            'tessera_apple.cpu.accelerate_gemm {result = "out", ordinal = 0}')
    with pytest.raises(RuntimeError):
        apple_cpu_native._verified_target_ir(
            'tessera_apple.cpu.accelerate_gemm_v2 {source = "s", result = "out", ordinal = 0}')


# ---------------------------------------------------------------------------
# Native compiler owns the admission
# ---------------------------------------------------------------------------


def _graph_text(body: str, args: str, result: str, arch: str = "apple_silicon_cpu") -> str:
    return (
        f'module attributes {{tessera.arch = "{arch}", tessera.target = "apple_cpu"}} {{\n'
        f"  func.func @f({args}) -> {result} {{\n    {body}\n"
        f"    return %y : {result}\n  }}\n}}\n"
    )


@_needs_compiler
@pytest.mark.parametrize("text", [
    _graph_text("%y = tessera.matmul %a, %b : (tensor<4x8xf32>, tensor<8x5xf32>) -> tensor<4x5xf32>",
                "%a: tensor<4x8xf32>, %b: tensor<8x5xf32>", "tensor<4x5xf32>", arch=""),
    _graph_text("%y = tessera.matmul %a, %b : (tensor<4x8xf16>, tensor<8x5xf16>) -> tensor<4x5xf32>",
                "%a: tensor<4x8xf16>, %b: tensor<8x5xf16>", "tensor<4x5xf32>"),
    _graph_text("%y = tessera.softmax %a : (tensor<2x4x8xf32>) -> tensor<2x4x8xf32>",
                "%a: tensor<2x4x8xf32>", "tensor<2x4x8xf32>"),
    _graph_text('%y = tessera.reduce %a {kind = "sum", axis = -1 : i64} : (tensor<4x8xf32>) -> tensor<4xf32>',
                "%a: tensor<4x8xf32>", "tensor<4xf32>"),
], ids=["no-arch", "f16-matmul", "rank3-softmax", "reduce"])
def test_native_graph_to_schedule_refuses_outside_the_envelope(text):
    with pytest.raises(RuntimeError, match="Graph->Schedule"):
        run_tessera_opt(find_tessera_opt(), text, "--tessera-graph-to-schedule")


# ---------------------------------------------------------------------------
# Census: the prune dashboard sees the routes
# ---------------------------------------------------------------------------


def test_prune_census_sees_the_compiled_routes():
    from tessera.compiler import bootstrap_prune_audit

    rows = {(t, f): status for t, f, _, status in bootstrap_prune_audit.family_rows()}
    assert rows[("apple_cpu", "matmul")] == "compiled"
    assert rows[("apple_cpu", "gemm")] == "compiled"
    assert rows[("apple_cpu", "softmax")] == "compiled"
    # The per-target gemm alias must not leak to another target. The other
    # Apple CPU families are pinned by test_apple_cpu_kernel_migration.py.
    assert rows[("apple_gpu", "batched_gemm")] == "gap"


# ---------------------------------------------------------------------------
# Device differential (M1 Max, Accelerate)
# ---------------------------------------------------------------------------


def _run(package, inputs, out_shape):
    from tessera import runtime

    buffers = {}
    for binding in package.descriptor.buffers:
        buffers[binding.name] = (inputs[binding.ordinal] if binding.direction == "input"
                                 else np.zeros(out_shape, np.float32))
    return runtime._submit_apple_cpu_native(package.image, package.descriptor, buffers, {}, None).copy()


@_needs_compiler
@pytest.mark.parametrize("module", _ADMITTED, ids=_ADMITTED_IDS)
def test_compiled_and_retired_images_agree_bit_for_bit_on_accelerate(module):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    # On an Apple chip the runtime dylib must build; its absence is a failure.
    assert apple_cpu_native.tools_available()
    rng = np.random.default_rng(1103)
    fn = module.functions[0]
    inputs = [rng.standard_normal(tuple(int(d) for d in a.ir_type.shape)).astype(np.float32)
              for a in fn.args]
    out_shape = tuple(int(d) for d in fn.result_types[0].shape)
    new = _run(apple_cpu_native.package_native(module, pipeline_name=PIPELINE), inputs, out_shape)
    old = _run(baseline.package(module, pipeline_name=PIPELINE), inputs, out_shape)
    assert np.array_equal(new.view(np.uint32), old.view(np.uint32))
    if fn.body[0].op_name == "tessera.softmax":
        x = inputs[0].astype(np.float64)
        e = np.exp(x - x.max(axis=1, keepdims=True))
        reference, tolerance = e / e.sum(axis=1, keepdims=True), 1e-6
    else:
        reference = inputs[0].astype(np.float64) @ inputs[1].astype(np.float64)
        tolerance = 1e-5 * max(1, inputs[0].shape[1])
    np.testing.assert_allclose(new, reference, rtol=0, atol=tolerance)
