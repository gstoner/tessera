"""E2E-REAL-6, ROCm unary family: retired Graph-owned packaging vs the compiled route.

``rocm_native.package_softmax`` / ``package_reduction`` used to read the Python
Graph object and author Tile IR text. They now lower through the native
Graph -> Schedule -> Tile route (``scheduled_kernel``) and package the replayed
artifact. The frozen constructors in ``tests/_support/rocm_unary_baseline.py``
are the declared oracle (Decision #31(a)); these tests are its differential.

Host-free half: every descriptor/ABI field and every Tile-kernel attribute the
old route produced is reproduced by the compiled route over the whole envelope
the old contract admitted, and everything it refused is still refused. Device
half (``TESSERA_ROCM_E2E_DEVICE_TEST=1`` on the exact gfx1151 host): both
images run on the same inputs and must agree bit-for-bit and with an oracle.
"""

from __future__ import annotations

import os
import re
from dataclasses import replace

import numpy as np
import pytest

from tessera.compiler import rocm_native, scheduled_kernel
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support import rocm_unary_baseline as baseline

PIPELINE = "tessera-lower-to-rocm"
_needs_compiler = pytest.mark.skipif(find_tessera_opt() is None, reason="requires production tessera-opt")
_ELEMENT = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}


def _type(shape: tuple[int, ...], dtype: str) -> IRType:
    dims = "x".join(map(str, shape))
    return IRType(f"tensor<{dims + 'x' if dims else ''}{_ELEMENT[dtype]}>", tuple(map(str, shape)), dtype)


def _module(op_name: str, shape, dtype, out_shape, out_dtype, kwargs) -> GraphIRModule:
    source, result = _type(shape, dtype), _type(out_shape, out_dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name="gfx1151_unary", args=[IRArg("x", source)], result_types=[result],
        body=[IROp(result="o", op_name=op_name, operands=["%x"], operand_types=[str(source)],
                   result_type=str(result), kwargs=dict(kwargs))],
        return_values=["%o"],
    )])


def _softmax(dtype="fp32", shape=(3, 17), op_name="tessera.softmax") -> GraphIRModule:
    kwargs = {} if op_name == "tessera.softmax_safe" else {"axis": -1}
    return _module(op_name, shape, dtype, shape, dtype, kwargs)


def _reduction(dtype="fp32", kind="sum", shape=(2, 3, 5), axis=1, keepdims=False, op_name=None):
    normalized = axis + len(shape) if axis < 0 else axis
    out = shape[:normalized] + ((1,) if keepdims else ()) + shape[normalized + 1:]
    op = op_name or {"sum": "tessera.sum", "mean": "tessera.mean", "max": "tessera.max"}[kind]
    kwargs = {"axis": axis, "keepdims": keepdims}
    if op == "tessera.reduce":
        kwargs["kind"] = kind
    return _module(op, shape, dtype, out, "fp32", kwargs)


#: The whole envelope the retired contracts admitted: storage x shape x axis x keepdims.
SOFTMAX_CASES = [
    (dtype, shape, op)
    for dtype in ("fp32", "fp16")
    for shape in ((1,), (9,), (1, 1), (3, 17), (4, 256), (2, 257), (2, 3, 5))
    for op in ("tessera.softmax", "tessera.softmax_safe")
]
REDUCTION_CASES = [
    (dtype, kind, shape, axis, keepdims, op)
    for dtype in ("fp32", "fp16", "bf16")
    for kind, op in (("sum", None), ("mean", None), ("max", None), ("max", "tessera.amax"),
                     ("sum", "tessera.reduce"))
    for shape, axis in (((7,), 0), ((2, 3, 5), 0), ((2, 3, 5), 1), ((2, 3, 5), -1), ((4, 257), 1))
    for keepdims in (False, True)
]


#: A Target IR stand-in naming one directive per family, so the packager can
#: read the image's kernel symbol from it (the shape-free route does).
_FAKE_TARGET = (
    'module {\n  tessera_rocm.softmax {name = "tessera_rocm_softmax_fake"}\n'
    '  tessera_rocm.reduce {name = "tessera_rocm_reduction_fake"}\n}\n'
)


def _recording_compile(calls: list[str]):
    def compile_(tile_ir: str, **_kwargs):
        calls.append(tile_ir)
        return (_FAKE_TARGET, "backend", b"\x7fELFrocm-unary", "compiler", "toolchain", (), "cold")
    return compile_


def _kernel_attrs(tile_ir: str) -> dict[str, str]:
    """The attribute dictionary of the one tile.{softmax,reduce}_kernel op."""
    match = re.search(r"tile\.(?:softmax|reduce)_kernel [^{]*\{([^}]*)\}", tile_ir)
    assert match is not None, tile_ir
    attrs = {}
    for item in re.findall(r'([\w.]+) = ("[^"]*"|[^,]+)', match[1]):
        attrs[item[0]] = item[1].strip()
    # Hash/workgroup bookkeeping is route identity, not kernel semantics.
    attrs.pop("tessera.schedule_hash", None)
    attrs.pop("tessera.workgroup_size", None)
    return attrs


_SEMANTIC_PROVENANCE = ("storage", "accum", "axis", "exp_mode", "ftz", "rows", "columns",
                        "kind", "keepdims", "nan_mode", "outer", "axis_extent", "inner", "shape")


def _assert_same_launch_contract(old, new) -> None:
    for field in ("abi_id", "buffers", "scalars", "shape_guards", "geometry", "ordering"):
        assert getattr(old.descriptor, field) == getattr(new.descriptor, field), field
    for key in _SEMANTIC_PROVENANCE:
        if key in old.descriptor.provenance:
            assert new.descriptor.provenance[key] == old.descriptor.provenance[key], key
    assert new.image.target == old.image.target == "rocm_gfx1151"
    assert new.image.architecture == old.image.architecture == "gfx1151"
    assert new.image.entry_points[0].abi_id == old.image.entry_points[0].abi_id
    # The compiled route is the Schedule-consuming one, not a Python Tile author.
    assert new.descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    assert "route" not in old.descriptor.provenance


@_needs_compiler
@pytest.mark.parametrize("dtype,shape,op_name", SOFTMAX_CASES)
def test_softmax_compiled_route_reproduces_retired_contract(monkeypatch, dtype, shape, op_name) -> None:
    calls: list[str] = []
    monkeypatch.setattr(rocm_native, "_compile_tile_ir", _recording_compile(calls))
    monkeypatch.setattr(rocm_native, "_compile_shape_free_tile_ir", _recording_compile(calls))
    module = _softmax(dtype, shape, op_name)
    assert rocm_native.supports_softmax(module)
    assert rocm_native.native_package_kind(module) == "softmax"
    old = baseline.baseline_softmax(module, pipeline_name=PIPELINE)
    new = rocm_native.package_softmax(module, pipeline_name=PIPELINE)
    _assert_same_launch_contract(old, new)
    old_tile, new_tile = calls
    assert _kernel_attrs(old_tile) == _kernel_attrs(new_tile)
    # The new Tile text is the native replay of its Schedule record.
    assert new.tile_ir == new_tile and "schedule.softmax" not in new_tile


@_needs_compiler
@pytest.mark.parametrize("dtype,kind,shape,axis,keepdims,op_name", REDUCTION_CASES)
def test_reduction_compiled_route_reproduces_retired_contract(
    monkeypatch, dtype, kind, shape, axis, keepdims, op_name
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(rocm_native, "_compile_reduction_tile_ir", _recording_compile(calls))
    monkeypatch.setattr(rocm_native, "_compile_shape_free_tile_ir", _recording_compile(calls))
    module = _reduction(dtype, kind, shape, axis, keepdims, op_name)
    assert rocm_native.supports_reduction(module)
    assert rocm_native.native_package_kind(module) == "reduction"
    old = baseline.baseline_reduction(module, pipeline_name=PIPELINE)
    new = rocm_native.package_reduction(module, pipeline_name=PIPELINE)
    _assert_same_launch_contract(old, new)
    old_tile, new_tile = calls
    old_attrs, new_attrs = _kernel_attrs(old_tile), _kernel_attrs(new_tile)
    assert old_attrs == new_attrs
    assert new_attrs["storage"] == f'"{_ELEMENT[dtype]}"' and new_attrs["accum"] == '"f32"'
    assert new.descriptor.provenance["nan_mode"] == "propagate"


@_needs_compiler
def test_reduce_kind_is_read_from_ir_not_from_the_op_name(monkeypatch) -> None:
    """The retired constructor summed ``tessera.reduce {kind = "max"}``.

    It derived its combiner from the op *name*; the tracer spells
    ``ts.reduce(x, op="max")`` as ``tessera.reduce {kind = "max"}``. The compiled
    route carries ``kind`` through Schedule and Tile, and refuses the combiners
    gfx1151 has no proved kernel for instead of silently summing them.
    """
    calls: list[str] = []
    monkeypatch.setattr(rocm_native, "_compile_reduction_tile_ir", _recording_compile(calls))
    monkeypatch.setattr(rocm_native, "_compile_shape_free_tile_ir", _recording_compile(calls))
    module = _reduction("fp32", "max", op_name="tessera.reduce")
    old = baseline.baseline_reduction(module, pipeline_name=PIPELINE)
    new = rocm_native.package_reduction(module, pipeline_name=PIPELINE)
    assert old.descriptor.provenance["kind"] == "sum"  # the retired defect, kept as evidence
    assert new.descriptor.provenance["kind"] == "max"
    assert 'kind = "max"' in calls[1]
    for unsupported in ("min", "prod"):
        refused = _reduction("fp32", unsupported, op_name="tessera.reduce")
        assert not rocm_native.supports_reduction(refused)
        with pytest.raises(ValueError, match="sum/mean/max"):
            rocm_native.package_reduction(refused, pipeline_name=PIPELINE)


def _refused_modules():
    axis0 = _softmax()
    axis0.functions[0].body[0].kwargs["axis"] = 0
    dynamic = _softmax()
    dynamic.functions[0].args[0].ir_type = IRType("tensor<?x17xf32>", ("?", "17"), "fp32")
    wrong_result = _softmax()
    wrong_result.functions[0].result_types[0] = _type((3, 17), "fp16")
    narrow_reduce_out = _reduction("fp16")
    narrow_reduce_out.functions[0].result_types[0] = _type((2, 5), "fp16")
    bad_shape = _reduction()
    bad_shape.functions[0].result_types[0] = _type((2, 3), "fp32")
    return {
        "softmax_bf16": _softmax("bf16"),
        "softmax_axis0": axis0,
        "softmax_dynamic": dynamic,
        "softmax_result_dtype": wrong_result,
        "reduction_narrow_output": narrow_reduce_out,
        "reduction_output_shape": bad_shape,
        "reduction_axis_out_of_range": _module("tessera.sum", (2, 3), "fp32", (2,), "fp32", {"axis": 5}),
    }


@pytest.mark.parametrize("case", sorted(_refused_modules()))
def test_retired_refusals_still_refuse(case) -> None:
    """Whatever the old contract refused, the compiled route refuses too."""
    module = _refused_modules()[case]
    assert baseline._softmax_contract(module) is None and baseline._reduction_contract(module) is None
    assert not rocm_native.supports_native_package(module)
    packager = rocm_native.package_softmax if case.startswith("softmax") else rocm_native.package_reduction
    with pytest.raises(ValueError):
        packager(module, pipeline_name=PIPELINE)
    with pytest.raises(ValueError, match="requires one supported static Graph contract"):
        rocm_native.package_native(module, pipeline_name=PIPELINE)


def _tightened_modules():
    int_keepdims = _reduction()
    int_keepdims.functions[0].body[0].kwargs["keepdims"] = 0
    no_kind = _reduction(op_name="tessera.reduce")
    del no_kind.functions[0].body[0].kwargs["kind"]
    schedule_hint = _softmax()
    schedule_hint.functions[0].body[0].kwargs["schedule"] = "cooperative_128"
    short_output = _softmax()
    short_output.functions[0].result_types[0] = _type((3, 16), "fp32")
    return {
        # A semantic key must be typed (#21a); the tracer emits a bool.
        "reduction_int_keepdims": int_keepdims,
        # The retired route summed a combiner-less ``tessera.reduce``.
        "reduction_missing_kind": no_kind,
        # A reduction-schedule hint is not a softmax policy.
        "softmax_schedule_hint": schedule_hint,
        # The retired route guarded the output with the *input* shape.
        "softmax_output_shape": short_output,
    }


@pytest.mark.parametrize("case", sorted(_tightened_modules()))
def test_retired_admissions_the_compiled_route_refuses_by_design(case) -> None:
    """Old-admitted, new-refused: every such case is listed and intended."""
    module = _tightened_modules()[case]
    old_contract = baseline._softmax_contract if case.startswith("softmax") else baseline._reduction_contract
    assert old_contract(module) is not None, "the retired constructor admitted this"
    assert not rocm_native.supports_native_package(module)
    packager = rocm_native.package_softmax if case.startswith("softmax") else rocm_native.package_reduction
    with pytest.raises(ValueError):
        packager(module, pipeline_name=PIPELINE)


@pytest.mark.parametrize("module", [_softmax("fp16"), _reduction("bf16"), _reduction("fp32", keepdims=True)],
                         ids=["softmax_f16", "reduce_bf16", "reduce_keepdims"])
def test_gfx1201_keeps_its_proved_f32_envelope(module) -> None:
    """The gfx1151 envelope is not inherited by gfx1201 (no device rows there)."""
    assert not scheduled_kernel.supports_scheduled_kernel(module, target="rocm_gfx1201")
    assert scheduled_kernel.supports_scheduled_kernel(module, target="rocm_gfx1151")


@pytest.mark.parametrize("target", ["nvidia_sm120", "apple_gpu", "rocm_gfx1201"])
def test_softmax_safe_admission_is_limited_to_proved_consumers(target) -> None:
    # gfx1151 (this cut) and x86 (E2E-REAL-6 x86 unary cut, 2026-09-28,
    # tests/unit/test_x86_unary_differential.py) carry device rows; nothing else does.
    assert not scheduled_kernel.supports_scheduled_kernel(
        _softmax(op_name="tessera.softmax_safe"), target=target
    )


@_needs_compiler
def test_forged_narrow_gfx1201_artifact_is_refused_before_compile(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a refused artifact must not reach target compilation")

    monkeypatch.setattr(rocm_native, "_compile_native_tile_ir", forbidden)
    artifact = scheduled_kernel.lower_scheduled_kernel(_reduction("bf16"), target="rocm_gfx1151")
    with pytest.raises(ValueError, match="gfx1201 scheduled semantic kernel has device proof only"):
        rocm_native.package_scheduled_kernel(replace(artifact, architecture="gfx1201"), pipeline_name=PIPELINE)
    with pytest.raises(ValueError, match="storage/f32-accumulation"):
        rocm_native.package_scheduled_kernel(replace(artifact, storage="f16"), pipeline_name=PIPELINE)


@_needs_compiler
def test_narrow_projection_rejects_stale_storage_and_keepdims(monkeypatch) -> None:
    """The descriptor is projected from native IR: a relabelled field fails."""
    from tessera.compiler.native_unary_contract import verify_unary_ancestry

    artifact = scheduled_kernel.lower_scheduled_kernel(
        _reduction("fp16", keepdims=True), target="rocm_gfx1151")
    verify_unary_ancestry(artifact, target="rocm", architecture="gfx1151")
    for stale in (replace(artifact, keepdims=False), replace(artifact, dtype="bf16", storage="bf16"),
                  replace(artifact, output_shape=(2, 5))):
        with pytest.raises(ValueError):
            verify_unary_ancestry(stale, target="rocm", architecture="gfx1151")


@_needs_compiler
def test_driver_selects_the_scheduled_boundary_for_the_whole_envelope(monkeypatch) -> None:
    """The production caller (driver) never reaches a Graph-owned constructor."""
    from tessera.compiler.driver import compile_graph_module

    calls: list[str] = []
    monkeypatch.setattr(rocm_native, "_compile_tile_ir", _recording_compile(calls))
    monkeypatch.setattr(rocm_native, "_compile_reduction_tile_ir", _recording_compile(calls))
    monkeypatch.setattr(rocm_native, "_compile_shape_free_tile_ir", _recording_compile(calls))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("gfx1151 unary packaging must consume the scheduled artifact")

    monkeypatch.setattr(rocm_native, "package_native", forbidden)
    for module in (_softmax("fp16"), _softmax(op_name="tessera.softmax_safe"),
                   _reduction("bf16", "max", axis=0, keepdims=True)):
        bundle = compile_graph_module(module, source_origin="unit", target="rocm_gfx1151",
                                      options={"package_native": True}, enable_tool_validation=False)
        assert bundle.orchestration_state == "launchable"
        assert bundle.schedule is not None and "schedule.artifact" in bundle.schedule.text
        assert bundle.launch_descriptor is not None
        assert bundle.launch_descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    assert len(calls) == 3


# --- exact-device differential (gfx1151 only) --------------------------------------------------

_device = pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1",
    reason="set TESSERA_ROCM_E2E_DEVICE_TEST=1 on the exact gfx1151 host",
)


def _numpy_dtype(dtype: str):
    if dtype == "bf16":
        return pytest.importorskip("ml_dtypes").bfloat16
    return {"fp16": np.float16, "fp32": np.float32}[dtype]


def _launch(package, x: np.ndarray, output: np.ndarray) -> np.ndarray:
    from tessera import runtime as rt

    provenance = package.descriptor.provenance
    names = [item.name for item in package.descriptor.scalars]
    scalars = ({"Rows": provenance["rows"], "K": provenance["columns"]} if names == ["Rows", "K"] else
               {"Outer": provenance["outer"], "AxisExtent": provenance["axis_extent"], "Inner": provenance["inner"]})
    artifact = rt.RuntimeArtifact(graph_ir="graph", tile_ir=package.tile_ir, target_ir=package.target_ir,
                                  metadata={"target": "rocm_gfx1151"}, native_image=package.image,
                                  launch_descriptor=package.descriptor)
    result = rt.launch(artifact, {"x": x, "o": output, **scalars})
    assert result["ok"] is True, result.get("reason")
    assert result["execution_kind"] == "native_gpu"
    return np.array(result["output"], copy=True)


def _inputs(shape, dtype, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) * 3.0
    if x.size >= 4:
        flat = x.reshape(-1)
        flat[0], flat[1], flat[-1] = 0.0, -0.0, 11.5  # signed zero and a large magnitude
    return np.ascontiguousarray(x, dtype=_numpy_dtype(dtype))


@_device
@pytest.mark.hardware_rocm
@pytest.mark.parametrize("dtype,shape,op_name", SOFTMAX_CASES)
def test_exact_gfx1151_softmax_images_agree_bitwise(dtype, shape, op_name) -> None:
    from tessera import runtime as rt

    assert rt._rocm_live_arch() == "gfx1151", "requires the exact owning device"
    module = _softmax(dtype, shape, op_name)
    old = baseline.baseline_softmax(module, pipeline_name=PIPELINE)
    new = rocm_native.package_softmax(module, pipeline_name=PIPELINE)
    x = _inputs(shape, dtype, 1151 + len(shape))
    old_out = _launch(old, x, np.zeros_like(x))
    new_out = _launch(new, x, np.zeros_like(x))
    np.testing.assert_array_equal(new_out.view(np.uint8), old_out.view(np.uint8))
    xf = x.astype(np.float64)
    e = np.exp(xf - xf.max(axis=-1, keepdims=True))
    expected = e / e.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(new_out.astype(np.float64), expected, atol=3e-3 if dtype == "fp16" else 2e-6, rtol=0)


@_device
@pytest.mark.hardware_rocm
@pytest.mark.parametrize("dtype,kind,shape,axis,keepdims,op_name", REDUCTION_CASES)
def test_exact_gfx1151_reduction_images_agree_bitwise(dtype, kind, shape, axis, keepdims, op_name) -> None:
    from tessera import runtime as rt

    assert rt._rocm_live_arch() == "gfx1151", "requires the exact owning device"
    module = _reduction(dtype, kind, shape, axis, keepdims, op_name)
    old = baseline.baseline_reduction(module, pipeline_name=PIPELINE)
    new = rocm_native.package_reduction(module, pipeline_name=PIPELINE)
    x = _inputs(shape, dtype, 2202 + axis)
    out_shape = tuple(item.value for item in new.descriptor.shape_guards if item.binding == "o")
    old_out = _launch(old, x, np.zeros(out_shape, np.float32))
    new_out = _launch(new, x, np.zeros(out_shape, np.float32))
    np.testing.assert_array_equal(new_out.view(np.uint32), old_out.view(np.uint32))
    oracle = {"sum": np.sum, "mean": np.mean, "max": np.max}[kind]
    expected = oracle(x.astype(np.float64), axis=axis, keepdims=keepdims)
    np.testing.assert_allclose(new_out, expected, atol=5e-5 * max(1, shape[axis]), rtol=1e-5)


@_device
@pytest.mark.hardware_rocm
def test_exact_gfx1151_reduce_kind_max_now_computes_max() -> None:
    """Device evidence for the retired constructor's kind defect."""
    from tessera import runtime as rt

    assert rt._rocm_live_arch() == "gfx1151", "requires the exact owning device"
    module = _reduction("fp32", "max", (2, 3, 5), 1, False, "tessera.reduce")
    x = _inputs((2, 3, 5), "fp32", 7)
    old_out = _launch(baseline.baseline_reduction(module, pipeline_name=PIPELINE), x, np.zeros((2, 5), np.float32))
    new_out = _launch(rocm_native.package_reduction(module, pipeline_name=PIPELINE), x, np.zeros((2, 5), np.float32))
    np.testing.assert_allclose(old_out, x.sum(axis=1), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(new_out, x.max(axis=1))
