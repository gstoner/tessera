"""E2E-REAL-6, x86 unary family: retired Graph-owned packaging vs the compiled route.

``x86_native.package_softmax`` / ``package_reduction`` and their admission
predicates used to read the Python Graph object (``_softmax_contract`` /
``_reduction_contract``) and, before 2026-09-08, author Tile IR text
(``emit_softmax_tile_ir`` / ``emit_reduce_tile_ir``). Admission and packaging
now belong to the native Graph -> Schedule -> Tile route (``scheduled_kernel``)
and ``package_scheduled_kernel``. The frozen constructors in
``tests/_support/x86_unary_baseline.py`` are the declared oracle
(Decision #31(a)); these tests are its differential.

Host-free half: over the whole envelope the old contract admitted (both
architectures), every descriptor/ABI field and every Tile-kernel attribute the
old route produced is reproduced by the compiled route; what it refused is
still refused; and each case the old contract admitted but the compiled one
refuses is pinned as intentional. Device half (an x86 host with the named
shared image; skips elsewhere): both packages run on the same inputs and must
agree bit-for-bit and with a numpy oracle.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import scheduled_kernel, x86_native
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support import x86_unary_baseline as baseline

PIPELINE = "tessera-lower-to-x86"
AVX512 = x86_native.X86_AVX512_ARCHITECTURE
BASE = x86_native.X86_BASE_ARCHITECTURE
_needs_compiler = pytest.mark.skipif(find_tessera_opt() is None, reason="requires production tessera-opt")
_ELEMENT = {"fp32": "f32", "fp16": "f16", "bf16": "bf16"}


def _type(shape: tuple[int, ...], dtype: str) -> IRType:
    dims = "x".join(map(str, shape))
    return IRType(f"tensor<{dims + 'x' if dims else ''}{_ELEMENT[dtype]}>", tuple(map(str, shape)), dtype)


def _module(op_name, shape, dtype, out_shape, out_dtype, kwargs) -> GraphIRModule:
    source, result = _type(shape, dtype), _type(out_shape, out_dtype)
    return GraphIRModule(functions=[GraphIRFunction(
        name="x86_unary", args=[IRArg("x", source)], result_types=[result],
        body=[IROp(result="o", op_name=op_name, operands=["%x"], operand_types=[str(source)],
                   result_type=str(result), kwargs=dict(kwargs))],
        return_values=["%o"],
    )])


def _softmax(shape=(3, 17), op_name="tessera.softmax", dtype="fp32", kwargs=None) -> GraphIRModule:
    if kwargs is None:
        kwargs = {} if op_name == "tessera.softmax_safe" else {"axis": -1}
    return _module(op_name, shape, dtype, shape, dtype, kwargs)


def _reduction(op_name="tessera.sum", shape=(2, 3, 5), axis=-1, keepdims=False,
               dtype="fp32", out_dtype="fp32", extra=None) -> GraphIRModule:
    normalized = axis + len(shape) if axis < 0 else axis
    out = shape[:normalized] + ((1,) if keepdims else ()) + shape[normalized + 1:]
    return _module(op_name, shape, dtype, out, out_dtype,
                   {"axis": axis, "keepdims": keepdims, **(extra or {})})


_SHAPES = [(1,), (17,), (1, 1), (3, 17), (5, 1), (2, 3, 5), (2, 1, 4, 16), (4, 33)]
_SOFTMAX_CASES = [
    (op, shape) for op in ("tessera.softmax", "tessera.softmax_safe") for shape in _SHAPES
]
_REDUCTION_OPS = ("tessera.sum", "tessera.mean", "tessera.max", "tessera.amax")
_REDUCTION_CASES = [
    (op, shape, axis, keepdims)
    for op in _REDUCTION_OPS for shape in _SHAPES
    for axis in sorted({-1, len(shape) - 1}) for keepdims in (False, True)
]


def _case_module(case):
    if case[0] in ("tessera.softmax", "tessera.softmax_safe"):
        return _softmax(case[1], case[0])
    op, shape, axis, keepdims = case
    return _reduction(op, shape, axis, keepdims)


def _attrs(tile_ir: str, family: str) -> dict[str, str]:
    """Semantic attributes of the one Tile kernel op, whitespace-normalized."""
    match = re.search(r"tile\.%s_kernel [^{]*\{([^{}]*)\}" % family, tile_ir, re.S)
    assert match, tile_ir
    pairs = re.findall(r'([\w.]+) = ("[^"]*"|-?\d+ : i64|true|false)', match[1])
    return {key: value for key, value in pairs if not key.startswith("tessera.")}


def _stub_lower(monkeypatch):
    monkeypatch.setattr(
        x86_native, "_lower",
        lambda tile, symbol, family, architecture=AVX512: (f"call @{symbol}", b"image", "c", "t"),
    )


def _packages(module, family, architecture):
    old = (baseline.package_softmax if family == "softmax" else baseline.package_reduction)(
        module, pipeline_name=PIPELINE, architecture=architecture)
    new = (x86_native.package_softmax if family == "softmax" else x86_native.package_reduction)(
        module, pipeline_name=PIPELINE, architecture=architecture)
    return old, new


_SHARED_PROVENANCE = ("kind", "shape", "axis", "keepdims", "outer", "axis_extent", "inner",
                      "rows", "columns", "storage", "accum")


@_needs_compiler
@pytest.mark.parametrize("architecture", [AVX512, BASE])
@pytest.mark.parametrize("case", _SOFTMAX_CASES + _REDUCTION_CASES, ids=str)
def test_compiled_route_reproduces_retired_descriptor_and_kernel(monkeypatch, case, architecture):
    _stub_lower(monkeypatch)
    module = _case_module(case)
    family = "softmax" if case[0].startswith("tessera.softmax") else "reduce"
    requester = x86_native.supports_softmax if family == "softmax" else x86_native.supports_reduction
    assert (baseline.supports_softmax if family == "softmax" else baseline.supports_reduction)(module)
    assert requester(module) and x86_native.supports_native_package(module)
    old, new = _packages(module, family, architecture)
    o, n = old.descriptor, new.descriptor
    assert (o.entry_symbol, o.abi_id, o.buffers, o.scalars, o.shape_guards, o.geometry, o.ordering) == (
        n.entry_symbol, n.abi_id, n.buffers, n.scalars, n.shape_guards, n.geometry, n.ordering)
    for key in _SHARED_PROVENANCE:
        if key in o.provenance:
            assert o.provenance[key] == n.provenance[key], key
    assert new.descriptor.provenance["route"] == "canonical_scheduled_tile_consumer"
    old_attrs, new_attrs = _attrs(old.tile_ir, family), _attrs(new.tile_ir, family)
    assert old_attrs == new_attrs
    if family == "reduce":
        assert n.provenance["nan_mode"] == "propagate" == new_attrs["nan_mode"].strip('"')
    assert (old.image.architecture, old.image.entry_points) == (new.image.architecture, new.image.entry_points)


_OLD_ADMITTED_NEW_REFUSED = {
    # bool(...) coercion: an integer keepdims is not a boolean (Decision #21a).
    "integer_keepdims": _reduction(keepdims=True, extra={"keepdims": 1}),
    # The retired route ignored a reduction schedule hint the x86 kernel cannot honour.
    "cooperative_reduction": _reduction(extra={"schedule": "cooperative_128"}),
    # ...and a reduction schedule hint on softmax, where it is not applicable.
    "softmax_schedule_hint": _softmax(kwargs={"axis": -1, "schedule": "serial_rows"}),
}


@pytest.mark.parametrize("name", sorted(_OLD_ADMITTED_NEW_REFUSED))
def test_cases_the_retired_route_admitted_are_refused_on_purpose(monkeypatch, name):
    module = _OLD_ADMITTED_NEW_REFUSED[name]
    family = "softmax" if module.functions[0].body[0].op_name.startswith("tessera.softmax") else "reduce"
    old_supports = baseline.supports_softmax if family == "softmax" else baseline.supports_reduction
    new_supports = x86_native.supports_softmax if family == "softmax" else x86_native.supports_reduction
    assert old_supports(module)
    assert not new_supports(module) and not x86_native.supports_native_package(module)
    monkeypatch.setattr(x86_native, "_lower", lambda *a, **k: pytest.fail("refused case reached lowering"))
    call = x86_native.package_softmax if family == "softmax" else x86_native.package_reduction
    with pytest.raises(ValueError):
        call(module, pipeline_name=PIPELINE)


_BOTH_REFUSE = {
    "fp16_softmax": _softmax(dtype="fp16"),
    "bf16_reduction": _reduction(dtype="bf16"),
    "fp16_reduction_f16_out": _reduction(dtype="fp16", out_dtype="fp16"),
    "non_last_axis_softmax": _softmax(kwargs={"axis": 0}),
    "non_last_axis_reduction": _reduction(axis=0),
    "reduction_wrong_output_shape": _module("tessera.sum", (2, 3, 5), "fp32", (2, 5), "fp32",
                                            {"axis": -1, "keepdims": False}),
    "softmax_wrong_output_shape": _module("tessera.softmax", (3, 17), "fp32", (3, 16), "fp32",
                                          {"axis": -1}),
    "dynamic_softmax": _module("tessera.softmax", (3, 17), "fp32", (3, 17), "fp32", {"axis": -1}),
    "min_reduction": _reduction("tessera.min"),
    "empty_extent": _reduction(shape=(3, 0)),
}
_dynamic = _BOTH_REFUSE["dynamic_softmax"].functions[0]
_dynamic.args[0].ir_type = IRType("tensor<?x17xf32>", ("?", "17"), "fp32")


@pytest.mark.parametrize("name", sorted(_BOTH_REFUSE))
def test_refusals_are_unchanged(name):
    module = _BOTH_REFUSE[name]
    assert not baseline.supports_softmax(module) and not baseline.supports_reduction(module)
    assert not x86_native.supports_softmax(module) and not x86_native.supports_reduction(module)


@_needs_compiler
def test_reduction_without_propagating_nan_policy_fails_closed(monkeypatch):
    _stub_lower(monkeypatch)
    artifact = scheduled_kernel.lower_scheduled_kernel(_reduction(), target="x86")
    monkeypatch.setattr(
        "tessera.compiler.native_unary_contract.verify_unary_ancestry", lambda *a, **k: None)
    from dataclasses import replace

    forged = replace(artifact, tile_ir=artifact.tile_ir.replace('nan_mode = "propagate"', 'nan_mode = "ignore"'))
    with pytest.raises(ValueError, match="nan_mode"):
        x86_native.package_scheduled_kernel(forged, pipeline_name=PIPELINE)


def test_retired_constructors_left_production():
    for name in ("emit_softmax_tile_ir", "emit_reduce_tile_ir", "_softmax_contract",
                 "_reduction_contract", "_package_unary"):
        assert not hasattr(x86_native, name), name


def test_unsupported_architecture_refuses_before_lowering(monkeypatch):
    monkeypatch.setattr(scheduled_kernel, "lower_scheduled_kernel",
                        lambda *a, **k: pytest.fail("unsupported architecture lowered"))
    with pytest.raises(ValueError, match="architecture"):
        x86_native.package_softmax(_softmax(), pipeline_name=PIPELINE, architecture="x86_64_avx2")


# ---------------------------------------------------------------------------
# Device differential: both images, the same inputs, bitwise.
# ---------------------------------------------------------------------------

def _inputs(shape, seed):
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(shape) * 8).astype(np.float32)
    flat = x.reshape(-1)
    if flat.size >= 4:
        flat[0], flat[1] = 88.0, -88.0  # exp range edges
    return x


def _args(package, x):
    d = package.descriptor
    out_shape = tuple(g.value for g in d.shape_guards if g.binding == d.buffers[1].name)
    output = np.zeros(out_shape, np.float32)
    args = {d.buffers[0].name: x, d.buffers[1].name: output}
    rows = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    if d.abi_id == x86_native.X86_SOFTMAX_F32_ABI:
        args.update(Rows=rows, K=x.shape[-1])
    else:
        args.update(Outer=rows, AxisExtent=x.shape[-1], Inner=1)
    return args, output


def _run(package, x):
    args, output = _args(package, x)
    artifact = rt.RuntimeArtifact(metadata={"target": "x86"}, native_image=package.image,
                                  launch_descriptor=package.descriptor, tile_ir=package.tile_ir,
                                  target_ir=package.target_ir)
    result = rt.launch(artifact, args)
    assert result["ok"], result
    return output


def _oracle(case, x):
    if case[0].startswith("tessera.softmax"):
        e = np.exp(x.astype(np.float64) - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)
    op, _, _, keepdims = case
    fn = {"tessera.sum": np.sum, "tessera.mean": np.mean, "tessera.max": np.max,
          "tessera.amax": np.max}[op]
    return fn(x.astype(np.float64), axis=-1, keepdims=keepdims)


@_needs_compiler
@pytest.mark.parametrize("architecture", [AVX512, BASE])
@pytest.mark.parametrize("case", _SOFTMAX_CASES + _REDUCTION_CASES, ids=str)
def test_device_retired_and_compiled_images_agree_bitwise(case, architecture):
    if not x86_native.tools_available_for_architecture(architecture):
        pytest.skip(f"{architecture} shared image not available on this host")
    module = _case_module(case)
    family = "softmax" if case[0].startswith("tessera.softmax") else "reduce"
    old, new = _packages(module, family, architecture)
    assert old.image.payload == new.image.payload
    shape = case[1]
    for seed in (0, 1):
        x = _inputs(shape, seed)
        got_old, got_new = _run(old, x), _run(new, x)
        assert got_old.tobytes() == got_new.tobytes()
        np.testing.assert_allclose(got_new, _oracle(case, x), rtol=2e-5, atol=2e-6)


@_needs_compiler
@pytest.mark.parametrize("architecture", [AVX512, BASE])
@pytest.mark.parametrize("op", _REDUCTION_OPS)
def test_device_reduction_nan_propagates_on_both_routes(op, architecture):
    if not x86_native.tools_available_for_architecture(architecture):
        pytest.skip(f"{architecture} shared image not available on this host")
    old, new = _packages(_reduction(op, (4, 33)), "reduce", architecture)
    x = _inputs((4, 33), 3)
    x[1, 7] = np.nan
    got_old, got_new = _run(old, x), _run(new, x)
    assert got_old.tobytes() == got_new.tobytes()
    assert np.isnan(got_new[1]) and np.isfinite(np.delete(got_new, 1)).all()


@_needs_compiler
@pytest.mark.parametrize("architecture", [AVX512, BASE])
def test_device_shapes_share_one_loaded_image(monkeypatch, architecture):
    """Distinct shapes are distinct images (their Target IR binds the launch
    constants) but one payload, so the runtime loads the object once."""
    if not x86_native.tools_available_for_architecture(architecture):
        pytest.skip(f"{architecture} shared image not available on this host")
    monkeypatch.setattr(rt, "_x86_native_image_libraries", {})
    monkeypatch.setattr(rt, "_x86_native_payload_libraries", {})
    packages = [x86_native.package_softmax(_softmax(shape), pipeline_name=PIPELINE,
                                           architecture=architecture)
                for shape in ((3, 17), (4, 33), (2, 3, 5))]
    assert len({p.image.image_digest for p in packages}) == 3
    assert len({p.image.payload for p in packages}) == 1
    handles = {rt._load_x86_native_image(p.image)._handle for p in packages}
    assert len(handles) == 1 and len(rt._x86_native_payload_libraries) == 1
    for package, shape in zip(packages, ((3, 17), (4, 33), (2, 3, 5))):
        x = _inputs(shape, 5)
        np.testing.assert_allclose(_run(package, x), _oracle(("tessera.softmax", shape), x),
                                   rtol=2e-5, atol=2e-6)
