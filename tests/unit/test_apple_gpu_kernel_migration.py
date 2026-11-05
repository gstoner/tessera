"""E2E-REAL-6, Apple GPU linalg: retired Graph-owned packaging vs the compiled route.

``apple_native.package_svd`` and the generic value branch of
``apple_native.package_native`` used to read the Python Graph object and build
the Metal launch descriptor for reduced ``svd``, ``cholesky``, ``tri_solve``
and ``cholesky_solve``. They now lower through the native Graph -> Schedule ->
Tile boundaries (``NativeAppleGPUKernel.h``) and the Apple GPU value lowering
(``--tile-to-apple_gpu=value=true``), and package the replayed artifact,
reaching the C ABI through a verified ``tessera_apple.gpu.kernel_call``. The
frozen constructors in ``tests/_support/apple_gpu_linalg_baseline.py`` are the
declared oracle (Decision #31(a)); these tests are its differential.

Host-free half: every descriptor/ABI field the old route produced is
reproduced over the envelope both admit, the envelope the old route admitted
and computed wrongly is now refused by name, and a tampered artifact fails
closed before the runtime is touched. Device half (an Apple GPU with the real
runtime dylib, the M1 Max): both images run on the same inputs and must agree
bit-for-bit, and with a float64 oracle.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tessera.compiler import apple_native, scheduled_apple_gpu_kernel as native
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests._support import apple_gpu_linalg_baseline as baseline

PIPELINE = "tessera-lower-to-apple_gpu"
ROOT = Path(__file__).resolve().parents[2]
_DEFAULTS = {"lower": True, "trans": False, "unit_diag": False, "full_matrices": False}


def _value_lowering_registered() -> bool:
    return native.lowering_available()


#: The Target IR is produced by the Apple GPU value lowering, which exists only
#: where the production compiler was configured with the Apple backend.
_needs_compiler = pytest.mark.skipif(
    not _value_lowering_registered(),
    reason="requires production tessera-opt with the native Apple GPU value lowering",
)


def _type(shape) -> IRType:
    return IRType(f"tensor<{'x'.join(map(str, shape))}xf32>", tuple(map(str, shape)), "fp32")


def _module(op_name, inputs, results, *, kwargs=None, dtype="fp32") -> GraphIRModule:
    """One static Graph call; ``results`` is ``[(name, shape), ...]``."""
    def typed(shape):
        if dtype == "fp32":
            return _type(shape)
        return IRType(f"tensor<{'x'.join(map(str, shape))}xf16>", tuple(map(str, shape)), dtype)
    args = [IRArg(f"a{i}", typed(shape)) for i, shape in enumerate(inputs)]
    result_types = [typed(shape) for _, shape in results]
    names = [name for name, _ in results]
    return GraphIRModule(functions=[GraphIRFunction(
        name="apple_gpu_kernel", args=args, result_types=result_types,
        body=[IROp(result=",".join(names), op_name=op_name, operands=[f"%{a.name}" for a in args],
                   operand_types=[str(a.ir_type) for a in args],
                   result_type=(str(result_types[0]) if len(result_types) == 1
                                else "(" + ", ".join(map(str, result_types)) + ")"),
                   kwargs=kwargs or {})],
        return_values=[f"%{name}" for name in names],
    )])


def _cholesky(*shape, **kw):
    return _module("tessera.cholesky", [shape], [("out", shape)], **kw)


def _solve(op, matrix, rhs, **kw):
    return _module(op, [matrix, rhs], [("out", rhs)], **kw)


def _svd(*shape, **kw):
    lead, (m, n) = shape[:-2], shape[-2:]
    return _module("tessera.svd", [shape], [("u", (*lead, m, n)), ("s", (*lead, n)), ("vh", (*lead, n, n))], **kw)


_ADMITTED = {
    "chol-1": _cholesky(1, 1), "chol-6": _cholesky(6, 6),
    "chol-lower": _cholesky(5, 5, kwargs={"lower": True}), "chol-batched": _cholesky(3, 4, 4),
    "tri-default": _solve("tessera.tri_solve", (5, 5), (5, 3)),
    "tri-upper": _solve("tessera.tri_solve", (5, 5), (5, 3), kwargs={"lower": False}),
    "tri-trans": _solve("tessera.tri_solve", (5, 5), (5, 2), kwargs={"trans": True}),
    "tri-unit": _solve("tessera.tri_solve", (4, 4), (4, 4), kwargs={"unit_diag": True}),
    "tri-upper-trans-unit": _solve("tessera.tri_solve", (6, 6), (6, 1),
                                   kwargs={"lower": False, "trans": True, "unit_diag": True}),
    "tri-batched": _solve("tessera.tri_solve", (2, 5, 5), (2, 5, 3), kwargs={"lower": False}),
    "cholsolve-default": _solve("tessera.cholesky_solve", (5, 5), (5, 3)),
    "cholsolve-lower": _solve("tessera.cholesky_solve", (4, 4), (4, 2), kwargs={"lower": True}),
    "cholsolve-batched": _solve("tessera.cholesky_solve", (2, 3, 3), (2, 3, 2)),
    "svd-6x4": _svd(6, 4), "svd-4x4": _svd(4, 4), "svd-1x1": _svd(1, 1),
    "svd-reduced": _svd(7, 3, kwargs={"full_matrices": False}), "svd-batched": _svd(2, 6, 4),
}


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    library = tmp_path / "libTesseraAppleRuntime.dylib"
    library.write_bytes(b"apple-gpu-kernel-migration-fixture")
    monkeypatch.setattr(apple_native, "_runtime_library_path", lambda: library)
    return library


def _retired_admits(module) -> bool:
    if module.functions[0].body[0].op_name == "tessera.svd":
        return baseline.retired_svd_contract(module) is not None
    return baseline.retired_value_contract(module)


# ---------------------------------------------------------------------------
# Differential: the compiled route reproduces the retired descriptor
# ---------------------------------------------------------------------------


@_needs_compiler
@pytest.mark.parametrize("module", list(_ADMITTED.values()), ids=list(_ADMITTED))
def test_compiled_route_reproduces_retired_descriptor(module, fake_runtime):
    assert _retired_admits(module)
    old = baseline.package(module, pipeline_name=PIPELINE)
    new = apple_native.package_native(module, pipeline_name=PIPELINE)
    for field in ("entry_symbol", "abi_id", "buffers", "shape_guards", "scalars", "geometry", "ordering"):
        assert getattr(new.descriptor, field) == getattr(old.descriptor, field), field
    old_p, new_p = old.descriptor.provenance, new.descriptor.provenance
    for key in ("route", "op_kind"):
        assert new_p[key] == old_p[key], key
    if new_p["op_kind"] == "svd":
        for key in ("result_order", "full_matrices", "tall_only", "batch"):
            assert new_p[key] == old_p[key], key
    else:
        old_call, new_call = old_p["value_call"], new_p["value_call"]
        for key in ("op", "op_kind", "symbol", "status"):
            assert new_call[key] == old_call[key], key
        # The retired call spread whatever keywords the Graph op carried; the
        # new one materializes every flag the delegate reads. They agree on the
        # value the runtime reads (its own defaults are the ODS defaults).
        for flag in set(_DEFAULTS) & (set(old_call) | set(new_call)):
            assert new_call[flag] == old_call.get(flag, _DEFAULTS[flag]), flag
    assert new.image.entry_points == old.image.entry_points
    assert new.image.payload == old.image.payload
    assert (new.image.target, new.image.architecture, new.image.binary_format) == (
        old.image.target, old.image.architecture, old.image.binary_format)
    # The new route is the scheduled one, and says so.
    kind = new_p["op_kind"]
    assert new_p["work_item"] == "E2E-REAL-6"
    assert new_p["target_ir_op"] == "tessera_apple.gpu.kernel_call"
    digest = new_p["schedule_digest"]
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert f"tessera.{kind} " not in new.tile_ir and f"tile.{kind} " in new.tile_ir
    assert new.target_ir.count("tessera_apple.gpu.kernel_call") == 1
    assert f'tessera.schedule_hash = "{digest}"' in new.target_ir
    assert f'symbol = "{new.descriptor.entry_symbol}"' in new.target_ir


@_needs_compiler
@pytest.mark.parametrize("name", ["chol-6", "tri-upper", "cholsolve-default", "svd-6x4", "svd-batched"])
def test_driver_routes_the_families_through_the_scheduled_package(name, fake_runtime, monkeypatch):
    from tessera.compiler.driver import compile_graph_module

    monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(fake_runtime))
    bundle = compile_graph_module(
        _ADMITTED[name], source_origin="apple-gpu-kernel-migration-driver", target="apple_gpu",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["work_item"] == "E2E-REAL-6"
    assert bundle.launch_descriptor.provenance["schedule_digest"]
    assert "tessera_apple.gpu.kernel_call" in bundle.target_ir.text


def test_driver_does_not_claim_packaging_without_the_lowering(fake_runtime, monkeypatch):
    """A host whose tessera-opt cannot produce the route packages nothing for
    these families rather than falling back to the retired constructor."""
    from tessera.compiler import driver

    monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(fake_runtime))
    monkeypatch.setattr(native, "lowering_available", lambda: False)
    module = _ADMITTED["svd-6x4"]
    assert apple_native.native_package_kind(module) == "svd"
    assert not apple_native.native_package_available(module)
    assert not driver.canonical_compile_options(module, target="apple_gpu").get("package_native")


def test_packaging_names_tessera_opt_when_it_cannot_lower(fake_runtime, monkeypatch, tmp_path):
    """Fail closed, naming the tool, on a host whose tessera-opt lacks the lowering."""
    fake_tool = tmp_path / "tessera-opt"
    fake_tool.write_text("#!/bin/sh\nexit 1\n")
    fake_tool.chmod(0o755)
    monkeypatch.setattr("tessera.compiler.scheduled_matmul.find_tessera_opt", lambda: fake_tool)
    native.apple_gpu_value_lowering_available.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="tessera-opt"):
            apple_native.package_native(_ADMITTED["chol-6"], pipeline_name=PIPELINE)
    finally:
        native.apple_gpu_value_lowering_available.cache_clear()


# ---------------------------------------------------------------------------
# What the retired route admitted and the native contract refuses, by name
# ---------------------------------------------------------------------------

_NARROWED = {
    # The Metal kernels write the lower factor whatever was asked: the retired
    # route returned L for an upper-factor request and spread `lower = False`
    # into a value call the executor ignored.
    "upper-cholesky": _cholesky(4, 4, kwargs={"lower": False}),
    "upper-cholsolve": _solve("tessera.cholesky_solve", (4, 4), (4, 2), kwargs={"lower": False}),
    # Shapes the Graph verifier rejects, or the kernels cannot compute; the
    # retired route admitted them and failed (or fell back) at launch.
    "rank4-cholesky": _cholesky(2, 2, 3, 3),
    "nonsquare-cholesky": _cholesky(4, 3),
    "vector-rhs": _module("tessera.tri_solve", [(4, 4), (4,)], [("out", (4,))]),
    "tri-rhs-mismatch": _solve("tessera.tri_solve", (4, 4), (5, 3)),
    "tri-batch-mismatch": _solve("tessera.tri_solve", (2, 4, 4), (3, 4, 2)),
    "cholsolve-result-shape": _module("tessera.cholesky_solve", [(4, 4), (4, 2)], [("out", (4, 3))]),
    # Keywords the delegate does not compute are refused, not ignored.
    "policy-keyword": _cholesky(4, 4, kwargs={"numeric_policy": "fp32"}),
    "string-flag": _solve("tessera.tri_solve", (4, 4), (4, 2), kwargs={"lower": "yes"}),
}


@pytest.mark.parametrize("module", list(_NARROWED.values()), ids=list(_NARROWED))
def test_native_contract_refuses_what_the_retired_route_admitted(module):
    assert _retired_admits(module)
    assert not native.supports_scheduled_apple_gpu_kernel(module)
    assert apple_native.native_package_kind(module) is None
    with pytest.raises(ValueError):
        native.admit(module)


@pytest.mark.parametrize("module", [
    _svd(4, 6), _svd(6, 4, kwargs={"full_matrices": True}),
    _module("tessera.svd", [(6, 4)], [("u", (6, 6)), ("s", (4,)), ("vh", (4, 4))]),
    _cholesky(4, 4, dtype="fp16"),
], ids=["wide-svd", "full-svd", "svd-wrong-shape", "f16-cholesky"])
def test_neither_route_admits(module):
    assert not _retired_admits(module)
    assert not native.supports_scheduled_apple_gpu_kernel(module)
    assert apple_native.native_package_kind(module) is None


def test_admission_names_apple_gpu_only():
    module = _cholesky(4, 4)
    assert native.supports_scheduled_apple_gpu_kernel(module, target="apple_gpu")
    for target in ("apple_cpu", "x86", "rocm_gfx1151"):
        assert not native.supports_scheduled_apple_gpu_kernel(module, target=target)
    with pytest.raises(ValueError, match="apple_gpu only"):
        native.lower_scheduled_apple_gpu_kernel(module, target="apple_cpu")


def test_retired_value_symbols_left_the_generic_branch():
    """The generic Graph-owned value branch can no longer package these."""
    for op in native.APPLE_GPU_KERNEL_OPS:
        assert op not in apple_native._VALUE_SYMBOLS
    assert not hasattr(apple_native, "package_svd")


# ---------------------------------------------------------------------------
# Tampering fails closed before the runtime is touched
# ---------------------------------------------------------------------------


def _no_runtime(monkeypatch):
    monkeypatch.setattr(apple_native, "_runtime_library_path",
                        lambda: pytest.fail("runtime touched before rejection"))


def _package(artifact):
    return apple_native.package_scheduled_apple_gpu_kernel(artifact, pipeline_name=PIPELINE)


@_needs_compiler
@pytest.mark.parametrize("field,value", [
    ("kind", "cholesky"), ("symbol", "tessera_apple_gpu_cholesky_f32"), ("graph_op", "tessera.cholesky"),
    ("input_names", ("a1", "a0")), ("output_names", ("other",)),
    ("shapes", ((5, 5), (5, 4), (5, 4))),
    ("semantics", {"lower": True, "trans": False, "unit_diag": False}),
    ("schedule_digest", "0" * 64), ("architecture", "apple7"), ("target", "apple_cpu"),
])
def test_descriptor_field_tampering_fails_closed(field, value, monkeypatch):
    artifact = native.lower_scheduled_apple_gpu_kernel(
        _solve("tessera.tri_solve", (5, 5), (5, 3), kwargs={"lower": False}))
    _no_runtime(monkeypatch)
    with pytest.raises(ValueError):
        _package(replace(artifact, **{field: value}))


@_needs_compiler
@pytest.mark.parametrize("stage,old,new", [
    ("graph_ir", "tensor<5x3xf32>", "tensor<5x4xf32>"),
    ("graph_ir", "lower = false", "lower = true"),
    ("schedule_ir", "semantics = {lower = false,", "semantics = {lower = true,"),
    ("schedule_ir", "K = 3 : i64", "K = 4 : i64"),
    ("schedule_ir", 'symbol = "tessera_apple_gpu_tri_solve_f32"', 'symbol = "tessera_apple_gpu_cholesky_f32"'),
    ("tile_ir", "lower = false, source", "lower = true, source"),
    ("tile_ir", 'abi = "tessera.apple.value.tri_solve.f32.v1"', 'abi = "tessera.apple.value.cholesky.f32.v1"'),
    ("target_ir", 'symbol = "tessera_apple_gpu_tri_solve_f32"', 'symbol = "tessera_apple_gpu_cholesky_f32"'),
    ("target_ir", "lower = false", "lower = true"),
])
def test_stage_tampering_fails_closed(stage, old, new, monkeypatch):
    artifact = native.lower_scheduled_apple_gpu_kernel(
        _solve("tessera.tri_solve", (5, 5), (5, 3), kwargs={"lower": False}))
    text = getattr(artifact, stage)
    assert old in text, (stage, old)
    _no_runtime(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        _package(replace(artifact, **{stage: text.replace(old, new)}))


@_needs_compiler
@pytest.mark.parametrize("stage,old,new", [
    ("schedule_ir", "M = 6 : i64", "M = 8 : i64"),
    ("tile_ir", 'symbol = "tessera_apple_gpu_svd_reduced_f32"', 'symbol = "tessera_apple_gpu_svd_f32"'),
    ("target_ir", 'result_order = "u,s,vh"', 'result_order = "u,s,v"'),
])
def test_svd_stage_tampering_fails_closed(stage, old, new, monkeypatch):
    artifact = native.lower_scheduled_apple_gpu_kernel(_svd(6, 4))
    text = getattr(artifact, stage)
    assert old in text, (stage, old)
    _no_runtime(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        _package(replace(artifact, **{stage: text.replace(old, new)}))


@_needs_compiler
def test_schedule_to_tile_refuses_an_altered_contract():
    """A Schedule record edited without its hash is refused by the native
    Schedule -> Tile boundary itself, not only by the Python replay."""
    artifact = native.lower_scheduled_apple_gpu_kernel(_solve("tessera.tri_solve", (5, 5), (5, 3)))
    tampered = artifact.schedule_ir.replace("K = 3 : i64", "K = 4 : i64")
    assert tampered != artifact.schedule_ir
    with pytest.raises(RuntimeError, match="contract was altered"):
        run_tessera_opt(find_tessera_opt(), tampered, "--tessera-schedule-to-tile")


@_needs_compiler
def test_value_lowering_refuses_a_contract_naming_another_symbol():
    """The lowering owns the symbol choice: a Tile contract that names another
    entry is refused at Tile -> Target, not resolved in either direction."""
    artifact = native.lower_scheduled_apple_gpu_kernel(_svd(6, 4))
    tampered = artifact.tile_ir.replace('symbol = "tessera_apple_gpu_svd_reduced_f32"',
                                        'symbol = "tessera_apple_gpu_svd_f32"')
    assert tampered != artifact.tile_ir
    with pytest.raises(RuntimeError, match="lowering selects"):
        run_tessera_opt(find_tessera_opt(), tampered, "--tile-to-apple_gpu=value=true")


# ---------------------------------------------------------------------------
# Native compiler owns the admission; the tables agree
# ---------------------------------------------------------------------------


def _graph_text(body: str, args: str, results: str, *, bindings: str, arch: str = "apple_gpu") -> str:
    return (
        f'module attributes {{tessera.arch = "{arch}", tessera.launch_bindings = [{bindings}], '
        f'tessera.target = "apple_gpu"}} {{\n'
        f"  func.func @f({args}) -> {results} {{\n    {body}\n  }}\n}}\n"
    )


_CHOL = ('%y = tessera.cholesky %a{attrs} : (tensor<4x4xf32>) -> tensor<4x4xf32>\n'
         "    return %y : tensor<4x4xf32>")


@_needs_compiler
@pytest.mark.parametrize("text", [
    _graph_text(_CHOL.format(attrs=" {lower = false}"), "%a: tensor<4x4xf32>", "tensor<4x4xf32>",
                bindings='"a", "y"'),
    _graph_text(_CHOL.format(attrs=""), "%a: tensor<4x4xf32>", "tensor<4x4xf32>", bindings='"a", "a"'),
    _graph_text(_CHOL.format(attrs=""), "%a: tensor<4x4xf32>", "tensor<4x4xf32>",
                bindings='"a", "y"', arch="apple7"),
    _graph_text('%y = tessera.cholesky %a : (tensor<4x4xf32>) -> tensor<4x4xf32>\n'
                '    %z = tessera.cholesky %y : (tensor<4x4xf32>) -> tensor<4x4xf32>\n'
                "    return %z : tensor<4x4xf32>", "%a: tensor<4x4xf32>", "tensor<4x4xf32>",
                bindings='"a", "z"'),
    _graph_text('%u, %s, %v = tessera.svd %a : (tensor<4x6xf32>) -> (tensor<4x4xf32>, tensor<4xf32>, '
                'tensor<4x6xf32>)\n    return %u, %s, %v : tensor<4x4xf32>, tensor<4xf32>, tensor<4x6xf32>',
                "%a: tensor<4x6xf32>", "(tensor<4x4xf32>, tensor<4xf32>, tensor<4x6xf32>)",
                bindings='"a", "u", "s", "v"'),
], ids=["upper", "aliased-bindings", "wrong-arch", "two-ops", "wide-svd"])
def test_native_graph_to_schedule_refuses_outside_the_envelope(text):
    with pytest.raises(RuntimeError, match="Apple GPU native kernel"):
        run_tessera_opt(find_tessera_opt(), text, "--tessera-graph-to-schedule")


def test_contract_tables_agree_across_python_and_cpp():
    header = (ROOT / "src/compiler/programming_model/lib/NativeAppleGPUKernel.h").read_text()
    cpp = {op: (kind, int(inputs), int(results)) for op, kind, inputs, results in re.findall(
        r'\{"(tessera\.\w+)", "(\w+)", (\d), (\d)\}', header)}
    assert cpp == native.APPLE_GPU_KERNEL_OPS
    # The header records, and the value lowering independently selects, the
    # same (symbol, ABI) per (kind, batched).
    lowering = (ROOT / "src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Lowering/"
                "TileToApple.cpp").read_text()
    table = {(kind, batched == "true"): (symbol, abi) for kind, batched, symbol, abi in re.findall(
        r'\{"(\w+)", (true|false), "(\w+)",\s*"([\w.]+)"\}', lowering)}
    assert table == native.APPLE_GPU_KERNEL_ENTRIES
    for (kind, _), (symbol, abi) in native.APPLE_GPU_KERNEL_ENTRIES.items():
        assert f'"{symbol}"' in header and f'"{abi}"' in header, kind
    # The retired descriptor's constants are the ones the route selects.
    assert native.APPLE_GPU_KERNEL_ENTRIES[("svd", False)] == (
        apple_native.APPLE_SVD_REDUCED_F32_SYMBOL, apple_native.APPLE_SVD_REDUCED_F32_ABI)
    assert native.APPLE_GPU_KERNEL_ENTRIES[("svd", True)] == (
        apple_native.APPLE_SVD_REDUCED_BATCHED_F32_SYMBOL, apple_native.APPLE_SVD_REDUCED_BATCHED_F32_ABI)


# ---------------------------------------------------------------------------
# Census: the prune dashboard sees the routes
# ---------------------------------------------------------------------------


def test_prune_census_sees_the_compiled_routes():
    from tessera.compiler import bootstrap_prune_audit

    rows = {(t, f): (route, status) for t, f, route, status in bootstrap_prune_audit.family_rows()}
    for family in ("svd", "value_cholesky", "value_cholesky_solve", "value_tri_solve"):
        assert rows[("apple_gpu", family)] == (
            "scheduled_apple_gpu_kernel.supports_scheduled_apple_gpu_kernel", "compiled"), family


# ---------------------------------------------------------------------------
# Device differential (M1 Max, Metal)
# ---------------------------------------------------------------------------


def _inputs(name: str, module: GraphIRModule) -> list[np.ndarray]:
    rng = np.random.default_rng(1104 + len(name))
    shapes = [tuple(int(d) for d in a.ir_type.shape) for a in module.functions[0].args]
    kind = module.functions[0].body[0].op_name.removeprefix("tessera.")
    values = [rng.standard_normal(shape).astype(np.float32) for shape in shapes]
    n = shapes[0][-1]
    if kind in {"cholesky", "cholesky_solve"}:
        x = values[0].astype(np.float64)
        values[0] = (x @ np.swapaxes(x, -1, -2) + n * np.eye(n)).astype(np.float32)
    elif kind == "tri_solve":
        values[0] = (values[0] + 2 * n * np.eye(n, dtype=np.float32)).astype(np.float32)
    return values


def _run(package, inputs):
    from tessera.runtime import RuntimeArtifact, launch

    artifact = RuntimeArtifact(
        metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
        target_ir=package.target_ir, native_image=package.image, launch_descriptor=package.descriptor,
    )
    shapes: dict[str, list[int]] = {g.binding: [] for g in package.descriptor.shape_guards}
    for guard in package.descriptor.shape_guards:
        shapes[guard.binding].append(guard.value)
    buffers = {binding.name: (inputs[binding.ordinal] if binding.direction == "input"
                              else np.full(shapes[binding.name], np.nan, np.float32))
               for binding in package.descriptor.buffers}
    request: dict = {"buffers": buffers}
    if package.descriptor.scalars:
        request["scalars"] = {s.name: _scalar(s.name, inputs[0]) for s in package.descriptor.scalars}
    launched = launch(artifact, request)
    assert launched["ok"], launched
    output = launched["output"]
    return [np.array(value, copy=True) for value in (output if isinstance(output, tuple | list) else (output,))]


def _scalar(name: str, a: np.ndarray) -> int:
    return {"Batch": a.shape[0] if a.ndim == 3 else 1, "M": a.shape[-2], "N": a.shape[-1]}[name]


def _reference_check(kind: str, module: GraphIRModule, inputs, outputs) -> None:
    flags = {**_DEFAULTS, **module.functions[0].body[0].kwargs}
    a = inputs[0].astype(np.float64)
    tolerance = 2e-4 * max(a.shape[-1], 1) * max(np.abs(a).max(), 1.0)
    if kind == "cholesky":
        lower = outputs[0].astype(np.float64)
        assert np.array_equal(lower, np.tril(lower))
        np.testing.assert_allclose(lower @ np.swapaxes(lower, -1, -2), a, rtol=0, atol=tolerance)
    elif kind == "tri_solve":
        t = np.tril(a) if flags["lower"] else np.triu(a)
        if flags["unit_diag"]:
            n = a.shape[-1]
            t = t * (1 - np.eye(n)) + np.eye(n)
        t = np.swapaxes(t, -1, -2) if flags["trans"] else t
        np.testing.assert_allclose(t @ outputs[0].astype(np.float64), inputs[1], rtol=0, atol=tolerance)
    elif kind == "cholesky_solve":
        np.testing.assert_allclose(a @ outputs[0].astype(np.float64), inputs[1], rtol=0, atol=tolerance)
    else:
        u, s, vh = (value.astype(np.float64) for value in outputs)
        assert np.all(s[..., :-1] >= s[..., 1:])
        np.testing.assert_allclose((u * s[..., None, :]) @ vh, a, rtol=0, atol=tolerance)
        np.testing.assert_allclose(s, np.linalg.svd(a, compute_uv=False), rtol=0, atol=tolerance)


@pytest.mark.hardware_apple_gpu
@_needs_compiler
@pytest.mark.parametrize("name", list(_ADMITTED))
def test_compiled_and_retired_images_agree_bit_for_bit_on_metal(name):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    # On an Apple chip the runtime dylib must build; its absence is a failure.
    assert apple_native.tools_available()
    module = _ADMITTED[name]
    inputs = _inputs(name, module)
    new = _run(apple_native.package_native(module, pipeline_name=PIPELINE), inputs)
    old = _run(baseline.package(module, pipeline_name=PIPELINE), inputs)
    assert len(new) == len(old)
    for fresh, retired in zip(new, old):
        assert fresh.dtype == retired.dtype and fresh.shape == retired.shape
        assert np.array_equal(fresh.view(np.uint32), retired.view(np.uint32))
    _reference_check(module.functions[0].body[0].op_name.removeprefix("tessera."), module, inputs, new)
