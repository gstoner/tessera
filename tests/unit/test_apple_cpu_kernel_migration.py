"""E2E-REAL-6, Apple CPU batched GEMM + LAPACK factorizations: retired
Graph-owned packaging vs the compiled route.

``apple_cpu_native.package_native`` used to read the Python Graph object and
build the Accelerate/LAPACK launch descriptor for ``batched_gemm``,
``tri_solve``, ``cholesky``, ``cholesky_solve``, ``lu``, ``qr`` and ``svd``.
They now lower through the native Graph -> Schedule -> Tile boundaries
(``NativeAppleCPUKernel.h``) and the Apple CPU value lowering, and package the
replayed artifact, reaching the C ABI through a verified
``tessera_apple.cpu.call``. The frozen constructor in
``tests/_support/apple_cpu_baseline.py`` is the declared oracle (Decision
#31(a)); these tests are its differential.

Host-free half: every descriptor/ABI field the old route produced is
reproduced over the envelope both admit, the envelope the old route admitted
and computed wrongly (or not at all) is now refused by name, and a tampered
artifact fails closed before the runtime is touched. Device half (an Apple
chip with the real Accelerate runtime dylib, the M1 Max): both images run on
the same inputs and must agree bit-for-bit, and with a float64 oracle.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tessera.compiler import apple_cpu_native, scheduled_apple_cpu_kernel as native
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt
from tests._support import apple_cpu_baseline as baseline

PIPELINE = "tessera-lower-to-apple_cpu"
ROOT = Path(__file__).resolve().parents[2]
_ELEMENT = {"fp32": "f32", "fp16": "f16", "int32": "i32"}
_DEFAULTS = {"lower": True, "trans": False, "unit_diag": False, "full_matrices": False}


def _value_lowering_registered() -> bool:
    tool = find_tessera_opt()
    if tool is None:
        return False
    probe = ('module {\n  func.func @f(%a: tensor<2x2xf32>) -> tensor<2x2xf32> {\n'
             '    %0 = tile.cholesky %a {source = "tessera.cholesky"} : (tensor<2x2xf32>) -> tensor<2x2xf32>\n'
             '    return %0 : tensor<2x2xf32>\n  }\n}\n')
    try:
        return "tessera_apple.cpu.call" in run_tessera_opt(tool, probe, "--tile-to-apple_cpu=value=true")
    except RuntimeError:
        return False


#: The Target IR is produced by the Apple CPU value lowering, which exists only
#: where the production compiler was configured with the Apple backend.
_needs_compiler = pytest.mark.skipif(
    not _value_lowering_registered(),
    reason="requires production tessera-opt with the Apple CPU value lowering",
)


def _type(shape, dtype: str = "fp32") -> IRType:
    return IRType(f"tensor<{'x'.join(map(str, shape))}x{_ELEMENT[dtype]}>", tuple(map(str, shape)), dtype)


def _module(op_name, inputs, results, *, dtype="fp32", kwargs=None) -> GraphIRModule:
    """One static Graph call; ``results`` is ``[(name, shape, dtype), ...]``."""
    args = [IRArg(f"a{i}", _type(shape, dtype)) for i, shape in enumerate(inputs)]
    result_types = [_type(shape, result_dtype) for _, shape, result_dtype in results]
    names = [name for name, _, _ in results]
    return GraphIRModule(functions=[GraphIRFunction(
        name="apple_cpu_kernel", args=args, result_types=result_types,
        body=[IROp(result=",".join(names), op_name=op_name, operands=[f"%{a.name}" for a in args],
                   operand_types=[str(a.ir_type) for a in args],
                   result_type=(str(result_types[0]) if len(result_types) == 1
                                else "(" + ", ".join(map(str, result_types)) + ")"),
                   kwargs=kwargs or {})],
        return_values=[f"%{name}" for name in names],
    )])


def _batched(b, m, k, n, **kw):
    return _module("tessera.batched_gemm", [(b, m, k), (b, k, n)], [("out", (b, m, n), "fp32")], **kw)


def _solve(op, n, k, **kw):
    return _module(op, [(n, n), (n, k)], [("out", (n, k), "fp32")], **kw)


def _cholesky(n, **kw):
    return _module("tessera.cholesky", [(n, n)], [("out", (n, n), "fp32")], **kw)


def _lu(n):
    return _module("tessera.lu", [(n, n)], [("lu", (n, n), "fp32"), ("pivots", (n,), "int32")])


def _qr(m, n, **kw):
    return _module("tessera.qr", [(m, n)], [("q", (m, n), "fp32"), ("r", (n, n), "fp32")], **kw)


def _svd(m, n, **kw):
    r = min(m, n)
    return _module("tessera.svd", [(m, n)],
                   [("u", (m, r), "fp32"), ("s", (r,), "fp32"), ("v", (r, n), "fp32")], **kw)


_ADMITTED = {
    "batched-1": _batched(1, 1, 1, 1), "batched-2x7x9x5": _batched(2, 7, 9, 5),
    "batched-3x16x8x33": _batched(3, 16, 8, 33),
    "tri-default": _solve("tessera.tri_solve", 5, 3),
    "tri-upper": _solve("tessera.tri_solve", 5, 3, kwargs={"lower": False}),
    "tri-trans": _solve("tessera.tri_solve", 5, 2, kwargs={"trans": True}),
    "tri-unit": _solve("tessera.tri_solve", 4, 4, kwargs={"unit_diag": True}),
    "tri-upper-trans-unit": _solve("tessera.tri_solve", 6, 1,
                                   kwargs={"lower": False, "trans": True, "unit_diag": True}),
    "chol-1": _cholesky(1), "chol-6": _cholesky(6), "chol-lower": _cholesky(6, kwargs={"lower": True}),
    "cholsolve-default": _solve("tessera.cholesky_solve", 5, 3),
    "cholsolve-upper": _solve("tessera.cholesky_solve", 5, 2, kwargs={"lower": False}),
    "lu-1": _lu(1), "lu-5": _lu(5),
    "qr-5x3": _qr(5, 3), "qr-4x4": _qr(4, 4), "qr-reduced": _qr(6, 2, kwargs={"full_matrices": False}),
    "svd-5x3": _svd(5, 3), "svd-3x5": _svd(3, 5), "svd-4x4": _svd(4, 4),
}


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    library = tmp_path / "libtessera_apple_cpu_runtime.dylib"
    library.write_bytes(b"apple-cpu-kernel-migration-fixture")
    monkeypatch.setattr(apple_cpu_native, "_runtime_library_path", lambda: library)
    return library


# ---------------------------------------------------------------------------
# Differential: the compiled route reproduces the retired descriptor
# ---------------------------------------------------------------------------


@_needs_compiler
@pytest.mark.parametrize("module", list(_ADMITTED.values()), ids=list(_ADMITTED))
def test_compiled_route_reproduces_retired_descriptor(module, fake_runtime):
    assert baseline.retired_contract(module)
    old = baseline.package(module, pipeline_name=PIPELINE)
    new = apple_cpu_native.package_native(module, pipeline_name=PIPELINE)
    for field in ("entry_symbol", "abi_id", "shape_guards", "scalars", "geometry", "ordering"):
        assert getattr(new.descriptor, field) == getattr(old.descriptor, field), field
    # Buffers agree field for field, except one named correction: the retired
    # constructor gave every non-fp32 buffer alignment 2, which is wrong for
    # LU's int32 pivots. The native contract binds them at their natural 4.
    assert len(new.descriptor.buffers) == len(old.descriptor.buffers)
    for fresh, retired in zip(new.descriptor.buffers, old.descriptor.buffers):
        expected = replace(retired, alignment=4) if retired.dtype == "int32" else retired
        assert fresh == expected
    for key in ("route", "op_kind", "result_count"):
        assert new.descriptor.provenance[key] == old.descriptor.provenance[key], key
    old_call, new_call = old.descriptor.provenance["value_call"], new.descriptor.provenance["value_call"]
    for key in ("op", "op_kind", "symbol", "status"):
        assert new_call[key] == old_call[key], key
    # The retired call spread whatever keywords the Graph op carried; the new
    # one materializes every flag the delegate reads, so a default is written
    # rather than assumed. They agree on the value the runtime reads.
    for flag in set(_DEFAULTS) & (set(old_call) | set(new_call)):
        assert new_call[flag] == old_call.get(flag, _DEFAULTS[flag]), flag
    assert new.image.entry_points == old.image.entry_points
    assert new.image.payload == old.image.payload
    assert (new.image.target, new.image.architecture, new.image.binary_format) == (
        old.image.target, old.image.architecture, old.image.binary_format)
    # The new route is the scheduled one, and says so.
    provenance = new.descriptor.provenance
    kind = provenance["op_kind"]
    assert provenance["work_item"] == "E2E-REAL-6"
    assert provenance["target_ir_op"] == "tessera_apple.cpu.call"
    digest = provenance["schedule_digest"]
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert f"tessera.{kind} " not in new.tile_ir and f"tile.{kind} " in new.tile_ir
    assert new.target_ir.count("tessera_apple.cpu.call") == 1
    assert f'tessera.schedule_hash = "{digest}"' in new.target_ir
    assert f'symbol = "{new.descriptor.entry_symbol}"' in new.target_ir


@_needs_compiler
@pytest.mark.parametrize("kind", ["batched_gemm", "tri_solve", "lu", "svd"])
def test_driver_routes_the_families_through_the_scheduled_package(kind, fake_runtime, monkeypatch):
    from tessera.compiler.driver import compile_graph_module

    monkeypatch.setenv("TESSERA_APPLE_CPU_RUNTIME_LIB", str(fake_runtime))
    module = {"batched_gemm": _batched(2, 7, 9, 5), "tri_solve": _solve("tessera.tri_solve", 5, 3),
              "lu": _lu(5), "svd": _svd(5, 3)}[kind]
    bundle = compile_graph_module(
        module, source_origin="apple-cpu-kernel-migration-driver", target="apple_cpu",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.abi_id == f"tessera.apple.cpu.value.{kind}.fp32.v1"
    assert bundle.launch_descriptor.provenance["work_item"] == "E2E-REAL-6"
    assert bundle.launch_descriptor.provenance["schedule_digest"]


# ---------------------------------------------------------------------------
# What the retired route admitted and the native contract refuses, by name
# ---------------------------------------------------------------------------

_NARROWED = {
    # spotrf on the row-major buffer writes the lower factor whatever was
    # asked: the retired route returned L for an upper-factor request.
    "upper-cholesky": _cholesky(4, kwargs={"lower": False}),
    # Both ABIs compute the reduced factorization only.
    "full-qr": _qr(4, 4, kwargs={"full_matrices": True}),
    "full-svd": _svd(5, 3, kwargs={"full_matrices": True}),
    # The runtime refused these at launch; now the compiler does.
    "wide-qr": _module("tessera.qr", [(3, 5)], [("q", (3, 5), "fp32"), ("r", (5, 5), "fp32")]),
    "batched-cholesky": _module("tessera.cholesky", [(2, 4, 4)], [("out", (2, 4, 4), "fp32")]),
    "rank2-batched": _module("tessera.batched_gemm", [(4, 5), (5, 3)], [("out", (4, 3), "fp32")]),
    "broadcast-batched": _module("tessera.batched_gemm", [(2, 4, 5), (1, 5, 3)], [("out", (2, 4, 3), "fp32")]),
    "tri-rhs-mismatch": _module("tessera.tri_solve", [(4, 4), (5, 3)], [("out", (5, 3), "fp32")]),
    "lu-wrong-pivots": _module("tessera.lu", [(4, 4)], [("lu", (4, 4), "fp32"), ("pivots", (3,), "int32")]),
    "svd-wrong-shape": _module("tessera.svd", [(5, 3)],
                               [("u", (5, 5), "fp32"), ("s", (3,), "fp32"), ("v", (3, 3), "fp32")]),
    # Keywords the delegate does not compute are refused, not ignored.
    "policy-keyword": _cholesky(4, kwargs={"numeric_policy": "fp32"}),
    "string-flag": _solve("tessera.tri_solve", 4, 2, kwargs={"lower": "yes"}),
}


@pytest.mark.parametrize("module", list(_NARROWED.values()), ids=list(_NARROWED))
def test_native_contract_refuses_what_the_retired_route_admitted(module):
    assert baseline.retired_contract(module)
    assert not native.supports_scheduled_apple_cpu_kernel(module)
    assert apple_cpu_native.native_package_kind(module) is None
    with pytest.raises(ValueError):
        native.admit(module)


@pytest.mark.parametrize("module", [
    _batched(2, 7, 9, 5, dtype="fp16"),
    _module("tessera.batched_gemm", [(2, 7, 9), (2, 9, 5)], [("out", (2, 7, 5), "fp16")]),
], ids=["f16-batched", "mixed-output"])
def test_neither_route_admits_non_f32_storage(module):
    assert not baseline.retired_contract(module)
    assert not native.supports_scheduled_apple_cpu_kernel(module)


def test_admission_names_apple_cpu_only():
    module = _batched(2, 7, 9, 5)
    assert native.supports_scheduled_apple_cpu_kernel(module, target="apple_cpu")
    for target in ("apple_gpu", "x86", "rocm_gfx1151"):
        assert not native.supports_scheduled_apple_cpu_kernel(module, target=target)
    with pytest.raises(ValueError, match="apple_cpu only"):
        native.lower_scheduled_apple_cpu_kernel(module, target="apple_gpu")


def test_only_bnns_matmul_still_packages_from_graph_ir():
    """Every family classified on Apple CPU is scheduled except BNNS f16/bf16."""
    f16 = _module("tessera.matmul", [(7, 9), (9, 5)], [("out", (7, 5), "fp16")], dtype="fp16")
    assert apple_cpu_native.native_package_kind(f16) == "matmul"
    assert apple_cpu_native.scheduled_package_kind(f16) is None
    for name, module in _ADMITTED.items():
        assert apple_cpu_native.scheduled_package_kind(module) is not None, name


# ---------------------------------------------------------------------------
# Tampering fails closed before the runtime is touched
# ---------------------------------------------------------------------------


def _no_runtime(monkeypatch):
    monkeypatch.setattr(apple_cpu_native, "_runtime_library_path",
                        lambda: pytest.fail("runtime touched before rejection"))


def _package(artifact):
    return apple_cpu_native.package_scheduled_apple_cpu_kernel(artifact, pipeline_name=PIPELINE)


@_needs_compiler
@pytest.mark.parametrize("field,value", [
    ("kind", "cholesky"), ("symbol", "tessera_apple_cpu_cholesky_f32"), ("graph_op", "tessera.cholesky"),
    ("input_names", ("a1", "a0")), ("output_names", ("other",)),
    ("shapes", ((5, 5), (5, 4), (5, 4))), ("dtypes", ("fp32", "fp32", "int32")),
    ("semantics", {"lower": False, "trans": False, "unit_diag": False}),
    ("schedule_digest", "0" * 64), ("architecture", "apple_m2"), ("target", "apple_gpu"),
])
def test_descriptor_field_tampering_fails_closed(field, value, monkeypatch):
    artifact = native.lower_scheduled_apple_cpu_kernel(_solve("tessera.tri_solve", 5, 3))
    _no_runtime(monkeypatch)
    with pytest.raises(ValueError):
        _package(replace(artifact, **{field: value}))


@_needs_compiler
@pytest.mark.parametrize("stage,old,new", [
    ("graph_ir", "tensor<5x3xf32>", "tensor<5x4xf32>"),
    ("graph_ir", "lower = false", "lower = true"),
    ("schedule_ir", "semantics = {lower = false,", "semantics = {lower = true,"),
    ("schedule_ir", "K = 3 : i64", "K = 4 : i64"),
    ("schedule_ir", 'symbol = "tessera_apple_cpu_tri_solve_f32"', 'symbol = "tessera_apple_cpu_lu_f32"'),
    ("tile_ir", "lower = false, source", "lower = true, source"),
    ("tile_ir", 'abi = "lapack_strtrs"', 'abi = "lapack_sgetrf"'),
    ("target_ir", 'symbol = "tessera_apple_cpu_tri_solve_f32"', 'symbol = "tessera_apple_cpu_lu_f32"'),
    ("target_ir", "lower = false", "lower = true"),
])
def test_stage_tampering_fails_closed(stage, old, new, monkeypatch):
    artifact = native.lower_scheduled_apple_cpu_kernel(
        _solve("tessera.tri_solve", 5, 3, kwargs={"lower": False}))
    text = getattr(artifact, stage)
    assert old in text, (stage, old)
    _no_runtime(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        _package(replace(artifact, **{stage: text.replace(old, new)}))


@_needs_compiler
def test_schedule_to_tile_refuses_an_altered_contract():
    """A Schedule record edited without its hash is refused by the native
    Schedule -> Tile boundary itself, not only by the Python replay."""
    artifact = native.lower_scheduled_apple_cpu_kernel(_solve("tessera.tri_solve", 5, 3))
    tampered = artifact.schedule_ir.replace("K = 3 : i64", "K = 4 : i64")
    assert tampered != artifact.schedule_ir
    with pytest.raises(RuntimeError, match="contract was altered"):
        run_tessera_opt(find_tessera_opt(), tampered, "--tessera-schedule-to-tile")


# ---------------------------------------------------------------------------
# Native compiler owns the admission; the tables agree
# ---------------------------------------------------------------------------


def _graph_text(body: str, args: str, results: str, *, bindings: str, arch: str = "apple_silicon_cpu") -> str:
    return (
        f'module attributes {{tessera.arch = "{arch}", tessera.launch_bindings = [{bindings}], '
        f'tessera.target = "apple_cpu"}} {{\n'
        f"  func.func @f({args}) -> {results} {{\n    {body}\n  }}\n}}\n"
    )


@_needs_compiler
@pytest.mark.parametrize("text", [
    _graph_text('%y = tessera.cholesky %a {lower = false} : (tensor<4x4xf32>) -> tensor<4x4xf32>\n'
                "    return %y : tensor<4x4xf32>", "%a: tensor<4x4xf32>", "tensor<4x4xf32>",
                bindings='"a", "y"'),
    _graph_text('%y = tessera.cholesky %a : (tensor<4x4xf32>) -> tensor<4x4xf32>\n'
                "    return %y : tensor<4x4xf32>", "%a: tensor<4x4xf32>", "tensor<4x4xf32>",
                bindings='"a", "a"'),
    _graph_text('%y = tessera.cholesky %a : (tensor<4x4xf32>) -> tensor<4x4xf32>\n'
                "    return %y : tensor<4x4xf32>", "%a: tensor<4x4xf32>", "tensor<4x4xf32>",
                bindings='"a", "y"', arch="apple_m2"),
    _graph_text('%y = tessera.cholesky %a : (tensor<4x4xf32>) -> tensor<4x4xf32>\n'
                '    %z = tessera.cholesky %y : (tensor<4x4xf32>) -> tensor<4x4xf32>\n'
                "    return %z : tensor<4x4xf32>", "%a: tensor<4x4xf32>", "tensor<4x4xf32>",
                bindings='"a", "z"'),
], ids=["upper", "aliased-bindings", "wrong-arch", "two-ops"])
def test_native_graph_to_schedule_refuses_outside_the_envelope(text):
    with pytest.raises(RuntimeError, match="Apple CPU native kernel"):
        run_tessera_opt(find_tessera_opt(), text, "--tessera-graph-to-schedule")


def test_contract_tables_agree_across_python_and_cpp():
    header = (ROOT / "src/compiler/programming_model/lib/NativeAppleCPUKernel.h").read_text()
    cpp = {
        op: (kind, symbol, abi, int(inputs), int(results))
        for op, kind, symbol, abi, inputs, results in re.findall(
            r'\{"(tessera\.\w+)", "(\w+)",\s*"(\w+)",\s*"(\w+)", (\d), (\d)\}', header)
    }
    assert cpp == native.APPLE_CPU_KERNEL_OPS
    # The value lowering picks the Target symbol and routine on its own
    # (TileToApple.cpp kLinalgSpecs); it must name the same entry.
    lowering = (ROOT / "src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Lowering/"
                "TileToApple.cpp").read_text()
    for op, (kind, symbol, abi, _, _) in native.APPLE_CPU_KERNEL_OPS.items():
        if kind == "batched_gemm":
            assert f'"{symbol}",\n                               "{abi}"' in lowering
        else:
            assert re.search(r'\{"' + re.escape(op) + r'", "' + abi + r'",\s*"' + symbol + '"', lowering), op


# ---------------------------------------------------------------------------
# Census: the prune dashboard sees the routes
# ---------------------------------------------------------------------------


def test_prune_census_sees_the_compiled_routes():
    from tessera.compiler import bootstrap_prune_audit

    rows = {(t, f): (route, status) for t, f, route, status in bootstrap_prune_audit.family_rows()}
    for family in ("batched_gemm", "tri_solve", "cholesky", "cholesky_solve", "lu", "qr", "svd"):
        assert rows[("apple_cpu", family)] == (
            "scheduled_apple_cpu_kernel.supports_scheduled_apple_cpu_kernel", "compiled"), family
    # The per-target declaration must not leak to the Apple GPU rows.
    assert rows[("apple_gpu", "batched_gemm")][1] == "gap"
    assert rows[("apple_gpu", "svd")][1] == "gap"


# ---------------------------------------------------------------------------
# Device differential (M1 Max, Accelerate)
# ---------------------------------------------------------------------------


def _inputs(name: str, module: GraphIRModule) -> list[np.ndarray]:
    rng = np.random.default_rng(1104 + len(name))
    shapes = [tuple(int(d) for d in a.ir_type.shape) for a in module.functions[0].args]
    kind = module.functions[0].body[0].op_name.removeprefix("tessera.")
    values = [rng.standard_normal(shape).astype(np.float32) for shape in shapes]
    if kind in {"cholesky", "cholesky_solve"}:
        n = shapes[0][0]
        x = values[0].astype(np.float64)
        values[0] = (x @ x.T + n * np.eye(n)).astype(np.float32)
    elif kind == "tri_solve":
        n = shapes[0][0]
        values[0] = (values[0] + 2 * n * np.eye(n, dtype=np.float32)).astype(np.float32)
    elif kind == "lu":
        n = shapes[0][0]
        values[0] = (values[0] + n * np.eye(n, dtype=np.float32)).astype(np.float32)
    return values


def _run(package, inputs):
    from tessera import runtime

    buffers = {}
    shapes = {g.binding: [] for g in package.descriptor.shape_guards}
    for guard in package.descriptor.shape_guards:
        shapes[guard.binding].append(guard.value)
    for binding in package.descriptor.buffers:
        buffers[binding.name] = (inputs[binding.ordinal] if binding.direction == "input"
                                 else np.zeros(shapes[binding.name], np.dtype(
                                     np.int32 if binding.dtype == "int32" else np.float32)))
    result = runtime._submit_apple_cpu_native(package.image, package.descriptor, buffers, {}, None)
    return [np.array(value, copy=True) for value in (result if isinstance(result, tuple) else (result,))]


def _reference_check(kind: str, module: GraphIRModule, inputs, outputs) -> None:
    op = module.functions[0].body[0]
    flags = {**_DEFAULTS, **op.kwargs}
    a = inputs[0].astype(np.float64)
    tolerance = 1e-4 * max(a.shape)
    if kind == "batched_gemm":
        np.testing.assert_allclose(outputs[0], np.einsum("bmk,bkn->bmn", a, inputs[1].astype(np.float64)),
                                   rtol=0, atol=tolerance)
    elif kind == "cholesky":
        lower = outputs[0].astype(np.float64)
        assert np.array_equal(lower, np.tril(lower))
        np.testing.assert_allclose(lower @ lower.T, a, rtol=0, atol=tolerance * np.abs(a).max())
    elif kind == "tri_solve":
        t = np.tril(a) if flags["lower"] else np.triu(a)
        if flags["unit_diag"]:
            t = t - np.diag(np.diag(t)) + np.eye(len(t))
        t = t.T if flags["trans"] else t
        np.testing.assert_allclose(t @ outputs[0].astype(np.float64), inputs[1], rtol=0, atol=tolerance)
    elif kind == "cholesky_solve":
        np.testing.assert_allclose(a @ outputs[0].astype(np.float64), inputs[1], rtol=0,
                                   atol=tolerance * np.abs(a).max())
    elif kind == "lu":
        packed, pivots = outputs[0].astype(np.float64), outputs[1]
        n = len(a)
        lower, upper = np.tril(packed, -1) + np.eye(n), np.triu(packed)
        permuted = a.copy()
        for row, pivot in enumerate(pivots):  # LAPACK: 1-based sequential swaps
            permuted[[row, pivot - 1]] = permuted[[pivot - 1, row]]
        np.testing.assert_allclose(lower @ upper, permuted, rtol=0, atol=tolerance * n)
    elif kind == "qr":
        q, r = (value.astype(np.float64) for value in outputs)
        np.testing.assert_allclose(q @ r, a, rtol=0, atol=tolerance)
        np.testing.assert_allclose(q.T @ q, np.eye(q.shape[1]), rtol=0, atol=tolerance)
    else:
        u, s, v = (value.astype(np.float64) for value in outputs)
        np.testing.assert_allclose(u @ np.diag(s) @ v, a, rtol=0, atol=tolerance)
        np.testing.assert_allclose(s, np.linalg.svd(a, compute_uv=False), rtol=0, atol=tolerance)


@_needs_compiler
@pytest.mark.parametrize("name", list(_ADMITTED))
def test_compiled_and_retired_images_agree_bit_for_bit_on_accelerate(name):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    # On an Apple chip the runtime dylib must build; its absence is a failure.
    assert apple_cpu_native.tools_available()
    module = _ADMITTED[name]
    inputs = _inputs(name, module)
    new = _run(apple_cpu_native.package_native(module, pipeline_name=PIPELINE), inputs)
    old = _run(baseline.package(module, pipeline_name=PIPELINE), inputs)
    assert len(new) == len(old)
    for fresh, retired in zip(new, old):
        assert fresh.dtype == retired.dtype and fresh.shape == retired.shape
        assert np.array_equal(fresh.view(np.uint32), retired.view(np.uint32))
    _reference_check(module.functions[0].body[0].op_name.removeprefix("tessera."), module, inputs, new)
