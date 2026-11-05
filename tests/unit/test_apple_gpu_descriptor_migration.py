"""E2E-REAL-6, Apple GPU runtime-descriptor families: retired Graph-owned
packaging vs the compiled route.

``apple_native.package_native`` used to read the Python Graph object and build
the MPS / MPSGraph / MSL launch descriptor for ``batched_gemm``,
``transpose`` and the dynamic ``softmax``, ``gelu``, ``popcount``,
``count_nonzero`` and ``top_k``. They now lower through the native Graph ->
Schedule -> Tile boundaries (``NativeAppleGPUDescriptor.h``) and the Apple GPU
value lowering, and package the replayed artifact, reaching the runtime C ABI
through a verified ``tessera_apple.gpu.kernel_call``. The retired
constructors stay in ``apple_native`` as the declared oracle (Decision #31(a));
these tests are its differential.

Host-free half: every descriptor/ABI field the old route produced is
reproduced over the envelope both admit, the envelope the old route admitted
but the compiler cannot (invalid Graph IR) or should not (ignored keywords)
take is refused by name, a tampered artifact fails closed before the runtime
is touched, and a ``tessera-opt`` without the lowering fails closed naming it.
Device half (an Apple chip with the real runtime dylib, the M1 Max): both
descriptors run on the same inputs and must agree bit-for-bit.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tessera.compiler import apple_native
from tessera.compiler import scheduled_apple_gpu_descriptor as native
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, IRType
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt

PIPELINE = "tessera-lower-to-apple_gpu-runtime"
ROOT = Path(__file__).resolve().parents[2]
_ELEMENT = {"fp32": "f32", "fp16": "f16", "bf16": "bf16", "int32": "i32", "int64": "i64"}


def _compiler_registered() -> bool:
    tool = find_tessera_opt()
    return tool is not None and native.apple_gpu_value_lowering_available(str(tool))


#: The Target IR is produced by the Apple GPU value lowering, which exists only
#: where the production compiler was configured with the Apple backend.
_needs_compiler = pytest.mark.skipif(
    not _compiler_registered(),
    reason="requires production tessera-opt with the Apple GPU descriptor lowering",
)


def _type(shape, dtype: str) -> IRType:
    dims = tuple(str(d) for d in shape)
    return IRType(f"tensor<{'x'.join(dims)}x{_ELEMENT[dtype]}>", dims, dtype)


def _module(op_name, inputs, results, *, kwargs=None) -> GraphIRModule:
    """``inputs`` is ``[(shape, dtype)]``; ``results`` is ``[(name, shape, dtype)]``."""
    args = [IRArg(f"a{i}", _type(shape, dtype)) for i, (shape, dtype) in enumerate(inputs)]
    result_types = [_type(shape, dtype) for _, shape, dtype in results]
    names = [name for name, _, _ in results]
    return GraphIRModule(functions=[GraphIRFunction(
        name="apple_gpu_descriptor", args=args, result_types=result_types,
        body=[IROp(result=",".join(names), op_name=op_name, operands=[f"%{a.name}" for a in args],
                   operand_types=[str(a.ir_type) for a in args],
                   result_type=(str(result_types[0]) if len(result_types) == 1
                                else "(" + ", ".join(map(str, result_types)) + ")"),
                   kwargs=kwargs or {})],
        return_values=[f"%{name}" for name in names],
    )])


def _bmm(b, m, k, n, dtype="fp32", *, rhs_batch=None, op="tessera.batched_gemm"):
    return _module(op, [((b, m, k), dtype), ((rhs_batch or b, k, n), dtype)],
                   [("out", (b, m, n), dtype)])


def _transpose(shape, axes, dtype="fp32", **kwargs):
    return _module("tessera.transpose", [(shape, dtype)],
                   [("out", tuple(shape[a] for a in axes), dtype)], kwargs={"axes": axes, **kwargs})


def _softmax(**kwargs):
    return _module("tessera.softmax", [(("?", "?"), "fp32")], [("out", ("?", "?"), "fp32")],
                   kwargs={"axis": -1, **kwargs})


def _gelu(dtype="fp32", **kwargs):
    return _module("tessera.gelu", [(("?", "?"), dtype)], [("out", ("?", "?"), dtype)], kwargs=kwargs)


def _popcount(**kwargs):
    return _module("tessera.popcount", [(("?",), "int32")], [("out", ("?",), "int32")], kwargs=kwargs)


def _count_nonzero(**kwargs):
    return _module("tessera.count_nonzero", [(("?", "?"), "fp32")], [("out", ("?",), "int32")],
                   kwargs={"axis": -1, "keepdims": False, **kwargs})


def _topk(k=3, index="int32", **kwargs):
    return _module("tessera.top_k", [(("?", "?"), "fp32")],
                   [("values", ("?", k), "fp32"), ("indices", ("?", k), index)],
                   kwargs={"k": k, "axis": -1, **kwargs})


_ORACLES = {
    "batched_gemm": apple_native.package_batched_gemm,
    "transpose": apple_native.package_transpose,
    "dynamic_softmax": apple_native.package_dynamic_softmax,
    "dynamic_gelu": apple_native.package_dynamic_gelu,
    "dynamic_popcount": apple_native.package_dynamic_popcount,
    "dynamic_count_nonzero": apple_native.package_dynamic_count_nonzero,
    "dynamic_topk": apple_native.package_dynamic_topk,
}

_ADMITTED = {
    "bmm-f32": _bmm(2, 4, 8, 16), "bmm-f16": _bmm(3, 7, 5, 9, "fp16"),
    "bmm-bf16": _bmm(1, 1, 1, 1, "bf16"),
    "transpose-2d": _transpose((3, 7), (1, 0)), "transpose-3d-f16": _transpose((2, 3, 5), (2, 0, 1), "fp16"),
    "transpose-3d-bf16": _transpose((2, 3, 5), (1, 2, 0), "bf16"),
    "transpose-negative-axes": _transpose((4, 6), (-1, -2)),
    "softmax": _softmax(), "softmax-axis1": _softmax(axis=1),
    "gelu-f32": _gelu(), "gelu-f16": _gelu("fp16"), "gelu-bf16": _gelu("bf16"),
    "popcount": _popcount(),
    "count-nonzero": _count_nonzero(), "count-nonzero-axis1": _count_nonzero(axis=1),
    "topk-3": _topk(3), "topk-1": _topk(1),
}


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    library = tmp_path / "libTesseraAppleRuntime.dylib"
    library.write_bytes(b"apple-gpu-descriptor-migration-fixture")
    monkeypatch.setenv("TESSERA_APPLE_GPU_RUNTIME_LIB", str(library))
    return library


#: Provenance the retired constructors stamped that the compiled route does
#: not: their work items, and the dynamic GELU's Graph-parent/native-pass
#: digests (the replacement records the Schedule and Tile digests instead).
_RETIRED_ONLY = {"work_item", "parent_ir_digest", "native_ir_digest"}


# ---------------------------------------------------------------------------
# Differential: the compiled route reproduces the retired descriptor
# ---------------------------------------------------------------------------


@_needs_compiler
@pytest.mark.parametrize("module", list(_ADMITTED.values()), ids=list(_ADMITTED))
def test_compiled_route_reproduces_retired_descriptor(module, fake_runtime):
    kind = apple_native.native_package_kind(module)
    assert kind in apple_native.SCHEDULED_DESCRIPTOR_KINDS
    new = apple_native.package_native(module, pipeline_name=PIPELINE)
    old = _ORACLES[kind](module, pipeline_name=PIPELINE)
    for field in ("entry_symbol", "abi_id", "buffers", "shape_guards", "scalars", "geometry", "ordering"):
        assert getattr(new.descriptor, field) == getattr(old.descriptor, field), field
    assert new.image.payload == old.image.payload
    assert new.image.entry_points == old.image.entry_points
    for key, value in old.descriptor.provenance.items():
        if key == "broadcast_b" and old.descriptor.provenance["shape"][0] == 1:
            # The retired label tested `B.shape[0] == 1`, so a batch-1 GEMM
            # read as broadcast. Nothing consumes the label and the work is
            # identical; the contract records the no-broadcast form it admits.
            assert new.descriptor.provenance[key] is False
        elif key not in _RETIRED_ONLY:
            assert new.descriptor.provenance[key] == value, key
    provenance = new.descriptor.provenance
    assert provenance["work_item"] == "E2E-REAL-6"
    assert provenance["target_ir_op"] == "tessera_apple.gpu.kernel_call"
    assert re.fullmatch(r"[0-9a-f]{64}", provenance["schedule_digest"])
    # The Target IR is the compiler's, not a Python-written string.
    assert new.target_ir.count("tessera_apple.gpu.kernel_call") == 1
    assert f'tessera.schedule_hash = "{provenance["schedule_digest"]}"' in new.target_ir
    assert new.image.target_ir_digest != old.image.target_ir_digest


@_needs_compiler
@pytest.mark.parametrize("name", ["bmm-f16", "transpose-3d-bf16", "softmax", "topk-3"])
def test_driver_routes_the_families_through_the_scheduled_package(name, fake_runtime):
    from tessera.compiler.driver import compile_graph_module

    bundle = compile_graph_module(
        _ADMITTED[name], source_origin="apple-gpu-descriptor-migration-driver", target="apple_gpu",
        options={"package_native": True}, enable_tool_validation=False,
    )
    assert bundle.launch_descriptor is not None
    assert bundle.launch_descriptor.provenance["work_item"] == "E2E-REAL-6"
    assert bundle.launch_descriptor.provenance["schedule_digest"]
    assert "tessera_apple.gpu.kernel_call" in bundle.target_ir.text


@_needs_compiler
def test_transpose_permutation_keyword_is_honoured(fake_runtime):
    """The Graph emitter accepts ``permutation`` as well as ``axes``. The retired
    constructor read only ``axes``, so a ``permutation`` request failed its
    shape check and was not packaged; the native contract reads the emitted
    attribute and admits it (a widening, not a change to admitted output)."""
    shape, axes = (2, 3, 5), (2, 0, 1)
    module = _module("tessera.transpose", [(shape, "fp32")],
                     [("out", tuple(shape[a] for a in axes), "fp32")], kwargs={"permutation": axes})
    package = apple_native.package_native(module, pipeline_name=PIPELINE)
    assert package.descriptor.provenance["axes"] == list(axes)
    assert apple_native._transpose_contract(module) is None  # the retired reading
    with pytest.raises(ValueError):
        apple_native.package_transpose(module, pipeline_name=PIPELINE)
    disagreeing = _module("tessera.transpose", [(shape, "fp32")],
                          [("out", tuple(shape[a] for a in axes), "fp32")],
                          kwargs={"axes": axes, "permutation": (1, 2, 0)})
    assert not native.supports_scheduled_apple_gpu_descriptor(disagreeing)


def test_static_softmax_and_gelu_stay_on_the_semantic_kernel():
    static_softmax = _module("tessera.softmax", [((3, 7), "fp32")], [("out", (3, 7), "fp32")],
                             kwargs={"axis": -1})
    static_gelu = _module("tessera.gelu", [((5, 11), "fp16")], [("out", (5, 11), "fp16")])
    assert apple_native.native_package_kind(static_softmax) == "softmax"
    assert apple_native.native_package_kind(static_gelu) == "gelu"
    for module in (static_softmax, static_gelu):
        assert not native.owns(module)
        assert not native.supports_scheduled_apple_gpu_descriptor(module)


# ---------------------------------------------------------------------------
# What the retired route admitted and the native contract refuses, by name
# ---------------------------------------------------------------------------

_NARROWED = {
    # Not valid Graph IR: tessera.batched_gemm forbids broadcasting and
    # tessera.matmul/gemm are rank-2, so tessera-opt rejects all three.
    "broadcast-batched": _bmm(2, 4, 5, 3, rhs_batch=1),
    "rank3-matmul": _bmm(2, 4, 5, 3, op="tessera.matmul"),
    "rank3-gemm": _bmm(2, 4, 5, 3, op="tessera.gemm"),
    # Keywords the runtime kernel does not compute are refused, not ignored.
    "softmax-policy": _softmax(numeric_policy="fp32"),
    "gelu-approximate": _gelu(approximate="erf"),
    "transpose-extra": _transpose((3, 7), (1, 0), conjugate=True),
    "topk-largest": _topk(3, largest=False),
    "popcount-policy": _popcount(numeric_policy="int32"),
}


@pytest.mark.parametrize("name", list(_NARROWED))
def test_native_contract_refuses_what_the_retired_route_admitted(name, fake_runtime):
    module = _NARROWED[name]
    # The retired constructor packaged it ...
    kind = {"transpose-extra": "transpose", "softmax-policy": "dynamic_softmax",
            "gelu-approximate": "dynamic_gelu", "topk-largest": "dynamic_topk",
            "popcount-policy": "dynamic_popcount"}.get(name, "batched_gemm")
    if kind != "dynamic_gelu":  # the gelu oracle replays its own native pass
        _ORACLES[kind](module, pipeline_name=PIPELINE)
    # ... and the native contract does not.
    assert not native.supports_scheduled_apple_gpu_descriptor(module)
    assert apple_native.native_package_kind(module) is None
    with pytest.raises(ValueError):
        native.admit(module)


@pytest.mark.parametrize("module", [
    _topk(3, index="int64"), _count_nonzero(keepdims=True), _count_nonzero(axis=0),
    _softmax(axis=0), _bmm(2, 3, 4, 5, "int32"),
    _module("tessera.transpose", [((2, 3, 4, 5), "fp32")], [("out", (5, 4, 3, 2), "fp32")]),
], ids=["topk-i64", "cnz-keepdims", "cnz-axis0", "softmax-axis0", "bmm-int", "transpose-rank4"])
def test_neither_route_admits_outside_the_runtime_abi(module, fake_runtime):
    assert apple_native.native_package_kind(module) is None
    assert not native.supports_scheduled_apple_gpu_descriptor(module)


def test_admission_names_apple_gpu_only():
    module = _ADMITTED["softmax"]
    assert native.supports_scheduled_apple_gpu_descriptor(module, target="apple_gpu")
    for target in ("apple_cpu", "x86", "rocm_gfx1151"):
        assert not native.supports_scheduled_apple_gpu_descriptor(module, target=target)
    with pytest.raises(ValueError, match="apple_gpu only"):
        native.lower_scheduled_apple_gpu_descriptor(module, target="apple_cpu")


# ---------------------------------------------------------------------------
# A host whose tessera-opt cannot lower these families fails closed
# ---------------------------------------------------------------------------


def test_missing_compiler_fails_closed_naming_tessera_opt(monkeypatch, fake_runtime):
    import tessera.compiler.scheduled_matmul as scheduled_matmul

    monkeypatch.setattr(scheduled_matmul, "find_tessera_opt", lambda: None)
    module = _ADMITTED["popcount"]
    assert apple_native.native_package_kind(module) == "dynamic_popcount"
    assert not apple_native.native_package_available(module)
    with pytest.raises(RuntimeError, match="tessera-opt"):
        apple_native.package_native(module, pipeline_name=PIPELINE)


def test_compiler_without_the_lowering_fails_closed(monkeypatch, fake_runtime, tmp_path):
    import tessera.compiler.scheduled_matmul as scheduled_matmul

    # A tessera-opt that parses but cannot lower (here: one that always fails).
    stub = tmp_path / "tessera-opt"
    stub.write_text("#!/bin/sh\necho 'unknown pass' >&2\nexit 1\n")
    stub.chmod(0o755)
    monkeypatch.setattr(scheduled_matmul, "find_tessera_opt", lambda: stub)
    assert not native.apple_gpu_value_lowering_available(str(stub))
    module = _ADMITTED["bmm-f32"]
    assert not apple_native.native_package_available(module)
    with pytest.raises(RuntimeError, match="tessera-opt"):
        apple_native.package_native(module, pipeline_name=PIPELINE)
    # The sibling linalg route (NativeAppleGPUKernel.h) needs the same lowering.
    svd = _module("tessera.svd", [((5, 3), "fp32")],
                  [("u", (5, 3), "fp32"), ("s", (3,), "fp32"), ("vh", (3, 3), "fp32")])
    assert not apple_native.native_package_available(svd)
    # A static softmax packages from the semantic kernel and needs no compiler.
    softmax = _module("tessera.softmax", [((4, 8), "fp32")], [("y", (4, 8), "fp32")])
    assert apple_native.native_package_kind(softmax) == "softmax"
    assert apple_native.native_package_available(softmax)


# ---------------------------------------------------------------------------
# Tampering fails closed before the runtime is touched
# ---------------------------------------------------------------------------


def _no_runtime(monkeypatch):
    monkeypatch.setattr(apple_native, "_runtime_library_path",
                        lambda: pytest.fail("runtime touched before rejection"))


def _package(artifact):
    return apple_native.package_scheduled_apple_gpu_descriptor(artifact, pipeline_name=PIPELINE)


@_needs_compiler
@pytest.mark.parametrize("field,value", [
    ("kind", "dynamic_softmax"), ("graph_op", "tessera.softmax"),
    ("input_names", ("x",)), ("output_names", ("indices", "values")),
    ("shapes", ((-1, -1), (-1, 4), (-1, 4))), ("dtypes", ("fp32", "fp32", "int64")),
    ("semantics", {"axis": -1, "k": 3, "value_order": "ascending", "nan_policy": "last",
                   "tie_policy": "lower_index"}),
    ("schedule_digest", "0" * 64), ("architecture", "apple8"), ("target", "apple_cpu"),
])
def test_descriptor_field_tampering_fails_closed(field, value, monkeypatch):
    artifact = native.lower_scheduled_apple_gpu_descriptor(_topk(3))
    _no_runtime(monkeypatch)
    with pytest.raises(ValueError):
        _package(replace(artifact, **{field: value}))


@_needs_compiler
@pytest.mark.parametrize("stage,old,new", [
    ("graph_ir", "array<i64: 2, 0, 1>", "array<i64: 1, 2, 0>"),
    ("graph_ir", "tensor<5x2x3xf16>", "tensor<5x2x3xf32>"),
    ("schedule_ir", "semantics = {axes = array<i64: 2, 0, 1>}", "semantics = {axes = array<i64: 1, 2, 0>}"),
    ("schedule_ir", 'symbol = "tessera_apple_gpu_mpsgraph_transpose_f16"',
     'symbol = "tessera_apple_gpu_mpsgraph_transpose_f32"'),
    ("tile_ir", "array<i64: 2, 0, 1>", "array<i64: 1, 2, 0>"),
    ("tile_ir", 'abi = "tessera.apple.mpsgraph.transpose.x_o_dims_perm.f16.v1"',
     'abi = "tessera.apple.mpsgraph.transpose.x_o_dims_perm.bf16.v1"'),
    ("target_ir", 'symbol = "tessera_apple_gpu_mpsgraph_transpose_f16"',
     'symbol = "tessera_apple_gpu_mpsgraph_transpose_f32"'),
    ("target_ir", 'framework = "MPSGraph", op_kind', 'framework = "Metal", op_kind'),
])
def test_stage_tampering_fails_closed(stage, old, new, monkeypatch):
    artifact = native.lower_scheduled_apple_gpu_descriptor(_transpose((2, 3, 5), (2, 0, 1), "fp16"))
    text = getattr(artifact, stage)
    assert old in text, (stage, old)
    _no_runtime(monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        _package(replace(artifact, **{stage: text.replace(old, new)}))


@_needs_compiler
def test_schedule_to_tile_refuses_an_altered_contract():
    """A Schedule record edited without its hash is refused by the native
    Schedule -> Tile boundary itself, not only by the Python replay."""
    artifact = native.lower_scheduled_apple_gpu_descriptor(_topk(3))
    tampered = artifact.schedule_ir.replace('nan_policy = "last"', 'nan_policy = "first"')
    assert tampered != artifact.schedule_ir
    with pytest.raises(RuntimeError, match="contract was altered"):
        run_tessera_opt(find_tessera_opt(), tampered, "--tessera-schedule-to-tile")


@_needs_compiler
def test_target_lowering_refuses_a_contract_naming_another_entry():
    """The value lowering derives the runtime entry itself and refuses a Tile
    contract that names a different one."""
    artifact = native.lower_scheduled_apple_gpu_descriptor(_bmm(2, 4, 8, 16, "fp16"))
    tampered = artifact.tile_ir.replace('symbol = "tessera_apple_gpu_bmm_f16"',
                                        'symbol = "tessera_apple_gpu_bmm_f32"')
    assert tampered != artifact.tile_ir
    with pytest.raises(RuntimeError, match="disagrees with the runtime entry"):
        run_tessera_opt(find_tessera_opt(), tampered, "--tile-to-apple_gpu=value=true")


# ---------------------------------------------------------------------------
# Native compiler owns the admission; the tables agree
# ---------------------------------------------------------------------------


def _graph_text(body: str, args: str, results: str, *, bindings: str, arch: str = "apple7") -> str:
    return (
        f'module attributes {{tessera.target = "apple_gpu", tessera.arch = "{arch}", '
        f'tessera.launch_bindings = {bindings}}} {{\n'
        f"  func.func @f({args}) -> {results} {{\n    {body}\n  }}\n}}\n"
    )


@_needs_compiler
@pytest.mark.parametrize("text,message", [
    (_graph_text('%0 = tessera.softmax %x {axis = 0 : i64} : (tensor<?x?xf32>) -> tensor<?x?xf32>\n'
                 '    return %0 : tensor<?x?xf32>', "%x: tensor<?x?xf32>", "tensor<?x?xf32>",
                 bindings='["x", "y"]'), "dynamic softmax requires"),
    (_graph_text('%0 = tessera.gelu %x : (tensor<?x7xf32>) -> tensor<?x7xf32>\n'
                 '    return %0 : tensor<?x7xf32>', "%x: tensor<?x7xf32>", "tensor<?x7xf32>",
                 bindings='["x", "y"]'), "dynamic gelu requires"),
    (_graph_text('%0 = tessera.popcount %x : (tensor<4xi32>) -> tensor<4xi32>\n'
                 '    return %0 : tensor<4xi32>', "%x: tensor<4xi32>", "tensor<4xi32>",
                 bindings='["x", "y"]'), "dynamic popcount requires"),
    (_graph_text('%0 = tessera.transpose %x {permutation = array<i64: 1, 0>, conjugate = true} : '
                 '(tensor<3x7xf32>) -> tensor<7x3xf32>\n    return %0 : tensor<7x3xf32>',
                 "%x: tensor<3x7xf32>", "tensor<7x3xf32>", bindings='["x", "y"]'),
     "unsupported policy attribute 'conjugate'"),
    (_graph_text('%0 = tessera.popcount %x : (tensor<?xi32>) -> tensor<?xi32>\n'
                 '    return %0 : tensor<?xi32>', "%x: tensor<?xi32>", "tensor<?xi32>",
                 bindings='["x", "y"]', arch="apple8"), "requires the apple7 Apple GPU target"),
    (_graph_text('%0 = tessera.popcount %x : (tensor<?xi32>) -> tensor<?xi32>\n'
                 '    return %0 : tensor<?xi32>', "%x: tensor<?xi32>", "tensor<?xi32>",
                 bindings='["x", "x"]'), "binding names must be distinct"),
])
def test_native_graph_to_schedule_refuses_outside_the_envelope(text, message):
    with pytest.raises(RuntimeError, match=re.escape(message)):
        run_tessera_opt(find_tessera_opt(), text, "--tessera-graph-to-schedule")


def test_contract_tables_agree_across_python_and_cpp():
    contract = (ROOT / "src/compiler/programming_model/lib/NativeAppleGPUDescriptor.h").read_text()
    specs = re.findall(r'\{"(tessera\.[a-z_]+)", "([a-z_]+)", "([a-z_]+)", (\d), (\d)\}', contract)
    assert {op: (kind, tile, int(i), int(r)) for op, kind, tile, i, r in specs} == \
        native.APPLE_GPU_DESCRIPTOR_OPS
    lowering = (ROOT / "src/compiler/codegen/Tessera_Apple_Backend/lib/Target/Apple/Lowering/"
                "TileToApple.cpp").read_text()
    entries = re.findall(r'\{"tile\.([a-z_]+)", "([a-z0-9]+)", "(tessera_apple_gpu_[a-z0-9_]+)", '
                         r'"([A-Za-z]+)"\}', lowering)
    storage = {"f32": "fp32", "f16": "fp16", "bf16": "bf16", "i32": "int32"}
    kind_of = {tile: kind for kind, tile, _, _ in native.APPLE_GPU_DESCRIPTOR_OPS.values()}
    assert {(kind_of[tile], storage[s]): (symbol, framework) for tile, s, symbol, framework in entries} == {
        key: (symbol, framework) for key, (symbol, _, framework) in native.APPLE_GPU_DESCRIPTOR_ENTRIES.items()}
    # Every symbol and ABI id is the runtime constant the launcher dispatches on.
    for symbol, abi, _ in native.APPLE_GPU_DESCRIPTOR_ENTRIES.values():
        assert f'"{symbol}"' in contract and f'"{abi}"' in contract
        assert symbol in vars(apple_native).values() and abi in vars(apple_native).values()


def test_prune_census_sees_the_compiled_routes():
    from tessera.compiler import bootstrap_prune_audit

    rows = {(t, f): (route, status) for t, f, route, status in bootstrap_prune_audit.family_rows()}
    for family in apple_native.SCHEDULED_DESCRIPTOR_KINDS:
        assert rows[("apple_gpu", family)] == (
            "scheduled_apple_gpu_descriptor.supports_scheduled_apple_gpu_descriptor", "compiled")


# ---------------------------------------------------------------------------
# Device differential (M1 Max): both descriptors, same inputs, same bits
# ---------------------------------------------------------------------------


def _inputs(name: str, module: GraphIRModule):
    rng = np.random.default_rng(sum(map(ord, name)))
    fn = module.functions[0]
    storage = {"fp32": np.float32, "fp16": np.float16, "int32": np.int32}
    shapes = {"softmax": (5, 11), "softmax-axis1": (3, 7), "gelu-f32": (5, 11), "gelu-f16": (3, 7),
              "gelu-bf16": (4, 9), "popcount": (37,), "count-nonzero": (5, 11),
              "count-nonzero-axis1": (3, 7), "topk-3": (5, 11), "topk-1": (2, 6)}
    arrays, scalars = {}, {}
    for arg in fn.args:
        dtype = arg.ir_type.dtype
        shape = shapes.get(name) or tuple(int(d) for d in arg.ir_type.shape)
        if dtype == "int32":
            value = rng.integers(-2**31, 2**31 - 1, size=shape, dtype=np.int64).astype(np.int32)
        else:
            value = rng.standard_normal(shape).astype(np.float32)
            if name.startswith("count-nonzero"):
                value *= rng.random(shape) > 0.4
            if name == "topk-3":
                value[0, :4] = (5.0, 5.0, np.nan, 5.0)  # exercise the tie and NaN policy
            dtype_np = (__import__("ml_dtypes").bfloat16 if dtype == "bf16" else storage[dtype])
            value = value.astype(dtype_np)
        arrays[arg.name] = value
    x = next(iter(arrays.values()))
    kind = apple_native.native_package_kind(module)
    if kind == "dynamic_softmax":
        scalars = {"Rows": x.shape[0], "Columns": x.shape[1]}
    elif kind in {"dynamic_gelu", "dynamic_popcount"}:
        scalars = {"Elements": x.size}
    elif kind == "dynamic_count_nonzero":
        scalars = {"Outer": x.shape[0], "AxisExtent": x.shape[1]}
    elif kind == "dynamic_topk":
        scalars = {"Rows": x.shape[0], "Columns": x.shape[1], "K": int(fn.body[0].kwargs["k"])}
    return arrays, scalars


def _outputs(module: GraphIRModule, arrays):
    fn = module.functions[0]
    x = next(iter(arrays.values()))
    kind = apple_native.native_package_kind(module)
    out = {}
    for name, ty in zip((v.removeprefix("%") for v in fn.return_values), fn.result_types):
        dims = tuple(ty.shape)
        if kind in {"dynamic_softmax", "dynamic_gelu", "dynamic_popcount"}:
            shape = x.shape
        elif kind == "dynamic_count_nonzero":
            shape = (x.shape[0],)
        elif kind == "dynamic_topk":
            shape = (x.shape[0], int(dims[1]))
        else:
            shape = tuple(int(d) for d in dims)
        out[name] = np.full(shape, -7, dtype=x.dtype if ty.dtype != "int32" else np.int32)
    return out


def _run(package, arrays, scalars, outputs):
    from tessera.runtime import RuntimeArtifact, launch

    artifact = RuntimeArtifact(
        metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
        target_ir=package.target_ir, native_image=package.image, launch_descriptor=package.descriptor,
    )
    buffers = {**arrays, **{k: v.copy() for k, v in outputs.items()}}
    launched = launch(artifact, {"buffers": buffers, "scalars": scalars} if scalars else buffers)
    assert launched["ok"], launched
    assert launched["execution_kind"] == "native_gpu"
    result = launched["output"]
    return list(result) if isinstance(result, tuple) else [result]


@pytest.mark.hardware_apple_gpu
@_needs_compiler
@pytest.mark.parametrize("name", list(_ADMITTED))
def test_compiled_and_retired_descriptors_agree_bit_for_bit_on_metal(name):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    if "bf16" in name:
        pytest.importorskip("ml_dtypes")
    # On an Apple chip the runtime dylib must build; its absence is a failure.
    assert apple_native.tools_available()
    module = _ADMITTED[name]
    kind = apple_native.native_package_kind(module)
    arrays, scalars = _inputs(name, module)
    outputs = _outputs(module, arrays)
    new = _run(apple_native.package_native(module, pipeline_name=PIPELINE), arrays, scalars, outputs)
    old = _run(_ORACLES[kind](module, pipeline_name=PIPELINE), arrays, scalars, outputs)
    assert len(new) == len(old)
    for fresh, retired in zip(new, old):
        assert fresh.dtype == retired.dtype and fresh.shape == retired.shape
        assert fresh.tobytes() == retired.tobytes()
