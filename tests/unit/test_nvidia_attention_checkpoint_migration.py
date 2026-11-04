"""E2E-REAL-6, SM120 attention: retired Graph-authored packaging vs the Tile projection.

Sync ``E2E-REAL-6-sm120-attention-2026-11-03``. The four SM120 attention
families (``attention_lse``, ``attention_scaled_kv``,
``attention_backward_lse``, ``attention_backward``) are packaged by typed
consumers of replayed artifacts:
``nvidia_native.package_scheduled_checkpoint`` and
``package_scheduled_attention_recompute``. The frozen packagers in
``tests/_support/nvidia_attention_checkpoint_baseline.py`` are the declared
Decision #31(a) oracle; these tests are its differential.

Host-free half: projected descriptors equal the retired Graph-authored ones
on every launch field, the consumers never re-enter Graph admission, and a
tampered artifact fails before target compilation. The device half lives in
``tests/device/nvidia/test_attention_checkpoint_migration.py``.
"""

from __future__ import annotations

import copy
import struct
from dataclasses import replace

import pytest

from tessera.compiler import bootstrap_prune_audit as audit
from tessera.compiler import nvidia_native as native
from tessera.compiler import scheduled_attention_recompute as recompute
from tessera.compiler import scheduled_checkpoint as checkpoint
from tessera.compiler.graph_ir import GraphIRFunction, GraphIRModule, IRArg, IROp, tensor_ir_type
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tests._support import nvidia_attention_checkpoint_baseline as baseline
from tests.device.nvidia.test_lse_checkpoint_native import _backward_module, _forward_module
from tests.unit.test_nvidia_native_recompute_schedule import recompute_module

PIPELINE = "tessera-nvidia-pipeline-sm120"
pytestmark = pytest.mark.skipif(find_tessera_opt() is None, reason="requires production tessera-opt")

#: Launch fields that must not move when authorship moves.
LAUNCH = ("abi_id", "entry_symbol", "buffers", "scalars", "shape_guards", "geometry", "workspace", "ordering")
#: Provenance that names the source text, which the two routes spell differently.
SOURCE_DIGESTS = {"graph_ir_digest", "schedule_ir_digest", "target_ir_digest", "tile_ir_digest"}
SHAPES = ((1, 2, 1, 3, 4, 4, 3), (2, 4, 2, 5, 7, 8, 6))
F32_POLICY = {"scale", "softcap", "dropout_p"}


def _f32(value: float) -> bytes:
    return struct.pack("f", value)


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


def _broadcast_bias(module: GraphIRModule, shape: tuple[int, ...], index: int) -> GraphIRModule:
    fn = module.functions[0]
    fn.args[index].ir_type = tensor_ir_type(shape, "fp32")
    fn.body[0].operand_types = [str(arg.ir_type) for arg in fn.args]
    return module


def _saved(backward: bool, shape, bias: str | None, permuted: bool) -> GraphIRModule:
    module = (_backward_module if backward else _forward_module)(saved=True, shape=shape, bias=bias is not None)
    if bias == "broadcast":
        b, hq = shape[0], shape[1]
        module = _broadcast_bias(module, (b, hq, 1, 1), 5 if backward else 3)
    if permuted:
        module.functions[0].args.reverse()
    return module


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("bias", [None, "full", "broadcast"])
@pytest.mark.parametrize("permuted", [False, True])
def test_saved_lse_projection_reproduces_retired_graph_authored_packager(compiles, backward, shape, bias, permuted):
    module = _saved(backward, shape, bias, permuted)
    before = copy.deepcopy(module)
    family = "attention_backward_lse" if backward else "attention_lse"
    assert native.native_package_kind(module) == family
    old = (baseline.baseline_attention_backward_lse if backward else baseline.baseline_attention_lse)(
        module, pipeline_name=PIPELINE)
    new = native.package_native(module, pipeline_name=PIPELINE)
    assert module == before
    for field in LAUNCH:
        assert getattr(old.descriptor, field) == getattr(new.descriptor, field), field
    old_provenance, new_provenance = old.descriptor.provenance, new.descriptor.provenance
    assert set(old_provenance) == set(new_provenance)
    assert ({k: v for k, v in old_provenance.items() if k not in SOURCE_DIGESTS}
            == {k: v for k, v in new_provenance.items() if k not in SOURCE_DIGESTS})
    assert compiles == [old.descriptor.entry_symbol, new.descriptor.entry_symbol]


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("policy", [{}, {"scale": .3, "window": (2, 0), "softcap": .7, "dropout": .2, "seed": 17}])
def test_recompute_projection_reproduces_retired_python_tile_packager(compiles, dtype, bias, permuted, policy):
    module = recompute_module(dtype, bias, permuted)
    module.functions[0].body[0].kwargs.update(policy)
    assert native.native_package_kind(module) == "attention_backward"
    old = baseline.baseline_attention_backward(module, pipeline_name=PIPELINE)
    new = native.package_native(module, pipeline_name=PIPELINE)
    for field in ("abi_id", "buffers", "scalars", "geometry", "workspace", "ordering"):
        assert getattr(old.descriptor, field) == getattr(new.descriptor, field), field
    assert set(old.descriptor.shape_guards) == set(new.descriptor.shape_guards)
    # The retired route named the kernel from a Python semantic hash; the
    # native route names it from the Schedule. Same storage, same policy.
    storage = {"fp16": "f16", "bf16": "bf16", "fp32": "f32"}[dtype]
    assert old.descriptor.entry_symbol.startswith(f"tessera_tile_attention_backward_{storage}_deterministic_")
    assert new.descriptor.entry_symbol.startswith(f"tessera_tile_attention_backward_{storage}_recompute_")
    shared = set(old.descriptor.provenance) - SOURCE_DIGESTS
    # The retired packager recorded the Python double; the native contract
    # records the f32 the kernel computes with. Compare those at f32.
    def physical(provenance):
        return {k: _f32(provenance[k]) if k in F32_POLICY else provenance[k] for k in shared}
    assert physical(old.descriptor.provenance) == physical(new.descriptor.provenance)
    assert new.descriptor.provenance["compiler_route"] == "canonical_scheduled_tile_consumer"


def _typed_forward(storage: str, kv: str, output: str, bias: tuple[int, ...] | None,
                   shape=(1, 4, 2, 3, 5, 8, 6)) -> GraphIRModule:
    b, hq, hkv, sq, sk, d, dv = shape
    args = [IRArg("q", tensor_ir_type((b, hq, sq, d), storage)),
            IRArg("k", tensor_ir_type((b, hkv, sk, d), kv)),
            IRArg("v", tensor_ir_type((b, hkv, sk, dv), kv))]
    if bias is not None:
        args.append(IRArg("bias", tensor_ir_type(bias, "fp32")))
    results = [tensor_ir_type((b, hq, sq, dv), output), tensor_ir_type((b, hq, sq), "fp32")]
    return GraphIRModule(functions=[GraphIRFunction(
        name="typed_saved", args=args, result_types=results,
        body=[IROp(result="o,row_lse", op_name="tessera.flash_attn", operands=["%" + a.name for a in args],
                   operand_types=[str(a.ir_type) for a in args], result_type=str(results[0]),
                   inferred_types=tuple(results), kwargs={"causal": True, "lse_checkpoint": "saved"})],
        return_values=["%o", "%row_lse"])])


def _scaled_forward(storage: str, kv: str, scales: tuple[tuple[int, ...], tuple[int, ...]],
                    bias: tuple[int, ...] | None, shape=(1, 4, 2, 3, 5, 8, 6)) -> GraphIRModule:
    b, hq, hkv, sq, sk, d, dv = shape
    args = [IRArg("q", tensor_ir_type((b, hq, sq, d), storage)),
            IRArg("k8", tensor_ir_type((b, hkv, sk, d), kv)),
            IRArg("v8", tensor_ir_type((b, hkv, sk, dv), kv)),
            IRArg("ks", tensor_ir_type(scales[0], "fp32")),
            IRArg("vs", tensor_ir_type(scales[1], "fp32"))]
    if bias is not None:
        args.append(IRArg("bias", tensor_ir_type(bias, "fp32")))
    key_t, value_t = tensor_ir_type((b, hkv, sk, d), "fp32"), tensor_ir_type((b, hkv, sk, dv), "fp32")
    results = [tensor_ir_type((b, hq, sq, dv), storage), tensor_ir_type((b, hq, sq), "fp32")]
    attention_operands = ["%q", "%k", "%v"] + (["%bias"] if bias is not None else [])
    return GraphIRModule(functions=[GraphIRFunction(
        name="scaled_saved", args=args, result_types=results,
        body=[IROp(result="k", op_name="tessera.dequantize_scaled", operands=["%k8", "%ks"],
                   operand_types=[str(args[1].ir_type), str(args[3].ir_type)], result_type=str(key_t)),
              IROp(result="v", op_name="tessera.dequantize_scaled", operands=["%v8", "%vs"],
                   operand_types=[str(args[2].ir_type), str(args[4].ir_type)], result_type=str(value_t)),
              IROp(result="o,row_lse", op_name="tessera.flash_attn", operands=attention_operands,
                   operand_types=[str(args[0].ir_type), str(key_t), str(value_t)]
                   + ([str(args[5].ir_type)] if bias is not None else []),
                   result_type=str(results[0]), inferred_types=tuple(results),
                   kwargs={"causal": True, "lse_checkpoint": "saved"})],
        return_values=["%o", "%row_lse"])])


TYPED = [
    ("fp16", "fp16", "fp16", None), ("bf16", "bf16", "fp32", (1, 4, 1, 1)),
    ("fp16", "bf16", "fp16", None), ("bf16", "fp32", "bf16", (1, 4, 3, 5)),
    ("fp16", "fp8_e4m3", "fp16", None), ("bf16", "fp8_e5m2", "fp32", (1, 1, 3, 5)),
]
SCALED = [
    ("fp16", "fp8_e4m3", ((1, 1, 1, 1), (1, 1, 1, 1)), None),
    ("bf16", "fp8_e5m2", ((1, 2, 1, 1), (1, 2, 1, 1)), (1, 4, 1, 1)),
    ("fp16", "fp8_e5m2", ((1, 2, 5, 1), (1, 1, 1, 6)), (1, 4, 3, 5)),
    ("fp32", "fp8_e4m3", ((1, 2, 5, 8), (1, 2, 5, 6)), None),
]


@pytest.mark.parametrize("storage,kv,output,bias", TYPED)
def test_typed_forward_projection_matches_graph_authored_launch(compiles, storage, kv, output, bias):
    module = _typed_forward(storage, kv, output, bias)
    assert native.native_package_kind(module) == "attention_lse"
    expected = baseline.graph_authored_forward_launch(module)
    package = native.package_native(module, pipeline_name=PIPELINE)
    assert package.descriptor.buffers == expected["buffers"]
    assert frozenset(package.descriptor.shape_guards) == expected["guards"]
    assert tuple(s.name for s in package.descriptor.scalars) == expected["scalars"]


@pytest.mark.parametrize("storage,kv,scales,bias", SCALED)
def test_scaled_kv_projection_matches_graph_authored_launch(compiles, storage, kv, scales, bias):
    module = _scaled_forward(storage, kv, scales, bias)
    assert native.native_package_kind(module) == "attention_scaled_kv"
    expected = baseline.graph_authored_forward_launch(module)
    package = native.package_native(module, pipeline_name=PIPELINE)
    assert package.descriptor.buffers == expected["buffers"]
    assert frozenset(package.descriptor.shape_guards) == expected["guards"]
    assert tuple(s.name for s in package.descriptor.scalars) == expected["scalars"]
    assert tuple(package.descriptor.provenance["kv_scale_shapes"]) == scales[0] + scales[1]


def _forbid_graph_admission(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("an artifact consumer re-entered Graph admission")

    for name in ("_attention_lse_contract", "_attention_scaled_kv_contract",
                 "_attention_backward_lse_contract", "_attention_backward_contract",
                 "supports_attention_lse", "supports_attention_scaled_kv",
                 "supports_attention_backward_lse", "supports_attention_backward", "_shape"):
        monkeypatch.setattr(native, name, forbidden)


def test_consumers_never_read_the_graph(compiles, monkeypatch):
    artifacts = [
        checkpoint.lower_checkpoint_graph(_saved(False, SHAPES[1], "broadcast", True)),
        checkpoint.lower_checkpoint_graph(_saved(True, SHAPES[1], "full", False), backward=True),
        checkpoint.lower_checkpoint_graph(_typed_forward("bf16", "fp8_e4m3", "fp32", (1, 4, 1, 1))),
        checkpoint.lower_checkpoint_graph(_scaled_forward("fp16", "fp8_e5m2", SCALED[2][2], (1, 4, 3, 5))),
    ]
    recomputed = recompute.lower_native_attention_recompute(recompute_module("bf16", True, True))
    _forbid_graph_admission(monkeypatch)
    for artifact in artifacts:
        package = native.package_scheduled_checkpoint(artifact, pipeline_name=PIPELINE)
        assert package.descriptor.entry_symbol == artifact.entry
    package = native.package_scheduled_attention_recompute(recomputed, pipeline_name=PIPELINE)
    assert package.descriptor.entry_symbol == recomputed.entry
    assert [b.name for b in package.descriptor.buffers] == ["do", "q", "k", "v", "bias", "dq", "dk", "dv"]


CHECKPOINT_TAMPERING = {
    "names_swapped": lambda a: {"names": (a.names[1], a.names[0], *a.names[2:])},
    "dims": lambda a: {"dims": (a.dims[0] + 1, *a.dims[1:])},
    "scale": lambda a: {"scale": a.scale * 2},
    "causal": lambda a: {"causal": not a.causal},
    "storage": lambda a: {"storage": "f16"},
    "bias_flag": lambda a: {"bias": not a.bias},
    "entry": lambda a: {"entry": a.entry + "_x"},
    "schedule_digest": lambda a: {"schedule_digest": "0" * 64},
    "tile_ir": lambda a: {"tile_ir": a.tile_ir.replace("causal = true", "causal = false")},
    "schedule_ir": lambda a: {"schedule_ir": a.schedule_ir.replace("causal = true", "causal = false")},
}


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("case", sorted(CHECKPOINT_TAMPERING))
def test_tampered_checkpoint_fails_before_compilation(compiles, backward, case):
    artifact = checkpoint.lower_checkpoint_graph(_saved(backward, SHAPES[0], None, False), backward=backward)
    changes = CHECKPOINT_TAMPERING[case](artifact)
    assert any(getattr(artifact, k) != v for k, v in changes.items()), case
    with pytest.raises(ValueError):
        native.package_scheduled_checkpoint(replace(artifact, **changes), pipeline_name=PIPELINE)
    assert compiles == []


RECOMPUTE_TAMPERING = {
    "input_names": lambda a: {"input_names": (a.input_names[1], a.input_names[0], *a.input_names[2:])},
    "output_names": lambda a: {"output_names": (a.output_names[2], a.output_names[1], a.output_names[0])},
    "dims": lambda a: {"dims": (2, *a.dims[1:])},
    "storage": lambda a: {"storage": "f32" if a.storage != "f32" else "f16"},
    "scale": lambda a: {"scale": a.scale / 2},
    "window": lambda a: {"window_left": 3},
    "dropout_seed": lambda a: {"dropout_seed": a.dropout_seed + 1},
    "entry": lambda a: {"entry": a.entry + "_x"},
    "tile_ir": lambda a: {"tile_ir": a.tile_ir.replace("dropout_seed = 0", "dropout_seed = 1")},
}


@pytest.mark.parametrize("case", sorted(RECOMPUTE_TAMPERING))
def test_tampered_recompute_fails_before_compilation(compiles, case):
    artifact = recompute.lower_native_attention_recompute(recompute_module("fp16", False, False))
    changes = RECOMPUTE_TAMPERING[case](artifact)
    assert any(getattr(artifact, k) != v for k, v in changes.items()), case
    with pytest.raises(ValueError):
        native.package_scheduled_attention_recompute(replace(artifact, **changes), pipeline_name=PIPELINE)
    assert compiles == []


def test_retired_python_tile_constructor_left_production():
    assert not hasattr(native, "emit_attention_backward_tile_ir")


def test_scheduled_admission_predicates_follow_graph_admission():
    cases = {
        "attention_lse": _saved(False, SHAPES[0], "full", False),
        "attention_backward_lse": _saved(True, SHAPES[0], None, True),
        "attention_scaled_kv": _scaled_forward(*SCALED[0]),
        "attention_backward": recompute_module("fp32", False, False),
    }
    predicates = {
        "attention_lse": checkpoint.supports_scheduled_attention_lse,
        "attention_backward_lse": checkpoint.supports_scheduled_attention_backward_lse,
        "attention_scaled_kv": checkpoint.supports_scheduled_attention_scaled_kv,
        "attention_backward": recompute.supports_scheduled_attention_recompute,
    }
    for family, module in cases.items():
        assert native.native_package_kind(module) == family
        for other, predicate in predicates.items():
            assert predicate(module) == (other == family), (family, other)
            assert not predicate(module, target="rocm_gfx1201")


def test_prune_rows_name_the_typed_consumers():
    rows = {(target, family): (route, status) for target, family, route, status in audit.family_rows()}
    for family, route in (
        ("attention_lse", "scheduled_checkpoint.supports_scheduled_attention_lse"),
        ("attention_scaled_kv", "scheduled_checkpoint.supports_scheduled_attention_scaled_kv"),
        ("attention_backward_lse", "scheduled_checkpoint.supports_scheduled_attention_backward_lse"),
        ("attention_backward", "scheduled_attention_recompute.supports_scheduled_attention_recompute"),
    ):
        assert rows[("nvidia_sm120", family)] == (route, "compiled")
    # The SM120 override does not leak onto x86's own backward route.
    assert rows[("x86", "attention_backward")][0] == (
        "scheduled_attention_backward.supports_scheduled_attention_backward")
    inventory = {inv.target: inv for inv in audit.collect_inventories()}["nvidia_sm120"]
    packagers = dict(inventory.packagers)
    assert packagers["package_scheduled_checkpoint"] == "ScheduledCheckpointArtifact"
    assert packagers["package_scheduled_attention_recompute"] == "ScheduledAttentionRecomputeArtifact"
