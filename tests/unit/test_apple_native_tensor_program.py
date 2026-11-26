"""APPLE-NTBI-2 slice 2: an apple7 tensor program packages every member.

Each frame stage binds the traced Graph op it came from; that op's standalone
Apple scheduled package must carry the same launch-level Tile carrier the
program projects for the member. The Metal test runs the member packages one
after another (no frame owner yet) and checks the numerics.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import apple_native
from tessera.compiler import apple_tensor_program as atp
from tessera.compiler import scheduled_apple_gpu_descriptor as descriptor
from tessera.compiler.scheduled_matmul import find_tessera_opt
from tessera.compiler.trace import to_graph_ir_module, trace


def _compiler_registered() -> bool:
    tool = find_tessera_opt()
    return tool is not None and descriptor.apple_gpu_value_lowering_available(str(tool))


pytestmark = pytest.mark.skipif(
    not (_compiler_registered() and apple_native.tools_available()),
    reason="requires tessera-opt with the Apple backend and the Apple GPU runtime dylib",
)


def _half(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float16)


def _graph(fn, arrays, names):
    return to_graph_ir_module(trace(fn, *arrays, arg_names=names), name="prog", target="apple_gpu")


def _scores_values(s, v):
    return ts.ops.matmul(ts.ops.softmax(s, axis=-1), v, output_dtype="fp32")


def _self_attention_like(x):
    return ts.ops.matmul(ts.ops.softmax(x, axis=-1), x, output_dtype="fp32")


def _self_product(x):
    p = ts.ops.softmax(x, axis=-1)
    return ts.ops.matmul(p, p, output_dtype="fp32")


@pytest.fixture(scope="module")
def program():
    return atp.compile_apple_tensor_program(
        _graph(_scores_values, (_half(64, 32), _half(32, 48)), ("s", "v")))


def test_every_member_is_bound_to_its_traced_op_and_packaged(program):
    frame = program.frame
    assert (frame["target"], frame["architecture"]) == ("apple_gpu", "apple7")
    assert [b["role"] for b in frame["buffers"]] == ["input", "input", "private", "output"]
    softmax, product = program.stages
    assert (softmax.op_name, softmax.carrier) == ("tessera.softmax", "tile.softmax_kernel")
    assert (product.op_name, product.carrier) == ("tessera.matmul", "tile.matmul_kernel")
    # The product consumes the softmax result, which is the private buffer.
    assert product.inputs == (softmax.output, "v")
    assert softmax.package.descriptor.provenance["route"] == "apple_softmax_native_library"
    assert product.package.descriptor.provenance["route"] == "apple_gpu_simdgroup_gemm_f16"
    for stage in program.stages:
        # The member projection and the standalone package carry one carrier
        # with the same content-addressed schedule digest.
        assert atp._carrier(stage.member_ir, stage.carrier) == atp._carrier(
            stage.artifact.tile_ir, stage.carrier)
        assert f"tessera.native.tensor_program_member_index = {stage.index}" in stage.member_ir


def test_an_aliased_product_keeps_one_package_binding_per_operand_position():
    # matmul(p, p): the member keeps both positions, so the package must too.
    aliased = atp.compile_apple_tensor_program(_graph(_self_product, (_half(64, 64),), ("x",)))
    product = aliased.stages[1]
    assert product.inputs == (aliased.stages[0].output,) * 2
    assert len(set(product.bindings)) == 2
    assert [b.name for b in product.package.descriptor.buffers][:2] == list(product.bindings)
    assert aliased.frame["stages"][1]["inputs"] == [1, 1]
    aliased.validate()


def test_validate_replays_the_source_and_every_member(program):
    program.validate()


def test_a_drifted_tile_program_is_refused_on_replay(program):
    tampered = dataclasses.replace(
        program, tile_ir=program.tile_ir.replace("tessera.macro_tile_m = 32", "tessera.macro_tile_m = 64"))
    with pytest.raises(atp.AppleTensorProgramError, match="Tile differs"):
        tampered.validate()


def test_a_member_whose_package_carries_another_schedule_is_refused(monkeypatch):
    real = atp._package

    def drifted(op_name, module):
        artifact, package = real(op_name, module)
        if op_name == "tessera.matmul":
            artifact = dataclasses.replace(
                artifact, tile_ir=artifact.tile_ir.replace(
                    "tessera.macro_tile_n = 32", "tessera.macro_tile_n = 64"))
        return artifact, package

    monkeypatch.setattr(atp, "_package", drifted)
    with pytest.raises(atp.AppleTensorProgramError, match="member 1 Tile carrier differs"):
        atp.compile_apple_tensor_program(
            _graph(_scores_values, (_half(64, 32), _half(32, 48)), ("s", "v")))


def test_frame_stages_must_bind_the_traced_operands(program):
    frame = program.frame
    frame["stages"][1]["inputs"] = list(reversed(frame["stages"][1]["inputs"]))
    with pytest.raises(atp.AppleTensorProgramError, match="stage 1 binds different buffers"):
        atp._bind_stages(program.graph.functions[0], frame, program.tile_ir, program.compiler)


@pytest.mark.parametrize("fn, arrays, why", [
    (lambda a, c, b: ts.ops.matmul(ts.ops.mul(a, c), b, output_dtype="fp32"),
     (_half(64, 32), _half(64, 32), _half(32, 48)),
     "must be tessera.matmul, tessera.softmax, tessera.gelu, tessera.relu, tessera.add, "
     "tessera.rmsnorm or tessera.layer_norm"),
    (lambda a, b: ts.ops.softmax(ts.ops.matmul(a, b, output_dtype="fp32"), axis=-1),
     (_half(64, 32), _half(32, 48)), "must end in its matrix product"),
    (lambda a: ts.ops.matmul(a, a, output_dtype="fp32"),
     (_half(32, 32),), "needs a producer and a matrix product"),
])
def test_requested_programs_outside_the_profile_are_refused(fn, arrays, why):
    names = ("a", "b", "c")[:len(arrays)]
    with pytest.raises(RuntimeError, match=f"APPLE_TENSOR_PROGRAM_UNSUPPORTED: .*{why}"):
        atp.compile_apple_tensor_program(_graph(fn, arrays, names))


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("fn, shapes, names", [
    (_scores_values, ((64, 32), (32, 48)), ("s", "v")),
    (_self_attention_like, ((64, 64),), ("x",)),
    (_self_product, ((64, 64),), ("x",)),
])
def test_member_packages_compute_the_program_on_metal(fn, shapes, names):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    arrays = tuple(_half(*shape, seed=i + 3) for i, shape in enumerate(shapes))
    program = atp.compile_apple_tensor_program(_graph(fn, arrays, names))
    output, receipt = atp.run_members_sequentially(program, dict(zip(names, arrays)))
    # Not frame-owned: one runtime launch per member, intermediates via host.
    assert receipt["frame_owned"] is False
    assert receipt["member_launches"] == len(program.stages) == 2
    scores = arrays[0].astype(np.float32)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    probabilities = (weights / weights.sum(-1, keepdims=True)).astype(np.float16)
    rhs = probabilities if fn is _self_product else arrays[-1]
    reference = probabilities.astype(np.float32) @ rhs.astype(np.float32)
    assert output.dtype == np.float32 and output.shape == reference.shape
    np.testing.assert_allclose(output, reference, rtol=0, atol=2e-3)
    framed, _ = atp.execute_frame(program, dict(zip(names, arrays)))
    assert framed.tobytes() == output.tobytes()


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("shapes", [((64, 32), (32, 48)), ((100, 37), (37, 53))])
def test_frame_owner_runs_the_program_in_one_command_buffer(shapes):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    s, v = (_half(*shape, seed=i + 7) for i, shape in enumerate(shapes))
    program = atp.compile_apple_tensor_program(_graph(_scores_values, (s, v), ("s", "v")))
    import ctypes

    from tessera._apple_gpu_dispatch import apple_gpu_runtime

    # The GPU clock is opt-in, like every dispatch clock in the runtime.
    telemetry = apple_gpu_runtime().tessera_apple_gpu_dispatch_telemetry_set_enabled
    telemetry(ctypes.c_int32(1))
    try:
        output, receipt = atp.execute_frame(program, {"s": s, "v": v})
    finally:
        telemetry(ctypes.c_int32(0))
    assert receipt["frame_owned"] is True
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, 2)
    # The softmax result is a device-only (MTLStorageModePrivate) buffer.
    assert receipt["private_device_buffers"] == 1
    assert receipt["gpu_time_ns"] is not None and receipt["gpu_time_ns"] > 0
    # Same kernels, same order: the frame equals the per-member launches exactly.
    sequential, _ = atp.run_members_sequentially(program, {"s": s, "v": v})
    assert output.dtype == np.float32 and output.tobytes() == sequential.tobytes()


@pytest.mark.hardware_apple_gpu
def test_frame_owner_refuses_a_member_packaged_against_another_runtime(program):
    stage = program.stages[1]
    image = dataclasses.replace(stage.package.image, payload=stage.package.image.payload + b"\0")
    other = dataclasses.replace(stage, package=dataclasses.replace(stage.package, image=image))
    drifted = dataclasses.replace(program, stages=(program.stages[0], other))
    with pytest.raises(atp.AppleTensorProgramError, match="different runtime image"):
        atp.execute_frame(drifted, {"s": _half(64, 32), "v": _half(32, 48)})


@pytest.mark.hardware_apple_gpu
def test_frame_owner_refuses_a_frame_whose_buffers_do_not_fit_the_stage():
    import ctypes

    from tessera._apple_gpu_dispatch import bind_registered

    owner = bind_registered("tessera_apple_gpu_tensor_frame_execute")
    assert owner is not None
    x, out = np.zeros((4, 8), np.float16), np.full((4, 8), -1, np.float16)
    # The output is declared one element short of the softmax result.
    buffers = (ctypes.c_int64 * 4)(0, x.nbytes, 2, out.nbytes - 2)
    hosts = (ctypes.c_void_p * 2)(x.ctypes.data, out.ctypes.data)
    stage = (ctypes.c_int64 * 10)(1, 0, -1, 1, 4, 1, 8, 0, 0, 0)
    none = (ctypes.c_char_p * 1)(None)
    receipt = (ctypes.c_int64 * 4)()
    assert owner(buffers, hosts, 2, stage, none, none, 1, receipt) == -1
    assert list(receipt) == [0, 0, 0, 0] and (out == -1).all()


# --- Slice 4: public @jit binding and RuntimeArtifact restoration ---------


def _jit_scores_values():
    @ts.jit(target="apple_gpu")
    def program(s, v):
        return ts.ops.matmul(ts.ops.softmax(s, axis=-1), v, output_dtype="fp32")
    return program


def _tampered(artifact, **changes):
    import copy

    metadata = copy.deepcopy(artifact.metadata)
    manifest = metadata["native_program"]
    for key, edit in changes.items():
        manifest[key] = edit(manifest[key])
    fields = {name: manifest[name] for name in ("source_ir", "schedule_ir", "tile_ir")}
    return dataclasses.replace(artifact, metadata=metadata, graph_ir=fields["source_ir"],
                               schedule_ir=fields["schedule_ir"], tile_ir=fields["tile_ir"])


def test_the_artifact_records_a_graph_that_rebuilds_its_source(program):
    import json

    artifact = atp.runtime_artifact(program, argument_names=("s", "v"))
    json.dumps(artifact.metadata)  # portable data, no executable payload
    assert artifact.metadata["compiler_path"] == "canonical_apple_tensor_program"
    assert artifact.metadata["restoration"] == "matching_compiler_required"
    graph = atp._graph_from_record(artifact.metadata["native_program"]["graph"])
    assert atp._source(graph) == program.source_ir
    restored = atp.restore_runtime_artifact(artifact)
    assert (restored.tile_ir, len(restored.stages)) == (program.tile_ir, 2)


@pytest.mark.parametrize("field, edit, why", [
    ("tile_ir", lambda t: t.replace("tessera.macro_tile_m = 32", "tessera.macro_tile_m = 64"),
     "differs from its compiler replay"),
    ("members", lambda m: [{**m[0], "image_digest": "0" * 64}, m[1]],
     "member packages differ on replay"),
    ("compiler_identity", lambda _: "0" * 64, "compiled by a different compiler"),
    ("graph", lambda g: {**g, "body": [g["body"][0], {**g["body"][1], "kwargs": {"output_dtype": "fp16"}}]},
     "does not rebuild its source"),
])
def test_restoration_refuses_an_artifact_that_differs_from_the_compiler(program, field, edit, why):
    artifact = _tampered(atp.runtime_artifact(program), **{field: edit})
    with pytest.raises(atp.AppleTensorProgramError, match=why):
        atp.restore_runtime_artifact(artifact)


@pytest.mark.hardware_apple_gpu
def test_public_jit_runs_the_program_in_one_command_buffer(monkeypatch):
    from tests._support.apple import require_apple_chip_identity
    from tessera import runtime as rt

    require_apple_chip_identity()
    s, v = _half(100, 37, seed=11), _half(37, 53, seed=12)
    program = _jit_scores_values()
    output = program(s, v)
    receipt = program._apple_tensor_last_receipt
    assert receipt["compiler_path"] == "canonical_apple_tensor_program"
    assert receipt["frame_owned"] is True
    assert (receipt["command_buffers"], receipt["dispatches"], receipt["private_device_buffers"]) == (1, 2, 1)
    # The requested fp32 result dtype is kept (the MPS lane returned fp16).
    assert output.dtype == np.float32 and output.shape == (100, 53)
    compiled, _ = next(iter(program._apple_tensor_program_cache.values()))
    roots = [arg.name for arg in compiled.graph.functions[0].args]
    framed, _ = atp.execute_frame(compiled, dict(zip(roots, (s, v))))
    assert output.tobytes() == framed.tobytes()
    scores = s.astype(np.float32)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    probabilities = (weights / weights.sum(-1, keepdims=True)).astype(np.float16)
    np.testing.assert_allclose(output, probabilities.astype(np.float32) @ v.astype(np.float32),
                               rtol=0, atol=2e-3)
    # A warm call binds the compiled program; it neither traces nor compiles.
    monkeypatch.setattr(atp, "compile_apple_tensor_program",
                        lambda *_: pytest.fail("warm call recompiled"))
    assert program(s, v).tobytes() == output.tobytes()
    # The exported artifact restores through the compiler and replays exactly.
    monkeypatch.undo()
    artifact = program.runtime_artifact()
    assert artifact.metadata["arg_names"] == ["s", "v"]
    launched = rt.launch(artifact, {"s": s, "v": v})
    assert launched["ok"] and launched["execution_kind"] == "native_gpu"
    assert np.asarray(launched["output"]).tobytes() == output.tobytes()


_SIDE_EFFECTS: list = []


def _logged_attention(s, v):
    _SIDE_EFFECTS.append(s.shape)
    return ts.ops.matmul(ts.ops.softmax(s, axis=-1), v, output_dtype="fp32")


def _attention_and_probabilities(s, v):
    probabilities = ts.ops.softmax(s, axis=-1)
    return probabilities, ts.ops.matmul(probabilities, v, output_dtype="fp32")


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("fn, arrays", [
    # A host side effect the Graph cannot carry: declined before tracing, so
    # warm calls keep running it.
    (_logged_attention, (_half(64, 32), _half(32, 48))),
    # Two results; the program has one: declined before admission.
    (_attention_and_probabilities, (_half(64, 32), _half(32, 48))),
    # Inputs outside the profile's f16 storage: declined before tracing.
    (lambda s, v: ts.ops.matmul(ts.ops.softmax(s, axis=-1), v),
     (_half(64, 32).astype(np.float32), _half(32, 48).astype(np.float32))),
    # A producer the owner has no member for: declined after tracing.
    (lambda s, v: ts.ops.matmul(ts.ops.mul(s, s), v, output_dtype="fp32"),
     (_half(64, 32), _half(32, 48))),
])
def test_public_jit_leaves_programs_outside_the_profile_to_other_routes(fn, arrays):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    program = ts.jit(target="apple_gpu")(fn)
    program(*arrays)
    assert program._apple_tensor_last_receipt is None
    # No program is admitted. A body with a call statement is declined before
    # any native route runs (JIT-NATIVE-STATEMENT-EFFECT-2026-11-24), so it
    # leaves the cache empty; the others record their decline as None.
    assert set(program._apple_tensor_program_cache.values()) <= {None}
    assert program.runtime_artifact().metadata.get("compiler_path") != "canonical_apple_tensor_program"


@pytest.mark.hardware_apple_gpu
def test_public_jit_declines_an_effectful_body_before_tracing(monkeypatch):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    program = ts.jit(target="apple_gpu")(_logged_attention)
    # The scan, not the trace, decides: tracing would run the side effect once
    # and cache a program that never runs it again.
    monkeypatch.setattr(type(program), "_traced_autodiff_module",
                        lambda *_: pytest.fail("an effectful body was traced"))
    assert program._compile_apple_tensor_program((_half(64, 32), _half(32, 48))) is None


# --- Slice 5: the tanh GELU activation as a producer member ----------------


def _gelu_product(x, w):
    return ts.ops.matmul(ts.ops.gelu(x), w, output_dtype="fp32")


def _gelu_softmax_product(x, w):
    return ts.ops.matmul(ts.ops.softmax(ts.ops.gelu(x), axis=-1), w, output_dtype="fp32")


def _both_operands_gelu(x, w):
    return ts.ops.matmul(ts.ops.gelu(x), ts.ops.gelu(w), output_dtype="fp32")


def _gelu_f16(x):
    """The kernel's GELU: tanh approximation in f32, rounded to f16 storage."""
    v = x.astype(np.float32)
    t = np.clip(np.float32(0.7978845608028654) * (v + np.float32(0.044715) * v ** 3), -30, 30)
    return (np.float32(0.5) * v * (1 + np.tanh(t))).astype(np.float16)


def _softmax_f16(x):
    v = x.astype(np.float32)
    w = np.exp(v - v.max(-1, keepdims=True))
    return (w / w.sum(-1, keepdims=True)).astype(np.float16)


def test_a_gelu_member_binds_the_shared_activation_carrier():
    gelu = atp.compile_apple_tensor_program(
        _graph(_gelu_product, (_half(64, 32), _half(32, 48)), ("x", "w")))
    activation, product = gelu.stages
    assert (activation.op_name, activation.carrier) == ("tessera.gelu", "tile.activation_kernel")
    provenance = activation.package.descriptor.provenance
    assert provenance["route"] == "apple_gelu_native_library"
    assert provenance["approximation"] == "tanh"
    assert activation.package.descriptor.abi_id == apple_native.APPLE_GELU_F16_ABI
    assert product.inputs == (activation.output, "w")
    assert [b["role"] for b in gelu.frame["buffers"]] == ["input", "input", "private", "output"]
    gelu.validate()


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("fn, shapes, members, private", [
    (_gelu_product, ((64, 32), (32, 48)), 2, 1),
    (_gelu_product, ((100, 37), (37, 53)), 2, 1),
    # A producer chain: GELU feeds softmax feeds the product.
    (_gelu_softmax_product, ((100, 37), (37, 53)), 3, 2),
    # A producer on each operand.
    (_both_operands_gelu, ((64, 32), (32, 48)), 3, 2),
])
def test_frame_owner_runs_gelu_programs_in_one_command_buffer(fn, shapes, members, private):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    x, w = (_half(*shape, seed=i + 21) for i, shape in enumerate(shapes))
    program = atp.compile_apple_tensor_program(_graph(fn, (x, w), ("x", "w")))
    assert len(program.stages) == members
    output, receipt = atp.execute_frame(program, {"x": x, "w": w})
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, members)
    assert receipt["private_device_buffers"] == private
    # Same kernels, same order: the frame equals the per-member launches exactly.
    sequential, _ = atp.run_members_sequentially(program, {"x": x, "w": w})
    assert output.dtype == np.float32 and output.tobytes() == sequential.tobytes()
    lhs = _gelu_f16(x)
    if fn is _gelu_softmax_product:
        lhs = _softmax_f16(lhs)
    rhs = _gelu_f16(w) if fn is _both_operands_gelu else w
    reference = lhs.astype(np.float32) @ rhs.astype(np.float32)
    # The softmax program's bound (measured max error ~3e-4 on the M1 Max).
    np.testing.assert_allclose(output, reference, rtol=0, atol=2e-3)


@pytest.mark.hardware_apple_gpu
def test_frame_owner_refuses_a_gelu_stage_whose_output_does_not_fit():
    import ctypes

    from tessera._apple_gpu_dispatch import bind_registered

    owner = bind_registered("tessera_apple_gpu_tensor_frame_execute")
    assert owner is not None
    x, out = np.zeros((4, 8), np.float16), np.full((4, 8), -1, np.float16)
    # The output is declared one element short of the GELU result.
    buffers = (ctypes.c_int64 * 4)(0, x.nbytes, 2, out.nbytes - 2)
    hosts = (ctypes.c_void_p * 2)(x.ctypes.data, out.ctypes.data)
    stage = (ctypes.c_int64 * 10)(3, 0, -1, 1, 4, 1, 8, 0, 0, 0)
    none = (ctypes.c_char_p * 1)(None)
    receipt = (ctypes.c_int64 * 4)()
    assert owner(buffers, hosts, 2, stage, none, none, 1, receipt) == -1
    assert list(receipt) == [0, 0, 0, 0] and (out == -1).all()


@pytest.mark.hardware_apple_gpu
def test_public_jit_runs_a_gelu_program_in_one_command_buffer():
    from tests._support.apple import require_apple_chip_identity
    from tessera import runtime as rt

    require_apple_chip_identity()
    x, w = _half(100, 37, seed=31), _half(37, 53, seed=32)
    program = ts.jit(target="apple_gpu")(_gelu_softmax_product)
    output = program(x, w)
    receipt = program._apple_tensor_last_receipt
    assert receipt["compiler_path"] == "canonical_apple_tensor_program"
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, 3)
    roots = [arg.name for arg in program._apple_tensor_last_program.graph.functions[0].args]
    framed, _ = atp.execute_frame(program._apple_tensor_last_program, dict(zip(roots, (x, w))))
    assert output.dtype == np.float32 and output.tobytes() == framed.tobytes()
    artifact = program.runtime_artifact()
    launched = rt.launch(artifact, {"x": x, "w": w})
    assert launched["ok"] and np.asarray(launched["output"]).tobytes() == output.tobytes()


# --- Slice 6: the f16 ReLU as compiler-synthesized MSL ----------------------


def _relu_product(x, w):
    return ts.ops.matmul(ts.ops.relu(x), w, output_dtype="fp32")


def _relu_gelu_softmax_product(x, w):
    return ts.ops.matmul(ts.ops.softmax(ts.ops.gelu(ts.ops.relu(x)), axis=-1), w,
                         output_dtype="fp32")


def _both_operands_relu(x, w):
    return ts.ops.matmul(ts.ops.relu(x), ts.ops.relu(w), output_dtype="fp32")


def _relu_bits(x):
    """The Tile contract on storage bits: NaN payloads kept, zeros and negatives to +0."""
    u = np.ascontiguousarray(x, np.float16).view(np.uint16)
    nan = (u & 0x7FFF) > 0x7C00
    clamp = ~nan & (((u & 0x8000) != 0) | ((u & 0x7FFF) == 0))
    return np.where(clamp, np.uint16(0), u).astype(np.uint16).view(np.float16)


def _special_f16(shape, seed):
    """Normal values plus every class the contract names: +/-0, +/-inf, NaN payloads."""
    x = _half(*shape, seed=seed).reshape(-1)
    specials = np.array([0x0000, 0x8000, 0x7C00, 0xFC00, 0x7E00, 0xFE01, 0x7C01, 0x0001, 0x8001],
                        np.uint16).view(np.float16)
    x[:specials.size] = specials
    return x.reshape(shape)


def test_relu_contract_reference_on_storage_bits():
    x = _special_f16((4, 8), seed=3)
    bits = _relu_bits(x).view(np.uint16).reshape(-1)
    assert list(bits[:9]) == [0, 0, 0x7C00, 0, 0x7E00, 0xFE01, 0x7C01, 0x0001, 0]


def test_the_synthesized_kernel_comes_from_the_tile_contract():
    from tessera.compiler.emit.apple_msl import synthesize_activation_msl

    source = synthesize_activation_msl("relu", "f16", nan_mode="propagate", zero_mode="positive")
    assert "kernel void synth_activation_relu_f16(" in source
    # Integer bit tests, never a float compare a fast-math compiler could fold.
    assert "max(" not in source and "half" not in source
    for contract in (("relu", "f16", "propagate", "preserve"), ("relu", "f32", "propagate", "positive"),
                     ("gelu", "f16", "propagate", "positive")):
        with pytest.raises(ValueError, match="no synthesized Apple activation"):
            synthesize_activation_msl(contract[0], contract[1], nan_mode=contract[2],
                                      zero_mode=contract[3])


def test_a_relu_member_packages_synthesized_msl():
    relu = atp.compile_apple_tensor_program(
        _graph(_relu_product, (_half(64, 32), _half(32, 48)), ("x", "w")))
    activation, product = relu.stages
    assert (activation.op_name, activation.carrier) == ("tessera.relu", "tile.activation_kernel")
    package = activation.package
    provenance = package.descriptor.provenance
    assert package.descriptor.abi_id == apple_native.APPLE_SYNTH_ACTIVATION_F16_ABI
    assert provenance["route"] == "apple_gpu_synth_activation_f16" and provenance["synthesized"]
    assert (provenance["nan_mode"], provenance["zero_mode"]) == ("propagate", "positive")
    # The Target IR carries the synthesized source its digest names.
    import hashlib

    source = package.target_ir.split("// ---- synthesized MSL ----\n", 1)[1]
    assert hashlib.sha256(source.encode()).hexdigest() == provenance["source_sha256"]
    assert 'zero_mode = "positive"' in activation.member_ir
    assert product.inputs == (activation.output, "w")
    relu.validate()


@pytest.mark.parametrize("fn, arrays", [
    # f32 storage has no apple7 ReLU (bf16 is refused in the lit fixture).
    (lambda x: ts.ops.relu(x), (np.zeros((8, 16), np.float32),)),
    # Rank one is outside the rank-two launch contract.
    (lambda x: ts.ops.relu(x), (np.zeros((16,), np.float16),)),
])
def test_scheduled_relu_outside_the_apple_contract_is_refused(fn, arrays):
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel

    module = to_graph_ir_module(trace(fn, *arrays, arg_names=("x",)), name="r", target="apple_gpu")
    with pytest.raises((ValueError, RuntimeError)):
        lower_scheduled_kernel(module, target="apple_gpu")


@pytest.mark.hardware_apple_gpu
def test_standalone_relu_package_runs_the_synthesized_kernel_bit_exactly():
    from tests._support.apple import require_apple_chip_identity
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel
    from tessera.runtime import RuntimeArtifact, launch

    require_apple_chip_identity()
    x = _special_f16((37, 53), seed=41)
    module = to_graph_ir_module(trace(lambda x: ts.ops.relu(x), x, arg_names=("x",)),
                                name="r", target="apple_gpu")
    kernel = lower_scheduled_kernel(module, target="apple_gpu")
    package = apple_native.package_scheduled_kernel(kernel, pipeline_name=atp.PIPELINE)
    out_name = package.descriptor.buffers[1].name
    artifact = RuntimeArtifact(
        metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
        target_ir=package.target_ir, native_image=package.image,
        launch_descriptor=package.descriptor)
    launched = launch(artifact, {"x": x, out_name: np.full(x.shape, np.nan, np.float16)})
    assert launched["ok"] and launched["execution_kind"] == "native_gpu"
    assert np.asarray(launched["output"]).view(np.uint16).tobytes() == _relu_bits(x).view(np.uint16).tobytes()


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("fn, shapes, members, private", [
    (_relu_product, ((64, 32), (32, 48)), 2, 1),
    (_relu_product, ((100, 37), (37, 53)), 2, 1),
    # A chain through all three producer kinds: ReLU -> GELU -> softmax -> product.
    (_relu_gelu_softmax_product, ((100, 37), (37, 53)), 4, 3),
    (_both_operands_relu, ((64, 32), (32, 48)), 3, 2),
])
def test_frame_owner_runs_relu_programs_in_one_command_buffer(fn, shapes, members, private):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    x, w = (_half(*shape, seed=i + 51) for i, shape in enumerate(shapes))
    program = atp.compile_apple_tensor_program(_graph(fn, (x, w), ("x", "w")))
    assert len(program.stages) == members
    output, receipt = atp.execute_frame(program, {"x": x, "w": w})
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, members)
    assert receipt["private_device_buffers"] == private
    sequential, _ = atp.run_members_sequentially(program, {"x": x, "w": w})
    assert output.dtype == np.float32 and output.tobytes() == sequential.tobytes()
    lhs = _relu_bits(x)
    if fn is _relu_gelu_softmax_product:
        lhs = _softmax_f16(_gelu_f16(lhs))
    rhs = _relu_bits(w) if fn is _both_operands_relu else w
    reference = lhs.astype(np.float32) @ rhs.astype(np.float32)
    np.testing.assert_allclose(output, reference, rtol=0, atol=2e-3)


@pytest.mark.hardware_apple_gpu
def test_frame_owner_refuses_a_synthesized_stage_without_its_source():
    import ctypes

    from tessera._apple_gpu_dispatch import bind_registered

    owner = bind_registered("tessera_apple_gpu_tensor_frame_execute")
    assert owner is not None
    x, out = np.zeros((4, 8), np.float16), np.full((4, 8), -1, np.float16)
    buffers = (ctypes.c_int64 * 4)(0, x.nbytes, 2, out.nbytes)
    hosts = (ctypes.c_void_p * 2)(x.ctypes.data, out.ctypes.data)
    stage = (ctypes.c_int64 * 10)(4, 0, -1, 1, 4, 1, 8, 0, 0, 0)
    none = (ctypes.c_char_p * 1)(None)
    receipt = (ctypes.c_int64 * 4)()
    assert owner(buffers, hosts, 2, stage, none, none, 1, receipt) == -1
    assert list(receipt) == [0, 0, 0, 0] and (out == -1).all()


@pytest.mark.hardware_apple_gpu
def test_public_jit_runs_a_relu_program_in_one_command_buffer():
    from tests._support.apple import require_apple_chip_identity
    from tessera import runtime as rt

    require_apple_chip_identity()
    x, w = _special_f16((100, 37), seed=61), _half(37, 53, seed=62)
    # Keep the product finite: the NaN/inf rows exercise the kernel, not the GEMM.
    x[0, :9] = np.float16(0)
    program = ts.jit(target="apple_gpu")(_relu_product)
    output = program(x, w)
    receipt = program._apple_tensor_last_receipt
    assert receipt["compiler_path"] == "canonical_apple_tensor_program"
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, 2)
    roots = [arg.name for arg in program._apple_tensor_last_program.graph.functions[0].args]
    framed, _ = atp.execute_frame(program._apple_tensor_last_program, dict(zip(roots, (x, w))))
    assert output.dtype == np.float32 and output.tobytes() == framed.tobytes()
    artifact = program.runtime_artifact()
    launched = rt.launch(artifact, {"x": x, "w": w})
    assert launched["ok"] and np.asarray(launched["output"]).tobytes() == output.tobytes()


# --- Slice 7: the f16 add as compiler-synthesized MSL -----------------------


def _add_product(x, y, w):
    return ts.ops.matmul(ts.ops.add(x, y), w, output_dtype="fp32")


def _residual_relu_product(x, w):
    # A residual: the add reads the program input and a private intermediate.
    return ts.ops.matmul(ts.ops.add(ts.ops.relu(x), x), w, output_dtype="fp32")


def _self_add_product(x, w):
    return ts.ops.matmul(ts.ops.add(x, x), w, output_dtype="fp32")


def _add_f16(a, b):
    """numpy's f16 add is the correctly rounded f16 sum the contract names."""
    with np.errstate(over="ignore", invalid="ignore"):
        return (np.asarray(a, np.float16) + np.asarray(b, np.float16)).astype(np.float16)


def _special_pair(shape, seed):
    """Normal values plus the classes the add contract names, paired so every
    rule fires: overflow to inf, inf + -inf, NaN operands, signed zeros and
    subnormal sums."""
    a = _half(*shape, seed=seed).reshape(-1)
    b = _half(*shape, seed=seed + 1).reshape(-1)
    pairs = [(65504, 65504), (-65504, -65504), (np.inf, -np.inf), (np.inf, 1), (np.nan, 1),
             (1, np.nan), (0.0, -0.0), (-0.0, -0.0), (6e-8, 6e-8), (6e-5, -6.1e-5),
             (2048, 1), (2048, 3)]
    for i, (x, y) in enumerate(pairs):
        a[i], b[i] = np.float16(x), np.float16(y)
    return a.reshape(shape), b.reshape(shape)


def _same_bits_up_to_nan_payload(got, want):
    got, want = np.asarray(got), np.asarray(want)
    nan = np.isnan(want)
    assert (np.isnan(got) == nan).all()
    assert got[~nan].view(np.uint16).tobytes() == want[~nan].view(np.uint16).tobytes()


def test_the_synthesized_add_comes_from_the_tile_contract():
    from tessera.compiler.emit.apple_msl import synthesize_binary_msl

    source = synthesize_binary_msl("add", "f16", nan_mode="propagate")
    assert "kernel void synth_binary_add_f16(" in source
    assert "float(in0[gid]) + float(in1[gid])" in source
    for kind, storage, nan in (("add", "f32", "propagate"), ("mul", "f16", "propagate"),
                               ("add", "f16", "suppress")):
        with pytest.raises(ValueError, match="no synthesized Apple binary kernel"):
            synthesize_binary_msl(kind, storage, nan_mode=nan)


def test_an_add_member_packages_synthesized_msl():
    program = atp.compile_apple_tensor_program(
        _graph(_add_product, (_half(64, 32), _half(64, 32), _half(32, 48)), ("x", "y", "w")))
    add, product = program.stages
    assert (add.op_name, add.carrier) == ("tessera.add", "tile.elementwise_kernel")
    descriptor = add.package.descriptor
    assert descriptor.abi_id == apple_native.APPLE_SYNTH_BINARY_F16_ABI
    assert descriptor.provenance["route"] == "apple_gpu_synth_binary_f16"
    assert [b.direction for b in descriptor.buffers] == ["input", "input", "output"]
    import hashlib

    source = add.package.target_ir.split("// ---- synthesized MSL ----\n", 1)[1]
    assert hashlib.sha256(source.encode()).hexdigest() == descriptor.provenance["source_sha256"]
    assert add.inputs == ("x", "y") and product.inputs == (add.output, "w")
    program.validate()


@pytest.mark.parametrize("fn, arrays", [
    # f32 storage has no apple7 add.
    (lambda a, b: ts.ops.add(a, b), (np.zeros((8, 16), np.float32),) * 2),
    # Rank one is outside the rank-two launch contract.
    (lambda a, b: ts.ops.add(a, b), (np.zeros((16,), np.float16),) * 2),
])
def test_scheduled_add_outside_the_apple_contract_is_refused(fn, arrays):
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel

    module = to_graph_ir_module(trace(fn, *arrays, arg_names=("a", "b")), name="s",
                                target="apple_gpu")
    with pytest.raises((ValueError, RuntimeError)):
        lower_scheduled_kernel(module, target="apple_gpu")


@pytest.mark.hardware_apple_gpu
def test_standalone_add_package_runs_the_synthesized_kernel():
    from tests._support.apple import require_apple_chip_identity
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel
    from tessera.runtime import RuntimeArtifact, launch

    require_apple_chip_identity()
    a, b = _special_pair((37, 53), seed=71)
    module = to_graph_ir_module(trace(lambda a, b: ts.ops.add(a, b), a, b, arg_names=("a", "b")),
                                name="s", target="apple_gpu")
    kernel = lower_scheduled_kernel(module, target="apple_gpu")
    package = apple_native.package_scheduled_kernel(kernel, pipeline_name=atp.PIPELINE)
    out_name = package.descriptor.buffers[2].name
    artifact = RuntimeArtifact(
        metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
        target_ir=package.target_ir, native_image=package.image,
        launch_descriptor=package.descriptor)
    launched = launch(artifact, {"a": a, "b": b, out_name: np.full(a.shape, -1, np.float16)})
    assert launched["ok"] and launched["execution_kind"] == "native_gpu"
    _same_bits_up_to_nan_payload(launched["output"], _add_f16(a, b))


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("fn, shapes, names, members, private", [
    (_add_product, ((64, 32), (64, 32), (32, 48)), ("x", "y", "w"), 2, 1),
    (_add_product, ((100, 37), (100, 37), (37, 53)), ("x", "y", "w"), 2, 1),
    (_residual_relu_product, ((100, 37), (37, 53)), ("x", "w"), 3, 2),
    (_self_add_product, ((64, 32), (32, 48)), ("x", "w"), 2, 1),
])
def test_frame_owner_runs_add_programs_in_one_command_buffer(fn, shapes, names, members, private):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    arrays = [_half(*shape, seed=i + 81) for i, shape in enumerate(shapes)]
    inputs = dict(zip(names, arrays))
    program = atp.compile_apple_tensor_program(_graph(fn, arrays, names))
    assert len(program.stages) == members
    output, receipt = atp.execute_frame(program, inputs)
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, members)
    assert receipt["private_device_buffers"] == private
    sequential, _ = atp.run_members_sequentially(program, inputs)
    assert output.dtype == np.float32 and output.tobytes() == sequential.tobytes()
    if fn is _add_product:
        lhs = _add_f16(inputs["x"], inputs["y"])
    elif fn is _residual_relu_product:
        lhs = _add_f16(_relu_bits(inputs["x"]), inputs["x"])
    else:
        lhs = _add_f16(inputs["x"], inputs["x"])
    reference = lhs.astype(np.float32) @ inputs["w"].astype(np.float32)
    np.testing.assert_allclose(output, reference, rtol=0, atol=4e-3)


@pytest.mark.hardware_apple_gpu
def test_frame_owner_refuses_an_add_stage_whose_rhs_is_unwritten():
    import ctypes

    from tessera._apple_gpu_dispatch import bind_registered
    from tessera.compiler.apple_native import synthesized_binary_source

    owner = bind_registered("tessera_apple_gpu_tensor_frame_execute")
    assert owner is not None
    source, entry = synthesized_binary_source("add", "f16", "propagate")
    x, out = np.zeros((4, 8), np.float16), np.full((4, 8), -1, np.float16)
    # Buffer 1 is a private intermediate no earlier stage wrote.
    buffers = (ctypes.c_int64 * 6)(0, x.nbytes, 1, x.nbytes, 2, out.nbytes)
    hosts = (ctypes.c_void_p * 3)(x.ctypes.data, None, out.ctypes.data)
    stage = (ctypes.c_int64 * 10)(4, 0, 1, 2, 4, 1, 8, 0, 0, 0)
    sources = (ctypes.c_char_p * 1)(source.encode())
    entries = (ctypes.c_char_p * 1)(entry.encode())
    receipt = (ctypes.c_int64 * 4)()
    assert owner(buffers, hosts, 3, stage, sources, entries, 1, receipt) == -1
    assert list(receipt) == [0, 0, 0, 0] and (out == -1).all()


@pytest.mark.hardware_apple_gpu
def test_public_jit_runs_an_add_program_in_one_command_buffer():
    from tests._support.apple import require_apple_chip_identity
    from tessera import runtime as rt

    require_apple_chip_identity()
    x, w = _half(100, 37, seed=91), _half(37, 53, seed=92)
    program = ts.jit(target="apple_gpu")(_residual_relu_product)
    output = program(x, w)
    receipt = program._apple_tensor_last_receipt
    assert receipt["compiler_path"] == "canonical_apple_tensor_program"
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, 3)
    roots = [arg.name for arg in program._apple_tensor_last_program.graph.functions[0].args]
    framed, _ = atp.execute_frame(program._apple_tensor_last_program, dict(zip(roots, (x, w))))
    assert output.dtype == np.float32 and output.tobytes() == framed.tobytes()
    artifact = program.runtime_artifact()
    launched = rt.launch(artifact, {"x": x, "w": w})
    assert launched["ok"] and np.asarray(launched["output"]).tobytes() == output.tobytes()


# --- Slice 8: bf16 programs ---------------------------------------------------

ml_dtypes = pytest.importorskip("ml_dtypes")
_BF16 = np.dtype(ml_dtypes.bfloat16)


def _bf(*shape, seed=0):
    return np.random.default_rng(seed).standard_normal(shape).astype(_BF16)


def _relu_bits_bf16(x):
    """The bf16 contract on bits: NaNs as loaded, negative nonzero to +0, zeros keep sign."""
    u = np.ascontiguousarray(x, _BF16).view(np.uint16)
    nan = (u & 0x7FFF) > 0x7F80
    clamp = ~nan & ((u & 0x8000) != 0) & ((u & 0x7FFF) != 0)
    return np.where(clamp, np.uint16(0), u).astype(np.uint16).view(_BF16)


def _add_bf16(a, b):
    """f32 sum rounded once to bf16: the correctly rounded bf16 sum."""
    with np.errstate(over="ignore", invalid="ignore"):
        return (np.asarray(a, np.float32) + np.asarray(b, np.float32)).astype(_BF16)


def _special_bf16(shape, seed):
    x = _bf(*shape, seed=seed).reshape(-1)
    specials = np.array([0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x7F81, 0x0001, 0x8001],
                        np.uint16).view(_BF16)
    x[:specials.size] = specials
    return x.reshape(shape)


def _bf16_relu_add_product(x, y, w):
    return ts.ops.matmul(ts.ops.add(ts.ops.relu(x), y), w, output_dtype="fp32")


def _bf16_relu_product(x, w):
    return ts.ops.matmul(ts.ops.relu(x), w, output_dtype="fp32")


def test_bf16_synthesized_kernels_name_their_contract():
    from tessera.compiler.emit.apple_msl import synthesize_activation_msl, synthesize_binary_msl

    relu = synthesize_activation_msl("relu", "bf16", nan_mode="propagate", zero_mode="preserve")
    assert "synth_activation_relu_bf16" in relu and "0x7f80u" in relu
    add = synthesize_binary_msl("add", "bf16", nan_mode="propagate")
    assert "device const bfloat* in0" in add and "bfloat(float(in0[gid]) + float(in1[gid]))" in add
    # bf16 keeps the zero sign; "positive" is f16's contract.
    with pytest.raises(ValueError, match="no synthesized Apple activation"):
        synthesize_activation_msl("relu", "bf16", nan_mode="propagate", zero_mode="positive")


def test_a_bf16_program_packages_every_member_at_bf16():
    program = atp.compile_apple_tensor_program(_graph(
        _bf16_relu_add_product, (_bf(64, 32), _bf(64, 32), _bf(32, 48)), ("x", "y", "w")))
    abis = [stage.package.descriptor.abi_id for stage in program.stages]
    assert abis == [apple_native.APPLE_SYNTH_ACTIVATION_BF16_ABI,
                    apple_native.APPLE_SYNTH_BINARY_BF16_ABI,
                    apple_native.APPLE_SIMDGROUP_GEMM_BF16_ABI]
    assert program.stages[0].package.descriptor.provenance["zero_mode"] == "preserve"
    assert [b["storage"] for b in program.frame["buffers"]][-1] == "f32"
    program.validate()


@pytest.mark.parametrize("fn", [
    lambda x, w: ts.ops.matmul(ts.ops.softmax(x, axis=-1), w, output_dtype="fp32"),
    lambda x, w: ts.ops.matmul(ts.ops.gelu(x), w, output_dtype="fp32"),
])
def test_bf16_softmax_and_gelu_members_are_refused(fn):
    # Their runtime routes round-trip through host f32 buffers: no on-device
    # bf16 kernel a frame with device-only intermediates could encode.
    with pytest.raises(RuntimeError, match="APPLE_TENSOR_PROGRAM_UNSUPPORTED: .*bf16 softmax and GELU"):
        atp.compile_apple_tensor_program(_graph(fn, (_bf(64, 32), _bf(32, 48)), ("x", "w")))


@pytest.mark.hardware_apple_gpu
def test_standalone_bf16_relu_and_add_are_bit_exact():
    from tests._support.apple import require_apple_chip_identity
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel
    from tessera.runtime import RuntimeArtifact, launch

    require_apple_chip_identity()

    def run(fn, arrays, names):
        module = to_graph_ir_module(trace(fn, *arrays, arg_names=names), name="s",
                                    target="apple_gpu")
        package = apple_native.package_scheduled_kernel(
            lower_scheduled_kernel(module, target="apple_gpu"), pipeline_name=atp.PIPELINE)
        out_name = package.descriptor.buffers[-1].name
        artifact = RuntimeArtifact(
            metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
            target_ir=package.target_ir, native_image=package.image,
            launch_descriptor=package.descriptor)
        buffers = dict(zip(names, arrays))
        buffers[out_name] = np.zeros(arrays[0].shape, _BF16)
        launched = launch(artifact, buffers)
        assert launched["ok"] and launched["execution_kind"] == "native_gpu"
        return np.asarray(launched["output"])

    x = _special_bf16((37, 53), seed=101)
    relu = run(lambda x: ts.ops.relu(x), (x,), ("x",))
    assert relu.view(np.uint16).tobytes() == _relu_bits_bf16(x).view(np.uint16).tobytes()
    a, b = _special_bf16((37, 53), seed=102), _bf(37, 53, seed=103)
    total = run(lambda a, b: ts.ops.add(a, b), (a, b), ("a", "b"))
    want = _add_bf16(a, b)
    nan = np.isnan(want.astype(np.float32))
    assert (np.isnan(total.astype(np.float32)) == nan).all()
    assert total[~nan].view(np.uint16).tobytes() == want[~nan].view(np.uint16).tobytes()


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("fn, shapes, names, members", [
    (_bf16_relu_add_product, ((64, 32), (64, 32), (32, 48)), ("x", "y", "w"), 3),
    (_bf16_relu_add_product, ((100, 37), (100, 37), (37, 53)), ("x", "y", "w"), 3),
    (_bf16_relu_product, ((64, 32), (32, 48)), ("x", "w"), 2),
])
def test_frame_owner_runs_bf16_programs_in_one_command_buffer(fn, shapes, names, members):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    arrays = [_bf(*shape, seed=i + 111) for i, shape in enumerate(shapes)]
    inputs = dict(zip(names, arrays))
    program = atp.compile_apple_tensor_program(_graph(fn, arrays, names))
    output, receipt = atp.execute_frame(program, inputs)
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, members)
    sequential, _ = atp.run_members_sequentially(program, inputs)
    assert output.dtype == np.float32 and output.tobytes() == sequential.tobytes()
    lhs = _relu_bits_bf16(inputs["x"])
    if "y" in inputs:
        lhs = _add_bf16(lhs, inputs["y"])
    reference = lhs.astype(np.float32) @ inputs["w"].astype(np.float32)
    # bf16 storage, f32 accumulation: the products are exact in f32, so only
    # the summation order differs from numpy.
    np.testing.assert_allclose(output, reference, rtol=0, atol=1e-3)


@pytest.mark.hardware_apple_gpu
def test_public_jit_runs_a_bf16_program_and_declines_bf16_softmax():
    from tests._support.apple import require_apple_chip_identity
    from tessera import runtime as rt

    require_apple_chip_identity()
    x, y, w = _bf(100, 37, seed=121), _bf(100, 37, seed=122), _bf(37, 53, seed=123)
    program = ts.jit(target="apple_gpu")(_bf16_relu_add_product)
    output = program(x, y, w)
    receipt = program._apple_tensor_last_receipt
    assert receipt["compiler_path"] == "canonical_apple_tensor_program"
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, 3)
    launched = rt.launch(program.runtime_artifact(), {"x": x, "y": y, "w": w})
    assert launched["ok"] and np.asarray(launched["output"]).tobytes() == output.tobytes()

    def bf16_softmax_product(x, w):
        return ts.ops.matmul(ts.ops.softmax(x, axis=-1), w, output_dtype="fp32")

    declined = ts.jit(target="apple_gpu")(bf16_softmax_product)
    declined(x, w)
    assert declined._apple_tensor_last_receipt is None
    assert set(declined._apple_tensor_program_cache.values()) <= {None}


# --- Slice 9: unweighted row norms as compiler-synthesized MSL -------------------


def _norm_reference(kind, x, eps):
    """The f32 contract: row statistics in f32, one rounding to storage after."""
    x = np.asarray(x).astype(np.float32)
    if kind == "layer_norm":
        mean = x.mean(-1, keepdims=True)
        return (x - mean) / np.sqrt(((x - mean) ** 2).mean(-1, keepdims=True) + eps)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def _within_one_storage_step(got, reference, dtype):
    """``got`` is the storage rounding of the f32 reference, or its neighbour."""
    rounded = reference.astype(dtype)
    bits = rounded.view(np.uint16).astype(np.int32)
    above = (bits + 1).astype(np.uint16).view(dtype).astype(np.float32)
    below = (bits - 1).astype(np.uint16).view(dtype).astype(np.float32)
    # fmax: the neighbour of a signed zero across the sign bit is a NaN pattern.
    with np.errstate(invalid="ignore"):
        step = np.fmax(np.abs(above - rounded.astype(np.float32)),
                       np.abs(rounded.astype(np.float32) - below))
    error = np.abs(np.asarray(got).astype(np.float32) - reference)
    assert np.isfinite(np.asarray(got).astype(np.float32)).all()
    assert (error <= step).all(), float((error - step).max())


def _rmsnorm_product(x, w):
    return ts.ops.matmul(ts.ops.rmsnorm(x), w, output_dtype="fp32")


def _layer_norm_product(x, w):
    return ts.ops.matmul(ts.ops.layer_norm(x, eps=1e-6), w, output_dtype="fp32")


def _residual_norm_product(x, w):
    # A pre-norm residual: the add reads the program input and the norm.
    return ts.ops.matmul(ts.ops.add(ts.ops.rmsnorm(x), x), w, output_dtype="fp32")


def test_the_synthesized_norm_comes_from_the_tile_contract():
    from tessera.compiler.emit.apple_msl import synthesize_norm_msl

    bits = apple_native._f32_bits(1e-5)
    source = synthesize_norm_msl("rmsnorm", "f16", epsilon_bits=bits)
    assert "kernel void synth_norm_rmsnorm_f16(" in source and "typedef half T;" in source
    assert f"as_type<float>(0x{bits:08x}u)" in source and "precise::sqrt" in source
    layer = synthesize_norm_msl("layernorm", "bf16", epsilon_bits=bits)
    assert "typedef bfloat T;" in layer and "- mean" in layer
    for kind, storage, eps in (("rmsnorm", "f32", bits), ("groupnorm", "f16", bits),
                               ("rmsnorm", "f16", apple_native._f32_bits(0.0)),
                               ("rmsnorm", "f16", apple_native._f32_bits(float("inf")))):
        with pytest.raises(ValueError, match="synthesized Apple norm"):
            synthesize_norm_msl(kind, storage, epsilon_bits=eps)


def _norm_abi_and_symbol(affine, storage, columns):
    """The ABI and symbol the Schedule's row-schedule cut selects (slice 11)."""
    if columns >= 16:
        return (apple_native.APPLE_SYNTH_SIMDGROUP_NORM_ABIS[(affine, storage)],
                apple_native.APPLE_SYNTH_SIMDGROUP_ROWS_SYMBOL, "simdgroup_32")
    if affine == "none":
        return (apple_native.APPLE_SYNTH_NORM_ABIS[storage],
                apple_native.APPLE_SYNTH_ROWS_SYMBOL, "serial")
    return (apple_native.APPLE_SYNTH_AFFINE_NORM_ABIS[(affine, storage)],
            apple_native.APPLE_SYNTH_AFFINE_ROWS_SYMBOL, "serial")


@pytest.mark.parametrize("columns", [8, 32])
@pytest.mark.parametrize("fn, kind", [
    (_rmsnorm_product, "rmsnorm"),
    (_layer_norm_product, "layernorm"),
])
def test_a_norm_member_packages_synthesized_msl(fn, kind, columns):
    import hashlib

    program = atp.compile_apple_tensor_program(
        _graph(fn, (_half(64, columns), _half(columns, 48)), ("x", "w")))
    norm, product = program.stages
    assert norm.carrier == "tile.norm_kernel"
    descriptor = norm.package.descriptor
    abi, symbol, schedule = _norm_abi_and_symbol("none", "f16", columns)
    assert (descriptor.abi_id, descriptor.entry_symbol) == (abi, symbol)
    assert descriptor.provenance["schedule"] == schedule
    assert (descriptor.provenance["kind"], descriptor.provenance["route"]) == (
        kind, "apple_gpu_synth_norm_f16")
    source = norm.package.target_ir.split("// ---- synthesized MSL ----\n", 1)[1]
    assert hashlib.sha256(source.encode()).hexdigest() == descriptor.provenance["source_sha256"]
    eps = 1e-6 if kind == "layernorm" else 1e-5
    assert descriptor.provenance["epsilon_bits"] == apple_native._f32_bits(eps)
    assert product.inputs == (norm.output, "w")
    program.validate()


@pytest.mark.parametrize("columns", [8, 32])
def test_a_bf16_norm_member_packages_at_bf16(columns):
    program = atp.compile_apple_tensor_program(
        _graph(_rmsnorm_product, (_bf(64, columns), _bf(columns, 48)), ("x", "w")))
    assert [stage.package.descriptor.abi_id for stage in program.stages] == [
        _norm_abi_and_symbol("none", "bf16", columns)[0],
        apple_native.APPLE_SIMDGROUP_GEMM_BF16_ABI]


def test_a_norm_package_refuses_a_tile_launch_with_another_epsilon():
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel

    x = _half(8, 16)
    module = to_graph_ir_module(trace(lambda x: ts.ops.rmsnorm(x), x, arg_names=("x",)),
                                name="s", target="apple_gpu")
    kernel = lower_scheduled_kernel(module, target="apple_gpu")
    drifted = dataclasses.replace(kernel, epsilon=2e-5)
    with pytest.raises(ValueError):
        apple_native.package_scheduled_kernel(drifted, pipeline_name=atp.PIPELINE)


@pytest.mark.parametrize("fn, arrays", [
    # f32 storage has no apple7 synthesized norm.
    (lambda x: ts.ops.rmsnorm(x), (np.zeros((8, 16), np.float32),)),
    # Rank three is outside the rank-two launch contract.
    (lambda x: ts.ops.rmsnorm(x), (np.zeros((2, 8, 16), np.float16),)),
    # A numeric-policy request selects nothing the kernel computes.
    (lambda x: ts.ops.rmsnorm(x, numeric_policy={"accum": "fp32"}), (np.zeros((8, 16), np.float16),)),
])
def test_scheduled_norm_outside_the_apple_contract_is_refused(fn, arrays):
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel

    module = to_graph_ir_module(trace(fn, *arrays, arg_names=("x",)), name="s",
                                target="apple_gpu")
    with pytest.raises((ValueError, RuntimeError)):
        lower_scheduled_kernel(module, target="apple_gpu")


def test_an_f32_norm_member_is_refused():
    # Slice 10 admits weighted norms; f32 storage still has no apple7 kernel.
    def weighted(x, g, w):
        return ts.ops.matmul(ts.ops.rmsnorm(x, g), w, output_dtype="fp32")

    f32 = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
    with pytest.raises(RuntimeError, match="APPLE_TENSOR_PROGRAM_UNSUPPORTED"):
        atp.compile_apple_tensor_program(
            _graph(weighted, (f32(64, 32), f32(32), f32(32, 48)), ("x", "g", "w")))


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("columns", [11, 53])
@pytest.mark.parametrize("kind", ["rmsnorm", "layer_norm"])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_standalone_norm_rounds_the_f32_contract_once(kind, dtype, columns):
    from tests._support.apple import require_apple_chip_identity
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel
    from tessera.runtime import RuntimeArtifact, launch

    require_apple_chip_identity()
    storage = np.dtype(np.float16) if dtype == "f16" else _BF16
    rng = np.random.default_rng(131)
    # An offset mean, a wide scale, a constant row (layer_norm's eps-only
    # variance) and an all-zero row (rmsnorm's eps-only mean square). 11
    # columns keep the serial row schedule; 53 take simdgroup_32 (slice 11),
    # with a ragged last lane stride.
    x = (rng.standard_normal((37, columns)) * 3 + 0.5).astype(storage)
    x[3] = storage.type(2.0)
    x[4] = storage.type(0.0)
    op = getattr(ts.ops, kind)
    module = to_graph_ir_module(trace(lambda x: op(x, eps=1e-5), x, arg_names=("x",)),
                                name="s", target="apple_gpu")
    package = apple_native.package_scheduled_kernel(
        lower_scheduled_kernel(module, target="apple_gpu"), pipeline_name=atp.PIPELINE)
    assert package.descriptor.provenance["schedule"] == (
        "simdgroup_32" if columns >= 16 else "serial")
    out_name = package.descriptor.buffers[-1].name
    artifact = RuntimeArtifact(
        metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
        target_ir=package.target_ir, native_image=package.image,
        launch_descriptor=package.descriptor)
    launched = launch(artifact, {"x": x, out_name: np.full(x.shape, np.nan, storage)})
    assert launched["ok"] and launched["execution_kind"] == "native_gpu"
    _within_one_storage_step(launched["output"], _norm_reference(kind, x, 1e-5), storage)


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("fn, shapes, members", [
    (_rmsnorm_product, ((64, 32), (32, 48)), 2),
    (_layer_norm_product, ((100, 37), (37, 53)), 2),
    (_residual_norm_product, ((100, 37), (37, 53)), 3),
    # 8 columns keep the serial row schedule (slice 11's cut is 16).
    (_layer_norm_product, ((100, 8), (8, 53)), 2),
])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_frame_owner_runs_norm_programs_in_one_command_buffer(fn, shapes, members, dtype):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    make = _half if dtype == "f16" else _bf
    arrays = [make(*shape, seed=i + 141) for i, shape in enumerate(shapes)]
    inputs = dict(zip(("x", "w"), arrays))
    program = atp.compile_apple_tensor_program(_graph(fn, arrays, ("x", "w")))
    output, receipt = atp.execute_frame(program, inputs)
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, members)
    assert receipt["private_device_buffers"] == members - 1
    sequential, _ = atp.run_members_sequentially(program, inputs)
    assert output.dtype == np.float32 and output.tobytes() == sequential.tobytes()


@pytest.mark.hardware_apple_gpu
def test_frame_owner_refuses_a_norm_stage_without_its_kernel_or_extent():
    import ctypes

    from tessera._apple_gpu_dispatch import bind_registered

    owner = bind_registered("tessera_apple_gpu_tensor_frame_execute")
    assert owner is not None
    x, out = np.zeros((4, 8), np.float16), np.full((4, 8), -1, np.float16)
    hosts = (ctypes.c_void_p * 2)(x.ctypes.data, out.ctypes.data)
    none = (ctypes.c_char_p * 1)(None)
    receipt = (ctypes.c_int64 * 4)()
    # Kind 5 carries its synthesized source; a stage without one is refused.
    buffers = (ctypes.c_int64 * 4)(0, x.nbytes, 2, out.nbytes)
    stage = (ctypes.c_int64 * 10)(5, 0, -1, 1, 4, 1, 8, 0, 0, 0)
    assert owner(buffers, hosts, 2, stage, none, none, 1, receipt) == -1
    # With its kernel, an output one element short is refused.
    source, entry = apple_native.synthesized_norm_source(
        "rmsnorm", "f16", apple_native._f32_bits(1e-5))
    sources = (ctypes.c_char_p * 1)(source.encode())
    entries = (ctypes.c_char_p * 1)(entry.encode())
    short = (ctypes.c_int64 * 4)(0, x.nbytes, 2, out.nbytes - 2)
    assert owner(short, hosts, 2, stage, sources, entries, 1, receipt) == -1
    assert list(receipt) == [0, 0, 0, 0] and (out == -1).all()


@pytest.mark.hardware_apple_gpu
def test_public_jit_runs_a_norm_program_and_declines_an_f32_norm():
    from tests._support.apple import require_apple_chip_identity
    from tessera import runtime as rt

    require_apple_chip_identity()
    x, w = _half(100, 37, seed=151), _half(37, 53, seed=152)
    program = ts.jit(target="apple_gpu")(_residual_norm_product)
    output = program(x, w)
    receipt = program._apple_tensor_last_receipt
    assert receipt["compiler_path"] == "canonical_apple_tensor_program"
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, 3)
    launched = rt.launch(program.runtime_artifact(), {"x": x, "w": w})
    assert launched["ok"] and np.asarray(launched["output"]).tobytes() == output.tobytes()

    def weighted(x, g, w):
        return ts.ops.matmul(ts.ops.rmsnorm(x, g), w, output_dtype="fp32")

    # f32 storage has no apple7 norm kernel: the program route declines.
    declined = ts.jit(target="apple_gpu")(weighted)
    declined(x.astype(np.float32), _half(37, seed=153).astype(np.float32), w.astype(np.float32))
    assert declined._apple_tensor_last_receipt is None
    assert set(declined._apple_tensor_program_cache.values()) <= {None}


# --- Slice 10: weighted (gamma / gamma+beta) row norms --------------------------


def _affine_reference(kind, x, eps, gamma, beta=None):
    """The f32 contract: gamma (and beta) applied in f32 before the one rounding."""
    y = _norm_reference(kind, x, eps) * np.asarray(gamma).astype(np.float32)
    if beta is not None:
        # The kernel's fma(y, gamma, beta): one f32 rounding of y*gamma+beta.
        y = (_norm_reference(kind, x, eps).astype(np.float64)
             * np.asarray(gamma).astype(np.float64)
             + np.asarray(beta).astype(np.float64)).astype(np.float32)
    return y


def _weighted_rmsnorm_product(x, g, w):
    return ts.ops.matmul(ts.ops.rmsnorm(x, g), w, output_dtype="fp32")


def _weighted_layer_norm_product(x, g, b, w):
    return ts.ops.matmul(ts.ops.layer_norm(x, g, b, eps=1e-6), w, output_dtype="fp32")


def _gamma_layer_norm_product(x, g, w):
    return ts.ops.matmul(ts.ops.layer_norm(x, g), w, output_dtype="fp32")


def _beta_layer_norm_product(x, b, w):
    return ts.ops.matmul(ts.ops.layer_norm(x, beta=b), w, output_dtype="fp32")


def _weighted_residual_product(x, g, b, w):
    # A pre-norm residual block: add(layer_norm(x; gamma, beta), x) @ w.
    return ts.ops.matmul(ts.ops.add(ts.ops.layer_norm(x, g, b), x), w, output_dtype="fp32")


def test_the_synthesized_weighted_norm_names_its_affine_contract():
    from tessera.compiler.emit.apple_msl import norm_entry, synthesize_norm_msl

    bits = apple_native._f32_bits(1e-5)
    plain = synthesize_norm_msl("rmsnorm", "f16", epsilon_bits=bits)
    # The unweighted kernel is byte-identical to slice 9's (no new buffers).
    assert "buffer(4)" not in plain and "affine=false" in plain
    gamma = synthesize_norm_msl("rmsnorm", "f16", epsilon_bits=bits, affine="gamma")
    assert norm_entry("rmsnorm", "f16", "gamma") == "synth_norm_rmsnorm_gamma_f16"
    assert "kernel void synth_norm_rmsnorm_gamma_f16(" in gamma
    assert "G    [[buffer(4)]]" in gamma and "buffer(5)" not in gamma
    assert "o[c] = T(y * float(G[c]));" in gamma and "affine=true bias=false" in gamma
    both = synthesize_norm_msl("layernorm", "bf16", epsilon_bits=bits, affine="gamma_beta")
    assert "B    [[buffer(5)]]" in both and "fma(y, float(G[c]), float(B[c]))" in both
    assert "affine=true bias=true" in both and "typedef bfloat T;" in both
    # The weighted forms are rmsnorm(gamma) and layernorm(gamma, beta); a
    # gamma-only layernorm does not decode, and an unknown form never falls back.
    for kind, affine in (("rmsnorm", "gamma_beta"), ("layernorm", "gamma"),
                         ("layernorm", "beta")):
        with pytest.raises(ValueError, match="synthesized Apple norm"):
            synthesize_norm_msl(kind, "f16", epsilon_bits=bits, affine=affine)


@pytest.mark.parametrize("columns", [8, 32])
@pytest.mark.parametrize("fn, names, kind, affine", [
    (_weighted_rmsnorm_product, ("x", "g", "w"), "rmsnorm", "gamma"),
    (_weighted_layer_norm_product, ("x", "g", "b", "w"), "layernorm", "gamma_beta"),
])
def test_a_weighted_norm_member_packages_synthesized_msl(fn, names, kind, affine, columns):
    import hashlib

    vectors = tuple(_half(columns) for _ in names[1:-1])
    arrays = (_half(64, columns), *vectors, _half(columns, 48))
    program = atp.compile_apple_tensor_program(_graph(fn, arrays, names))
    norm, product = program.stages
    assert norm.carrier == "tile.norm_kernel"
    descriptor = norm.package.descriptor
    abi, symbol, schedule = _norm_abi_and_symbol(affine, "f16", columns)
    assert (descriptor.abi_id, descriptor.entry_symbol) == (abi, symbol)
    assert descriptor.provenance["schedule"] == schedule
    assert (descriptor.provenance["kind"], descriptor.provenance["affine"]) == (kind, affine)
    # Bindings in Tile operand order: x, gamma[, beta], o; vectors are rank one.
    params = names[1:-1]
    assert [(b.name, b.direction, b.rank) for b in descriptor.buffers][:-1] == (
        [("x", "input", 2)] + [(p, "input", 1) for p in params])
    source = norm.package.target_ir.split("// ---- synthesized MSL ----\n", 1)[1]
    assert hashlib.sha256(source.encode()).hexdigest() == descriptor.provenance["source_sha256"]
    assert norm.inputs == names[:-1] and product.inputs == (norm.output, "w")
    program.validate()


def test_a_bf16_weighted_norm_member_packages_at_bf16():
    program = atp.compile_apple_tensor_program(_graph(
        _weighted_layer_norm_product, (_bf(64, 32), _bf(32), _bf(32), _bf(32, 48)),
        ("x", "g", "b", "w")))
    assert [stage.package.descriptor.abi_id for stage in program.stages] == [
        apple_native.APPLE_SYNTH_SIMDGROUP_NORM_ABIS[("gamma_beta", "bf16")],
        apple_native.APPLE_SIMDGROUP_GEMM_BF16_ABI]


def test_a_weighted_norm_package_refuses_a_dropped_affine_binding():
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel

    x, g, b = _half(8, 16), _half(16), _half(16)
    module = to_graph_ir_module(
        trace(lambda x, g, b: ts.ops.layer_norm(x, g, b), x, g, b, arg_names=("x", "g", "b")),
        name="s", target="apple_gpu")
    kernel = lower_scheduled_kernel(module, target="apple_gpu")
    assert kernel.input_names == ("x", "g", "b")
    # The native IR carries beta; a package that drops its binding is refused,
    # as is one whose epsilon drifted from the Tile launch.
    for drifted in (dataclasses.replace(kernel, input_names=("x", "g")),
                    dataclasses.replace(kernel, epsilon=2e-5)):
        with pytest.raises(ValueError):
            apple_native.package_scheduled_kernel(drifted, pipeline_name=atp.PIPELINE)


def test_a_weighted_norm_outside_the_apple_contract_is_refused():
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel

    # f32 storage has no apple7 weighted norm kernel.
    x, g = np.zeros((8, 16), np.float32), np.zeros(16, np.float32)
    module = to_graph_ir_module(trace(lambda x, g: ts.ops.rmsnorm(x, g), x, g,
                                      arg_names=("x", "g")), name="s", target="apple_gpu")
    with pytest.raises((ValueError, RuntimeError)):
        lower_scheduled_kernel(module, target="apple_gpu")


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("columns", [11, 53])
@pytest.mark.parametrize("kind, affine", [
    ("rmsnorm", "gamma"), ("layer_norm", "gamma_beta")])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_standalone_weighted_norm_rounds_the_f32_contract_once(kind, affine, dtype, columns):
    from tests._support.apple import require_apple_chip_identity
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel
    from tessera.runtime import RuntimeArtifact, launch

    require_apple_chip_identity()
    storage = np.dtype(np.float16) if dtype == "f16" else _BF16
    rng = np.random.default_rng(161)
    x = (rng.standard_normal((37, columns)) * 3 + 0.5).astype(storage)
    x[3] = storage.type(2.0)
    x[4] = storage.type(0.0)
    gamma = (rng.standard_normal(columns) * 2).astype(storage)
    gamma[5] = storage.type(0.0)
    beta = rng.standard_normal(columns).astype(storage) if affine == "gamma_beta" else None
    op = getattr(ts.ops, kind)
    arrays = (x, gamma) if beta is None else (x, gamma, beta)
    names = ("x", "g") if beta is None else ("x", "g", "b")
    fn = ((lambda x, g: op(x, g, eps=1e-5)) if beta is None
          else (lambda x, g, b: op(x, g, b, eps=1e-5)))
    module = to_graph_ir_module(trace(fn, *arrays, arg_names=names), name="s",
                                target="apple_gpu")
    package = apple_native.package_scheduled_kernel(
        lower_scheduled_kernel(module, target="apple_gpu"), pipeline_name=atp.PIPELINE)
    assert package.descriptor.provenance["affine"] == affine
    assert package.descriptor.provenance["schedule"] == (
        "simdgroup_32" if columns >= 16 else "serial")
    out_name = package.descriptor.buffers[-1].name
    artifact = RuntimeArtifact(
        metadata={"target": "apple_gpu", "compiler_path": "apple_native_descriptor"},
        target_ir=package.target_ir, native_image=package.image,
        launch_descriptor=package.descriptor)
    launched = launch(artifact, {**dict(zip(names, arrays)),
                                 out_name: np.full(x.shape, np.nan, storage)})
    assert launched["ok"] and launched["execution_kind"] == "native_gpu"
    reference = _affine_reference(kind, x, 1e-5, gamma, beta)
    _within_one_storage_step(launched["output"], reference, storage)


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("fn, shapes, names, members", [
    (_weighted_rmsnorm_product, ((64, 32), (32,), (32, 48)), ("x", "g", "w"), 2),
    (_weighted_rmsnorm_product, ((64, 8), (8,), (8, 48)), ("x", "g", "w"), 2),
    (_weighted_layer_norm_product, ((100, 37), (37,), (37,), (37, 53)), ("x", "g", "b", "w"), 2),
    (_weighted_residual_product, ((100, 37), (37,), (37,), (37, 53)), ("x", "g", "b", "w"), 3),
])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_frame_owner_runs_weighted_norm_programs_in_one_command_buffer(
        fn, shapes, names, members, dtype):
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    make = _half if dtype == "f16" else _bf
    arrays = [make(*shape, seed=i + 171) for i, shape in enumerate(shapes)]
    inputs = dict(zip(names, arrays))
    program = atp.compile_apple_tensor_program(_graph(fn, arrays, names))
    output, receipt = atp.execute_frame(program, inputs)
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, members)
    assert receipt["private_device_buffers"] == members - 1
    sequential, _ = atp.run_members_sequentially(program, inputs)
    assert output.dtype == np.float32 and output.tobytes() == sequential.tobytes()


@pytest.mark.hardware_apple_gpu
def test_frame_owner_refuses_a_weighted_norm_stage_outside_its_contract():
    import ctypes

    from tessera._apple_gpu_dispatch import bind_registered

    owner = bind_registered("tessera_apple_gpu_tensor_frame_execute")
    assert owner is not None
    x, g, b = np.zeros((4, 8), np.float16), np.ones(8, np.float16), np.zeros(8, np.float16)
    out = np.full((4, 8), -1, np.float16)
    hosts = (ctypes.c_void_p * 4)(x.ctypes.data, g.ctypes.data, b.ctypes.data, out.ctypes.data)
    source, entry = apple_native.synthesized_norm_source(
        "layernorm", "f16", apple_native._f32_bits(1e-5), "gamma_beta")
    sources = (ctypes.c_char_p * 1)(source.encode())
    entries = (ctypes.c_char_p * 1)(entry.encode())
    none = (ctypes.c_char_p * 1)(None)
    receipt = (ctypes.c_int64 * 4)()
    buffers = (ctypes.c_int64 * 8)(0, x.nbytes, 0, g.nbytes, 0, b.nbytes, 2, out.nbytes)

    def stage(*fields):
        return (ctypes.c_int64 * 10)(*fields)

    good = stage(6, 0, 1, 3, 4, 1, 8, 2, 0, 0)
    # Without its kernel; gamma one element short; beta naming the output;
    # a nonzero reserved slot.
    assert owner(buffers, hosts, 4, good, none, none, 1, receipt) == -1
    short = (ctypes.c_int64 * 8)(0, x.nbytes, 0, g.nbytes - 2, 0, b.nbytes, 2, out.nbytes)
    assert owner(short, hosts, 4, good, sources, entries, 1, receipt) == -1
    assert owner(buffers, hosts, 4, stage(6, 0, 1, 3, 4, 1, 8, 3, 0, 0),
                 sources, entries, 1, receipt) == -1
    assert owner(buffers, hosts, 4, stage(6, 0, 1, 3, 4, 1, 8, 2, 1, 0),
                 sources, entries, 1, receipt) == -1
    assert list(receipt) == [0, 0, 0, 0] and (out == -1).all()
    # The same stage, in contract, runs: layer_norm of zeros is beta (zero).
    assert owner(buffers, hosts, 4, good, sources, entries, 1, receipt) == 1
    assert (out == 0).all() and list(receipt)[:3] == [1, 1, 0]


@pytest.mark.hardware_apple_gpu
def test_public_jit_runs_a_weighted_norm_program():
    from tests._support.apple import require_apple_chip_identity
    from tessera import runtime as rt

    require_apple_chip_identity()
    x, g, b, w = (_half(100, 37, seed=181), _half(37, seed=182), _half(37, seed=183),
                  _half(37, 53, seed=184))
    program = ts.jit(target="apple_gpu")(_weighted_residual_product)
    output = program(x, g, b, w)
    receipt = program._apple_tensor_last_receipt
    assert receipt["compiler_path"] == "canonical_apple_tensor_program"
    assert (receipt["command_buffers"], receipt["dispatches"]) == (1, 3)
    launched = rt.launch(program.runtime_artifact(), {"x": x, "g": g, "b": b, "w": w})
    assert launched["ok"] and np.asarray(launched["output"]).tobytes() == output.tobytes()


@pytest.mark.hardware_apple_gpu
def test_public_jit_declines_a_vector_input_that_is_not_a_norm_weight():
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()

    def unused_vector(x, v, w):
        return ts.ops.matmul(ts.ops.rmsnorm(x), w, output_dtype="fp32")

    program = ts.jit(target="apple_gpu")(unused_vector)
    program(_half(16, 8, seed=191), _half(8, seed=192), _half(8, 4, seed=193))
    assert program._apple_tensor_last_receipt is None
    assert set(program._apple_tensor_program_cache.values()) <= {None}



@pytest.mark.parametrize("fn", [_gamma_layer_norm_product, _beta_layer_norm_product])
def test_a_two_operand_layer_norm_member_is_refused(fn):
    # The tracer drops keyword names: layer_norm(x, beta=b) and
    # layer_norm(x, gamma) both arrive as (x, v), so neither may be read as a
    # gamma (tessera.layer_norm is in the undecodable-operand ratchet).
    program = _graph(fn, (_half(64, 32), _half(32), _half(32, 48)), ("x", "v", "w"))
    assert len(program.functions[0].body[0].operands) == 2
    with pytest.raises(RuntimeError, match="APPLE_TENSOR_PROGRAM_UNSUPPORTED: an Apple weighted "
                                           "norm member is rmsnorm"):
        atp.compile_apple_tensor_program(program)


def test_an_isolated_two_operand_layer_norm_is_not_lowered_as_weighted():
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel

    x, b = _half(8, 16), _half(16)
    module = to_graph_ir_module(trace(lambda x, b: ts.ops.layer_norm(x, beta=b), x, b,
                                      arg_names=("x", "b")), name="s", target="apple_gpu")
    with pytest.raises(ValueError, match="layer_norm\\(x, gamma, beta\\)"):
        lower_scheduled_kernel(module, target="apple_gpu")


@pytest.mark.hardware_apple_gpu
def test_public_jit_declines_a_beta_only_layer_norm_program():
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    program = ts.jit(target="apple_gpu")(_beta_layer_norm_product)
    program(_half(16, 8, seed=201), _half(8, seed=202), _half(8, 4, seed=203))
    assert program._apple_tensor_last_receipt is None
    assert set(program._apple_tensor_program_cache.values()) <= {None}


# --- Slice 11: the simdgroup-per-row norm schedule ----------------------------


def test_the_simdgroup_norm_is_the_serial_body_lane_strided():
    from tessera.compiler.emit.apple_msl import NORM_SCHEDULES, norm_entry, synthesize_norm_msl

    bits = apple_native._f32_bits(1e-5)
    assert NORM_SCHEDULES == ("serial", "simdgroup_32")
    for kind, affine, reductions in (("rmsnorm", "none", 1), ("layernorm", "none", 2),
                                     ("rmsnorm", "gamma", 1), ("layernorm", "gamma_beta", 2)):
        serial = synthesize_norm_msl(kind, "f16", epsilon_bits=bits, affine=affine)
        simd = synthesize_norm_msl(kind, "f16", epsilon_bits=bits, affine=affine,
                                   schedule="simdgroup_32")
        entry = norm_entry(kind, "f16", affine, "simdgroup_32")
        assert entry == norm_entry(kind, "f16", affine) + "_simdgroup_32"
        assert f"kernel void {entry}(" in simd and "schedule=simdgroup_32" in simd
        # Every column loop is lane-strided, and each f32 accumulator is
        # combined across the simdgroup before it is read.
        assert "for (uint c = 0; c < C; ++c)" not in simd
        assert simd.count("simd_sum(") == reductions
        assert "[[thread_index_in_simdgroup]]" in simd and "tid / 32u" in simd
        assert "simd_sum" not in serial and "schedule=serial" in serial
    with pytest.raises(ValueError, match="synthesized Apple norm"):
        synthesize_norm_msl("rmsnorm", "f16", epsilon_bits=bits, schedule="cooperative_128")


@pytest.mark.parametrize("columns, schedule, workgroup", [
    (15, "serial", 1), (16, "simdgroup_32", 32), (4096, "simdgroup_32", 32)])
def test_the_schedule_picks_the_row_schedule_at_sixteen_columns(columns, schedule, workgroup):
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel

    x = _half(4, columns)
    module = to_graph_ir_module(trace(lambda x: ts.ops.rmsnorm(x), x, arg_names=("x",)),
                                name="s", target="apple_gpu")
    kernel = lower_scheduled_kernel(module, target="apple_gpu")
    assert (kernel.schedule, kernel.workgroup_size) == (schedule, workgroup)
    package = apple_native.package_scheduled_kernel(kernel, pipeline_name=atp.PIPELINE)
    assert package.descriptor.abi_id == _norm_abi_and_symbol("none", "f16", columns)[0]
    assert package.descriptor.geometry.policy == (
        "apple_msl_synth_simdgroup_rows" if schedule == "simdgroup_32"
        else "apple_msl_synth_rows")


def test_a_simdgroup_norm_package_refuses_a_tile_launch_without_its_schedule():
    from tessera.compiler.scheduled_kernel import lower_scheduled_kernel

    x = _half(4, 64)
    module = to_graph_ir_module(trace(lambda x: ts.ops.rmsnorm(x), x, arg_names=("x",)),
                                name="s", target="apple_gpu")
    kernel = lower_scheduled_kernel(module, target="apple_gpu")
    assert 'schedule = "simdgroup_32"' in kernel.tile_ir
    for tile_ir in (kernel.tile_ir.replace('schedule = "simdgroup_32", ', ""),
                    kernel.tile_ir.replace('schedule = "simdgroup_32"', 'schedule = "serial"')):
        with pytest.raises(ValueError):
            apple_native.package_scheduled_kernel(
                dataclasses.replace(kernel, tile_ir=tile_ir), pipeline_name=atp.PIPELINE)


@pytest.mark.hardware_apple_gpu
def test_the_simdgroup_row_abis_refuse_a_lane_count_their_entry_does_not_name():
    import ctypes

    from tessera._apple_gpu_dispatch import bind_registered
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    simd = bind_registered(apple_native.APPLE_SYNTH_SIMDGROUP_ROWS_SYMBOL)
    serial = bind_registered(apple_native.APPLE_SYNTH_ROWS_SYMBOL)
    owner = bind_registered("tessera_apple_gpu_tensor_frame_execute")
    assert simd is not None and serial is not None and owner is not None
    bits = apple_native._f32_bits(1e-5)
    x, g = _half(4, 64, seed=191), _half(64, seed=192)
    out = np.full((4, 64), -1, np.float16)
    simd_src, simd_entry = apple_native.synthesized_norm_source(
        "rmsnorm", "f16", bits, "none", "simdgroup_32")
    serial_src, serial_entry = apple_native.synthesized_norm_source("rmsnorm", "f16", bits)
    ptr = lambda a: a.ctypes.data  # noqa: E731
    # Each ABI refuses the other schedule's kernel, and beta without gamma.
    assert simd(serial_src.encode(), serial_entry.encode(), ptr(x), None, None, ptr(out), 4, 64) == 0
    assert serial(simd_src.encode(), simd_entry.encode(), ptr(x), ptr(out), 4, 64) == 0
    assert simd(simd_src.encode(), simd_entry.encode(), ptr(x), None, ptr(g), ptr(out), 4, 64) == 0
    assert (out == -1).all()
    # In contract it runs, deterministically, within one f16 step of f32.
    assert simd(simd_src.encode(), simd_entry.encode(), ptr(x), None, None, ptr(out), 4, 64) == 1
    again = np.full_like(out, -1)
    assert simd(simd_src.encode(), simd_entry.encode(), ptr(x), None, None, ptr(again), 4, 64) == 1
    assert again.tobytes() == out.tobytes()
    _within_one_storage_step(out, _norm_reference("rmsnorm", x, 1e-5), np.dtype(np.float16))

    # The frame owner: kind 5 lanes are 0 or 32, named by the entry, with the
    # bm/bn slots zero.
    frame_out = np.full((4, 64), -1, np.float16)
    hosts = (ctypes.c_void_p * 2)(x.ctypes.data, frame_out.ctypes.data)
    buffers = (ctypes.c_int64 * 4)(0, x.nbytes, 2, frame_out.nbytes)
    receipt = (ctypes.c_int64 * 4)()
    simd_sources = (ctypes.c_char_p * 1)(simd_src.encode())
    simd_entries = (ctypes.c_char_p * 1)(simd_entry.encode())
    serial_sources = (ctypes.c_char_p * 1)(serial_src.encode())
    serial_entries = (ctypes.c_char_p * 1)(serial_entry.encode())
    for lanes, bm, srcs, ents in ((16, 0, simd_sources, simd_entries),
                                  (0, 0, simd_sources, simd_entries),
                                  (32, 0, serial_sources, serial_entries),
                                  (32, 1, simd_sources, simd_entries)):
        stage = (ctypes.c_int64 * 10)(5, 0, -1, 1, 4, 1, 64, bm, 0, lanes)
        assert owner(buffers, hosts, 2, stage, srcs, ents, 1, receipt) == -1
    assert list(receipt) == [0, 0, 0, 0] and (frame_out == -1).all()
    good = (ctypes.c_int64 * 10)(5, 0, -1, 1, 4, 1, 64, 0, 0, 32)
    assert owner(buffers, hosts, 2, good, simd_sources, simd_entries, 1, receipt) == 1
    assert frame_out.tobytes() == out.tobytes()


@pytest.mark.hardware_apple_gpu
def test_public_jit_runs_the_simdgroup_and_serial_norm_schedules():
    from tests._support.apple import require_apple_chip_identity

    require_apple_chip_identity()
    for columns, schedule in ((8, "serial"), (64, "simdgroup_32")):
        x, w = _half(40, columns, seed=193), _half(columns, 24, seed=194)
        program = ts.jit(target="apple_gpu")(_rmsnorm_product)
        output = program(x, w)
        receipt = program._apple_tensor_last_receipt
        assert receipt["compiler_path"] == "canonical_apple_tensor_program"
        compiled = program._apple_tensor_last_program
        norm = compiled.stages[0]
        assert norm.package.descriptor.provenance["schedule"] == schedule
        roots = [arg.name for arg in compiled.graph.functions[0].args]
        sequential, _ = atp.run_members_sequentially(compiled, dict(zip(roots, (x, w))))
        assert output.tobytes() == sequential.tobytes()


@pytest.mark.parametrize("record", ["apple7_norm_row_schedules.json",
                                    "apple7_norm_row_schedules_bf16.json"])
def test_the_row_schedule_cut_matches_the_committed_device_evidence(record):
    """The Schedule's 16-column cut is a measured performance key (slice 11).

    From 16 columns up the simdgroup schedule's median must beat serial at
    every shape and it must win at least two thirds of the paired
    device-clock samples (single pairs near the ~5 us launch floor can go
    either way), and the next width below the cut must lose somewhere -- or
    the cut is wrong or lower than the evidence allows.
    """
    import json
    import pathlib

    path = pathlib.Path(__file__).parents[2] / "benchmarks" / "baselines" / record
    data = json.loads(path.read_text())
    assert data["schema"] == "tessera.apple.norm_row_schedules.v1"
    assert "GPUStartTime/GPUEndTime" in data["clock"] and data["rounds"] >= 15
    rows = data["rows"]
    above = [r for r in rows if r["columns"] >= 16]
    below = [r for r in rows if r["columns"] < 16]
    forms = {(r["kind"], r["affine"]) for r in rows}
    assert forms == {("rmsnorm", "none"), ("layernorm", "none"),
                     ("rmsnorm", "gamma"), ("layernorm", "gamma_beta")}
    assert above and below and {r["columns"] for r in above} >= {16, 4096}
    for r in above:
        assert r["pairs"] >= 15 and r["speedup_median"] > 1.2, r
        assert 3 * r["simdgroup_wins_pairs"] >= 2 * r["pairs"], r
    # The next measured width down (8) loses on tall tensors.
    assert any(r["columns"] == 8 and r["speedup_median"] < 1.0 for r in below)
