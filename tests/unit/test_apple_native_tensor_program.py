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
    (lambda a, b: ts.ops.matmul(ts.ops.relu(a), b, output_dtype="fp32"),
     (_half(64, 32), _half(32, 48)), "must be tessera.matmul or tessera.softmax"),
    (lambda a, b: ts.ops.softmax(ts.ops.matmul(a, b, output_dtype="fp32"), axis=-1),
     (_half(64, 32), _half(32, 48)), "must end in its matrix product"),
    (lambda a: ts.ops.matmul(a, a, output_dtype="fp32"),
     (_half(32, 32),), "needs a producer and a matrix product"),
])
def test_requested_programs_outside_the_profile_are_refused(fn, arrays, why):
    names = ("a", "b")[:len(arrays)]
    with pytest.raises(RuntimeError, match=f"APPLE_TENSOR_PROGRAM_UNSUPPORTED: .*{why}"):
        atp.compile_apple_tensor_program(_graph(fn, arrays, names))


@pytest.mark.hardware_apple_gpu
@pytest.mark.parametrize("fn, shapes, names", [
    (_scores_values, ((64, 32), (32, 48)), ("s", "v")),
    (_self_attention_like, ((64, 64),), ("x",)),
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
    reference = probabilities.astype(np.float32) @ arrays[-1].astype(np.float32)
    assert output.dtype == np.float32 and output.shape == reference.shape
    np.testing.assert_allclose(output, reference, rtol=0, atol=2e-3)
