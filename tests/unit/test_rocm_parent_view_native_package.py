"""Host-free: compile parent-view packages and reject invalid frames before HIP.

These tests require matching native compiler tools, never a live ROCm device.
GPU numerical/replay proof belongs to tests/device/rocm/test_native_parent_view_jit.py.
"""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest
from tessera import runtime as rt
from tessera.compiler import rocm_parent_view_native as native
from tessera.compiler.scheduled_matmul import run_tessera_opt
from tests.unit.test_native_gate_up_view_schedule import source, tool

pytestmark = pytest.mark.compiler_route


@pytest.fixture(scope="module")
def packaged():
    graph = source(3, 17)
    schedule = run_tessera_opt(tool(), graph, "--tessera-graph-to-schedule")
    tile = run_tessera_opt(tool(), schedule, "--tessera-schedule-to-tile")
    recipe = native.ParentViewRecipe(graph, schedule, tile)
    package = native.package_parent_view_recipe(recipe, pipeline_name="tessera-rocm-executable")
    artifact = rt.RuntimeArtifact(graph_ir=graph, schedule_ir=schedule, tile_ir=tile,
        target_ir=package.target_ir, native_image=package.image, launch_descriptor=package.descriptor,
        metadata={"target": "rocm_gfx1201", "compiler_path": "rocm_parent_view_native_descriptor",
                  "execution_kind": "native_gpu"})
    return package, artifact


def test_portable_parent_view_validation_uses_no_compiler(packaged, monkeypatch):
    _, artifact = packaged
    restored = rt.RuntimeArtifact.from_json(artifact.to_json())

    def forbidden(*a, **k):
        pytest.fail("portable parent-view validation invoked the compiler")

    monkeypatch.setattr(native, "find_tessera_opt", forbidden)
    monkeypatch.setattr(native, "run_tessera_opt", forbidden)
    program = native.validate_parent_view_runtime_artifact(restored)
    program.validate()
    assert json.loads(program.program_json)["buffers"][0]["ownership"] == 0


@pytest.mark.parametrize("field", ["graph_ir", "schedule_ir", "tile_ir", "target_ir"])
def test_portable_parent_view_stage_mutations_fail_before_hip(packaged, monkeypatch, field):
    _, artifact = packaged
    if field == "target_ir":
        with pytest.raises(ValueError, match="E_LAUNCH_STALE_IMAGE"):
            replace(artifact, target_ir=artifact.target_ir + "\n// changed\n")
        return
    changed = replace(artifact, **{field: getattr(artifact, field) + "\n// changed\n"})
    monkeypatch.setattr(rt, "_load_hip_for_launch", lambda: pytest.fail("invalid stage reached HIP"))
    receipt = rt.launch(changed, {})
    assert not receipt["ok"] and receipt["diagnostic_code"] == "E_LAUNCH_BINDING_MISMATCH"


@pytest.mark.parametrize("field", ["geometry", "scalars", "bindings", "parent_shape", "inputs"])
def test_native_parent_view_frame_mutations_rejected(packaged, field):
    package, _ = packaged
    program, member = native._frame(package.target_ir)
    member = deepcopy(member)
    member[field] = {"geometry": [1, 1, 1, 128, 1, 1], "scalars": [52],
        "bindings": ["parent", "parent"], "parent_shape": [3, 35], "inputs": [1]}[field]
    with pytest.raises(ValueError):
        native._validate_frame(program, member)


@pytest.mark.parametrize("bad", ["count", "shape", "noncompact", "readonly", "alias"])
def test_native_parent_view_invocation_rejected_before_runtime(packaged, monkeypatch, bad):
    package, _ = packaged
    monkeypatch.setattr(rt, "_rocm_live_arch", lambda: "gfx1201")
    monkeypatch.setattr(rt, "_load_rocm_native_movement_runtime",
                        lambda: pytest.fail("invalid invocation reached the program owner"))
    parent = np.ones((3, 34), np.float32)
    output = np.empty((3, 17), np.float32)
    scalars = {"N": 51}
    if bad == "count":
        scalars["N"] = 52
    if bad == "shape":
        parent = np.ones((3, 35), np.float32)
    if bad == "noncompact":
        parent = np.ones((3, 68), np.float32)[:, ::2]
    if bad == "readonly":
        output.flags.writeable = False
    if bad == "alias":
        output = parent.reshape(-1)[:51].reshape(3, 17)
    with pytest.raises(ValueError):
        native.submit_parent_view(package.image, package.descriptor,
            {"parent": parent, "out": output}, scalars, None)


def test_parent_view_cannot_relabel_the_native_image_producer(monkeypatch):
    def forbidden(*a, **k):
        pytest.fail("invalid producer reached compiler discovery")

    monkeypatch.setattr(native, "find_tessera_opt", forbidden)
    with pytest.raises(ValueError, match="image producer"):
        native.package_parent_view_recipe(native.ParentViewRecipe("", "", ""),
            pipeline_name="tessera-lower-to-rocm")


def test_portable_parent_view_rejects_coherently_relabelled_producer(packaged, monkeypatch):
    package, artifact = packaged
    image = replace(package.image, pipeline_name="tessera-lower-to-rocm")
    descriptor = replace(package.descriptor, image_digest=image.image_digest)
    changed = replace(artifact, native_image=image, launch_descriptor=descriptor)
    monkeypatch.setattr(rt, "_load_hip_for_launch", lambda: pytest.fail("relabelled image reached HIP"))
    receipt = rt.launch(changed, {})
    assert not receipt["ok"] and receipt["diagnostic_code"] == "E_LAUNCH_BINDING_MISMATCH"
