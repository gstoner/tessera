"""Standalone ReLU carries a compiler-owned typed native package."""
from dataclasses import replace

import ml_dtypes
import numpy as np
import pytest
import tessera as ts
from tessera.compiler import nvidia_native as native, scheduled_kernel

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


def relu(x):
    return ts.ops.relu(x)


def module(storage="f32"):
    dtype = {"f16": np.float16, "bf16": ml_dtypes.bfloat16, "f32": np.float32}[storage]
    fn = ts.jit(target="nvidia_sm120")(relu)
    return fn._traced_autodiff_module((np.zeros((7, 32), dtype),), {})


@pytest.mark.parametrize("storage", ["f16", "bf16", "f32"])
@pytest.mark.parametrize("schedule", ["serial", "flat_128"])
def test_standalone_relu_native_package(storage, schedule):
    graph = module(storage)
    assert scheduled_kernel.supports_scheduled_kernel(graph, target="nvidia_sm120")
    artifact = scheduled_kernel.lower_scheduled_kernel(
        graph, target="nvidia_sm120", schedule=schedule)
    package = native.package_scheduled_kernel(artifact, pipeline_name="tessera-nvidia-pipeline-sm120")
    assert package.descriptor.abi_id == f"tessera.nvidia.relu.x_o_rows_columns.{storage}.v1"
    assert package.descriptor.provenance["zero_mode"] == ("positive" if storage == "f16" else "preserve")
    assert native.native_package_kind(graph) == "activation"
    assert artifact.family == "activation"
    assert "schedule.activation" in artifact.schedule_ir
    assert "tile.activation_kernel" in artifact.tile_ir
    assert "tile.activation_kernel" not in package.target_ir
    package.descriptor.validate_image(package.image)


@pytest.mark.parametrize("field,value", [
    ("zero_mode", "positive"), ("nan_mode", "ignore"), ("storage", "f16"),
])
def test_standalone_f32_relu_refuses_forged_policy(field, value):
    artifact = scheduled_kernel.lower_scheduled_kernel(module(), target="nvidia_sm120")
    package = native.package_scheduled_kernel(artifact, pipeline_name="tessera-nvidia-pipeline-sm120")
    forged = replace(package.descriptor, provenance={**package.descriptor.provenance, field: value})
    with pytest.raises(ValueError, match="numerical ABI mismatch"):
        native._validate_scheduled_relu_descriptor(forged)


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("field,value", [
    ("dtype", "fp16"), ("dtype", "bf16"), ("rank", 1),
    ("layout", "col_major"), ("alignment", 2),
])
def test_standalone_f32_relu_rejects_forged_buffer_width_or_layout(index, field, value):
    artifact = scheduled_kernel.lower_scheduled_kernel(module(), target="nvidia_sm120")
    package = native.package_scheduled_kernel(artifact, pipeline_name="tessera-nvidia-pipeline-sm120")
    buffers = list(package.descriptor.buffers)
    buffers[index] = replace(buffers[index], **{field: value})
    guards = package.descriptor.shape_guards
    if field == "rank":
        with pytest.raises(ValueError, match="shape guard dimension exceeds rank"):
            replace(package.descriptor, buffers=tuple(buffers))
        # Removing the now-invalid shape guard must not bypass the typed ABI.
        guards = tuple(guard for guard in guards
                       if guard.binding != buffers[index].name or guard.dimension < value)
    forged = replace(package.descriptor, buffers=tuple(buffers), shape_guards=guards)
    with pytest.raises(ValueError, match="typed buffer/scalar numerical ABI mismatch"):
        native._validate_scheduled_relu_descriptor(forged)


def test_standalone_relu_does_not_replace_identity_with_dead_activation():
    graph = module()
    function = graph.functions[0]
    function.return_values = ["%" + function.args[0].name.removeprefix("%")]
    assert not scheduled_kernel.supports_scheduled_kernel(graph, target="nvidia_sm120")
    assert native.native_package_kind(graph) is None
    with pytest.raises(ValueError, match="returned SM120"):
        scheduled_kernel.lower_scheduled_kernel(graph, target="nvidia_sm120")


def test_canonical_activation_dispatch_consumes_existing_scheduled_package():
    graph = module()
    direct = native.package_native(graph, pipeline_name="tessera-nvidia-pipeline-sm120")
    scheduled = native.package_scheduled_kernel(
        scheduled_kernel.lower_scheduled_kernel(graph, target="nvidia_sm120"),
        pipeline_name="tessera-nvidia-pipeline-sm120")
    assert direct.image.payload == scheduled.image.payload
    assert direct.descriptor == scheduled.descriptor
