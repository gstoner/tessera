"""Shape companion ancestry and retained-provider prerequisites are explicit."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tessera.compiler.native_tensor_program import (
    compile_native_tensor_program, runtime_artifact, restore_runtime_artifact,
)
from tessera.compiler.prepared_native_tensor_frame import PreparedNativeTensorFrame, validate_frame
from tests.unit.test_native_tensor_shape_ownership import shape_source

pytestmark = [pytest.mark.compiler_route, pytest.mark.usefixtures("production_compiler")]


@pytest.fixture(scope="module")
def shape_package():
    package = compile_native_tensor_program(shape_source("norm_lhs"))
    assert package is not None
    return package


def test_packaging_binds_companion_to_original_graph_and_images(shape_package):
    package = shape_package
    assert package.frame["schema"] == "tessera.native.tensor_program_frame.v3"
    package.shape_companion.validate(package.frame["shape_program"])
    assert len(package.members) == len(package.images) == len(package.frame["stages"]) == 2
    assert all(b".target sm_120a" in image for image in package.images)
    package.validate()


def test_shape_artifact_restores_via_trusted_source_and_llvm_replay(shape_package):
    artifact = runtime_artifact(shape_package, argument_names=("a", "b"))
    assert artifact.metadata["native_program"]["schema"] == "tessera.native.tensor_program.v2"
    restored = restore_runtime_artifact(artifact)
    assert restored.shape_companion == shape_package.shape_companion
    assert restored.images == shape_package.images


@pytest.mark.parametrize("field", ["missing", "image", "entry", "tool_identity"])
def test_mutated_companion_cannot_replace_verified_program(shape_package, field):
    companion = shape_package.shape_companion
    if field == "missing":
        changed = None
    elif field == "image":
        changed = replace(companion, image=companion.image + b"changed")
    elif field == "entry":
        changed = replace(companion, entry="different")
    else:
        changed = replace(companion, tools=tuple((name, "0" * 64) for name, _ in companion.tools))
    with pytest.raises(ValueError, match="companion"):
        replace(shape_package, shape_companion=changed)


@pytest.mark.parametrize("field", ["root_words", "plan_words", "ir"])
def test_shape_transport_cannot_change_selected_extent_or_ir(shape_package, field):
    frame = shape_package.frame
    frame["shape_program"][field] += "\n" if field == "ir" else 2
    with pytest.raises(ValueError, match="shape"):
        validate_frame(frame)


def test_old_provider_never_prepares_shape_frame_as_static(shape_package):
    provider = SimpleNamespace(tessera_nvidia_native_frame_prepare_v1=lambda *_: pytest.fail("v1 called"))
    with pytest.raises(RuntimeError, match="updated native provider"):
        shape_package.prepare(provider=provider)


def test_missing_companion_rejected_before_provider_invocation(shape_package):
    provider = SimpleNamespace(tessera_nvidia_native_frame_prepare_v3=lambda *_: pytest.fail("v3 called"),
                               tessera_nvidia_native_frame_shapes_v3=lambda *_: pytest.fail("query called"))
    with pytest.raises(ValueError, match="compiled shape companion"):
        PreparedNativeTensorFrame(shape_package.frame, shape_package.images, provider=provider)


def test_shape_artifact_rejects_changed_library_bytes(shape_package):
    artifact = runtime_artifact(shape_package)
    metadata = copy.deepcopy(artifact.metadata)
    metadata["native_program"]["shape_companion"]["image"] = "Y2hhbmdlZA=="
    with pytest.raises(ValueError, match="companion"):
        restore_runtime_artifact(replace(artifact, metadata=metadata))



def test_partial_v3_provider_is_rejected_before_loading_shape_code(shape_package):
    provider = SimpleNamespace(tessera_nvidia_native_frame_prepare_v3=lambda *_: pytest.fail("v3 called"))
    with pytest.raises(RuntimeError, match="updated native provider"):
        shape_package.prepare(provider=provider)
