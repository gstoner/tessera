from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from tessera.aot import AOTArtifact, CompilationCache, compilation_cache_key
from tessera.compiler.canonical_compile import compile_result_from_bundle
from tessera.compiler.driver import CompileArtifactBundle, CompileRequest
from tessera.compiler.matmul_pipeline import LoweringArtifact
from tessera.compiler.native_artifact import (
    ArtifactContractError,
    BufferArgument,
    BufferBinding,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    ScalarArgument,
)
from tessera.runtime import (
    NativeBufferValue,
    RuntimeArtifact,
    launch,
    register_native_launcher,
    unregister_native_launcher,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _image(
    target_ir: str = "target-ir",
    *,
    target: str = "nvidia_sm120",
    architecture: str = "sm_120a",
    pipeline_name: str = "tessera-nvidia-pipeline-sm120",
    binary_format: str = "ptx",
) -> NativeImageArtifact:
    return NativeImageArtifact(
        target=target,
        architecture=architecture,
        pipeline_name=pipeline_name,
        compiler_fingerprint="tessera-opt:23:test",
        toolchain_fingerprint="cuda:test",
        target_ir_digest=_digest(target_ir),
        binary_format=binary_format,
        payload=b".version 9.3\n.entry spine() {}\n",
        entry_points=(NativeEntryPoint("spine", "tessera.spine.v1"),),
        compile_state="cold",
    )


def _descriptor(image: NativeImageArtifact) -> LaunchDescriptor:
    return LaunchDescriptor(
        image_digest=image.image_digest,
        entry_symbol="spine",
        abi_id="tessera.spine.v1",
        buffers=(BufferBinding(0, "x", "input", "fp32", 1, "row_major", 16),),
        scalars=(ScalarArgument(1, "count", "int32"),),
        geometry=LaunchGeometry(grid=(1, 1, 1), workgroup=(32, 1, 1)),
    )


def _bundle(
    *,
    image: NativeImageArtifact | None,
    descriptor: LaunchDescriptor | None,
    target: str = "nvidia_sm120",
    pipeline_name: str = "",
):
    request = CompileRequest(
        source_origin="unit", function_name="spine", graph_ir="graph-ir",
        target=target, pipeline_name=pipeline_name,
    )
    return CompileArtifactBundle(
        request=request,
        graph=LoweringArtifact("graph", "graph-ir"),
        schedule=LoweringArtifact("schedule", "schedule-ir"),
        tile=LoweringArtifact("tile", "tile-ir"),
        target_ir=LoweringArtifact("target", "target-ir"),
        backend=LoweringArtifact("backend", "backend-ir"),
        native_image=image,
        launch_descriptor=descriptor,
    )


def _runtime_artifact() -> RuntimeArtifact:
    image = _image()
    return RuntimeArtifact(
        graph_ir="graph-ir", schedule_ir="schedule-ir", tile_ir="tile-ir",
        target_ir="target-ir",
        metadata={"target": "nvidia_sm120", "compiler_path": "typed_spine"},
        abi_signature="tessera.canonical.v1.nvidia_sm120",
        native_image=image,
        launch_descriptor=_descriptor(image),
    )


def _args(array: np.ndarray | None = None):
    value = np.ones(8, dtype=np.float32) if array is None else array
    return {
        "x": NativeBufferValue(
            value,
            BufferArgument("fp32", value.shape, "row_major", 256),
        ),
        "count": 8,
    }


def test_bundle_records_packaged_and_launchable_stages() -> None:
    image = _image()
    packaged = _bundle(image=image, descriptor=None)
    launchable = _bundle(image=image, descriptor=_descriptor(image))

    assert packaged.orchestration_state == "packaged"
    assert launchable.orchestration_state == "launchable"
    assert [stage["stage"] for stage in launchable.spine_stages()] == [
        "graph", "schedule", "tile", "target", "backend",
        "native_image", "launch_descriptor",
    ]
    assert launchable.to_metadata()["launch_descriptor_digest"]


def test_bundle_rejects_image_provenance_drift() -> None:
    with pytest.raises(ValueError, match="Target IR"):
        _bundle(image=_image("different-target-ir"), descriptor=None)
    wrong_target = _image(
        target="nvidia_sm100",
        architecture="sm_100a",
        pipeline_name="tessera-nvidia-pipeline-sm100",
    )
    with pytest.raises(ValueError, match="target"):
        _bundle(image=wrong_target, descriptor=None)


@pytest.mark.parametrize(
    "target,architecture,pipeline_name,binary_format",
    [
        ("nvidia_sm120", "sm_120a", "tessera-nvidia-pipeline-sm120", "ptx"),
        ("rocm_gfx1151", "gfx1151", "tessera-lower-to-rocm", "hsaco"),
    ],
)
def test_bundle_accepts_target_registry_declared_producer(
    target: str,
    architecture: str,
    pipeline_name: str,
    binary_format: str,
) -> None:
    image = _image(
        target=target,
        architecture=architecture,
        pipeline_name=pipeline_name,
        binary_format=binary_format,
    )
    bundle = _bundle(image=image, descriptor=None, target=target)
    assert bundle.native_image is image


def test_bundle_rejects_unrelated_registered_target_pipeline() -> None:
    image = _image(pipeline_name="tessera-pipeline")
    with pytest.raises(ValueError, match="pipeline"):
        _bundle(image=image, descriptor=None)


def test_compile_result_projects_typed_artifacts_without_reconstruction() -> None:
    image = _image()
    descriptor = _descriptor(image)
    result = compile_result_from_bundle(_bundle(image=image, descriptor=descriptor))
    runtime = result.to_runtime_artifact()

    assert result.native_image is image
    assert result.launch_descriptor is descriptor
    assert runtime.native_image is image
    assert runtime.launch_descriptor is descriptor
    assert runtime.metadata["orchestration_state"] == "launchable"


def test_runtime_artifact_round_trip_validates_nested_hashes() -> None:
    artifact = _runtime_artifact()
    restored = RuntimeArtifact.from_json(artifact.to_json())
    assert restored == artifact
    assert restored.artifact_hash == artifact.artifact_hash


def test_runtime_artifact_rejects_descriptor_without_image() -> None:
    image = _image()
    with pytest.raises(ArtifactContractError, match="E_LAUNCH_STALE_IMAGE"):
        RuntimeArtifact(launch_descriptor=_descriptor(image))


def test_runtime_artifact_rejects_target_ir_provenance_drift() -> None:
    with pytest.raises(ArtifactContractError, match="E_LAUNCH_STALE_IMAGE"):
        replace(_runtime_artifact(), target_ir="different-target-ir")


def test_runtime_artifact_deserialization_rejects_target_ir_provenance_drift() -> None:
    data = _runtime_artifact().to_dict()
    data["target_ir"] = "different-target-ir"
    with pytest.raises(ArtifactContractError, match="E_LAUNCH_STALE_IMAGE"):
        RuntimeArtifact.from_dict(data)


def test_generic_runtime_validates_then_submits_exact_descriptor() -> None:
    calls = []

    def submit(image, descriptor, buffers, scalars, stream):
        calls.append((image, descriptor, buffers, scalars, stream))
        return buffers["x"] * scalars["count"]

    register_native_launcher(
        "nvidia_sm120", binary_formats=("ptx",), submit=submit,
    )
    try:
        result = launch(_runtime_artifact(), _args(), stream="stream-7")
    finally:
        unregister_native_launcher("nvidia_sm120")

    assert result["ok"] is True
    assert result["execution_mode"] == "descriptor"
    assert result["execution_kind"] == "native_gpu"
    assert result["image_digest"]
    assert result["launch_descriptor_digest"]
    assert calls[0][1].entry_symbol == "spine"
    assert calls[0][4] == "stream-7"


def test_binding_failure_happens_before_backend_submission() -> None:
    calls = []
    register_native_launcher(
        "nvidia_sm120", binary_formats=("ptx",),
        submit=lambda *values: calls.append(values),
    )
    bad = _args(np.ones((2, 4), dtype=np.float32))
    try:
        result = launch(_runtime_artifact(), bad)
    finally:
        unregister_native_launcher("nvidia_sm120")

    assert result["ok"] is False
    assert result["runtime_status"] == "invalid_artifact"
    assert result["diagnostic_code"] == "E_LAUNCH_BINDING_MISMATCH"
    assert calls == []


def test_descriptor_route_never_falls_back_when_launcher_is_absent() -> None:
    result = launch(_runtime_artifact(), _args())
    assert result["ok"] is False
    assert result["runtime_status"] == "unimplemented"
    assert "legacy fallback is disabled" in result["reason"]


def test_aot_cache_identity_includes_native_and_launch_fingerprints() -> None:
    artifact = _runtime_artifact()
    key = compilation_cache_key(artifact, target="nvidia_sm120")
    changed_image = replace(artifact.native_image, toolchain_fingerprint="cuda:new")
    changed = replace(
        artifact,
        native_image=changed_image,
        launch_descriptor=_descriptor(changed_image),
    )
    assert compilation_cache_key(changed, target="nvidia_sm120") != key


def test_compilation_cache_rejects_tampered_native_payload(tmp_path) -> None:
    artifact = AOTArtifact(runtime_artifact=_runtime_artifact())
    cache = CompilationCache(tmp_path)
    stored = cache.put("spine-key", artifact)
    data = json.loads((stored / "artifact.json").read_text())
    data["runtime_artifact"]["native_image"]["payload_digest"] = "0" * 64
    (stored / "artifact.json").write_text(json.dumps(data))

    with pytest.raises(ArtifactContractError, match="E_NATIVE_IMAGE_DIGEST_MISMATCH"):
        cache.get("spine-key")


def test_compilation_cache_manifest_rejects_valid_artifact_swap(tmp_path) -> None:
    cache = CompilationCache(tmp_path)
    stored = cache.put("spine-key", AOTArtifact(runtime_artifact=_runtime_artifact()))
    manifest_path = stored / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["native_image_cache_key"]
    assert manifest["launch_cache_fingerprint"]
    manifest["lookup_key"] = "different-key"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ArtifactContractError, match="E_NATIVE_IMAGE_DIGEST_MISMATCH"):
        cache.get("spine-key")
