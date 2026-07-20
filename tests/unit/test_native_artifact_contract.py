from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from tessera.compiler.native_artifact import (
    ArtifactContractError,
    BufferArgument,
    BufferBinding,
    DeviceLibraryRecord,
    LaunchDescriptor,
    LaunchGeometry,
    NativeEntryPoint,
    NativeImageArtifact,
    OrderingSemantics,
    ResourceRecord,
    ScalarArgument,
    ShapeGuard,
    WorkspaceRequirement,
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _image(**changes) -> NativeImageArtifact:
    values = {
        "target": "nvidia_sm120",
        "architecture": "sm_120a",
        "pipeline_name": "tessera-nvidia-pipeline-sm120",
        "compiler_fingerprint": "tessera-opt:23.0.0:abc123",
        "toolchain_fingerprint": "cuda:13.3:driver-590",
        "target_ir_digest": _digest("target-ir"),
        "binary_format": "ptx",
        "payload": b".version 9.3\n.entry tessera_mm() {}\n",
        "entry_points": (NativeEntryPoint("tessera_mm", "tessera.matmul.v1"),),
        "compile_state": "cold",
        "resource_record": ResourceRecord(
            provenance="ptxas --verbose",
            metrics={"registers": 40, "spill_bytes": 0},
        ),
        "device_libraries": (
            DeviceLibraryRecord("cuda.libdevice", _digest("libdevice"),
                                "llvm_link_only_needed"),
        ),
    }
    values.update(changes)
    return NativeImageArtifact(**values)


def _descriptor(image: NativeImageArtifact, **changes) -> LaunchDescriptor:
    values = {
        "image_digest": image.image_digest,
        "entry_symbol": "tessera_mm",
        "abi_id": "tessera.matmul.v1",
        "buffers": (
            BufferBinding(0, "a", "input", "f16", 2, "row_major", 16),
            BufferBinding(1, "b", "input", "fp16", 2, "row_major", 16),
            BufferBinding(2, "out", "output", "fp32", 2, "row_major", 16),
        ),
        "scalars": (ScalarArgument(3, "alpha", "f32"),),
        "shape_guards": (
            ShapeGuard("a", 1, "multiple_of", 16),
            ShapeGuard("b", 0, "multiple_of", 16),
        ),
        "geometry": LaunchGeometry(grid=(2, 2, 1), workgroup=(32, 1, 1)),
        "dynamic_local_memory_bytes": 4096,
        "workspace": WorkspaceRequirement(
            bytes=8192,
            alignment=256,
            lifetime="session",
            initialization="preserve",
        ),
        "ordering": OrderingSemantics(
            ordered_submission=True,
            residency="inputs",
            synchronization=("stream_ordered",),
        ),
        "provenance": {"compiler_stage": "target-to-native", "candidate": "tile_f16"},
    }
    values.update(changes)
    return LaunchDescriptor(**values)


def _buffers(**changes) -> dict[str, BufferArgument]:
    values = {
        "a": BufferArgument("fp16", (32, 64), "row_major", 256),
        "b": BufferArgument("fp16", (64, 48), "row_major", 256),
        "out": BufferArgument("fp32", (32, 48), "row_major", 256),
    }
    values.update(changes)
    return values


def test_native_image_round_trip_is_content_addressed_and_deterministic() -> None:
    image = _image()
    restored = NativeImageArtifact.from_json(image.to_json())
    assert restored == image
    assert restored.payload_digest == image.payload_digest
    assert restored.image_digest == image.image_digest
    assert restored.cache_key == image.cache_key
    assert restored.to_json() == image.to_json()
    assert json.loads(image.to_json())["payload_b64"]


def test_frozen_native_identity_digests_are_cached_after_first_use() -> None:
    image = _image()
    descriptor = _descriptor(image)
    assert image.payload_digest == image.payload_digest
    assert image.image_digest == image.image_digest
    assert image.cache_key == image.cache_key
    assert descriptor.descriptor_digest == descriptor.descriptor_digest
    assert descriptor.cache_fingerprint == descriptor.cache_fingerprint
    assert {
        "payload_digest",
        "image_digest",
        "cache_key",
    } <= image.__dict__.keys()
    assert {"descriptor_digest", "cache_fingerprint"} <= descriptor.__dict__.keys()


def test_workspace_lifecycle_is_round_tripped_and_rejects_invalid_combinations() -> None:
    image = _image()
    descriptor = _descriptor(image)
    restored = LaunchDescriptor.from_json(descriptor.to_json())
    assert restored.workspace == WorkspaceRequirement(
        bytes=8192,
        alignment=256,
        lifetime="session",
        initialization="preserve",
    )
    assert WorkspaceRequirement.from_dict({"bytes": 64, "alignment": 16}) == (
        WorkspaceRequirement(bytes=64, alignment=16)
    )
    with pytest.raises(ArtifactContractError, match="preserved workspace requires session"):
        WorkspaceRequirement(bytes=4, lifetime="launch", initialization="preserve")
    with pytest.raises(ArtifactContractError, match="invalid workspace lifetime"):
        WorkspaceRequirement(bytes=4, lifetime="process")


def test_compile_state_and_resource_measurement_do_not_change_image_identity() -> None:
    cold = _image()
    warm = replace(
        cold,
        compile_state="warm_cache",
        resource_record=ResourceRecord("ncu", {"occupancy": 0.5}),
    )
    assert warm.image_digest == cold.image_digest
    assert warm.cache_key == cold.cache_key
    assert warm.to_dict()["compile_state"] == "warm_cache"


def test_cache_key_changes_for_target_ir_or_toolchain_drift() -> None:
    image = _image()
    assert replace(image, target_ir_digest=_digest("new-ir")).cache_key != image.cache_key
    assert replace(image, toolchain_fingerprint="cuda:13.4").cache_key != image.cache_key
    changed_library = replace(
        image,
        device_libraries=(
            DeviceLibraryRecord(
                "cuda.libdevice", _digest("new-libdevice"), "llvm_link_only_needed"
            ),
        ),
    )
    assert changed_library.cache_key != image.cache_key


def test_device_library_records_round_trip_without_host_paths() -> None:
    image = _image()
    restored = NativeImageArtifact.from_json(image.to_json())
    assert restored.device_libraries == image.device_libraries
    payload = image.to_json()
    assert "cuda.libdevice" in payload
    assert "/usr/local/cuda" not in payload


def test_v1_native_image_without_device_library_field_remains_readable() -> None:
    image = _image(device_libraries=())
    legacy = image.to_dict()
    legacy.pop("device_libraries")
    assert NativeImageArtifact.from_dict(legacy) == image


@pytest.mark.parametrize("field", ["payload_digest", "image_digest", "cache_key"])
def test_native_image_rejects_serialized_digest_drift(field: str) -> None:
    data = _image().to_dict()
    data[field] = "0" * 64
    with pytest.raises(ArtifactContractError) as exc:
        NativeImageArtifact.from_dict(data)
    assert exc.value.code == "E_NATIVE_IMAGE_DIGEST_MISMATCH"


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": "tessera.native_image.v0"},
        {"binary_format": "cuda_magic"},
        {"pipeline_name": "not-registered"},
        {"pipeline_name": "tessera-lower-to-rocm"},
        {"payload": b""},
        {"entry_points": ()},
        {"compile_state": "maybe_cached"},
    ],
)
def test_native_image_rejects_malformed_contract(changes) -> None:
    with pytest.raises(ArtifactContractError) as exc:
        _image(**changes)
    assert exc.value.code == "E_NATIVE_IMAGE_SCHEMA"


def test_launch_descriptor_round_trip_and_fingerprint_are_stable() -> None:
    image = _image()
    descriptor = _descriptor(image)
    restored = LaunchDescriptor.from_json(descriptor.to_json())
    assert restored == descriptor
    assert restored.descriptor_digest == descriptor.descriptor_digest
    assert restored.cache_fingerprint == descriptor.cache_fingerprint
    assert restored.buffers[0].dtype == "fp16"
    assert restored.scalars[0].dtype == "fp32"


def test_portable_schema_has_no_backend_physical_schedule_fields() -> None:
    payload = json.dumps(_descriptor(_image()).to_dict(), sort_keys=True)
    for forbidden in ("warp_map", "wave_layout", "threadgroup_schedule", "vector_width"):
        assert forbidden not in payload


def test_launch_descriptor_validates_image_and_invocation() -> None:
    image = _image()
    descriptor = _descriptor(image)
    descriptor.validate_invocation(image, _buffers(), {"alpha": 1.0})


def test_launch_descriptor_rejects_stale_image_symbol_and_abi() -> None:
    image = _image()
    descriptor = _descriptor(image)
    with pytest.raises(ArtifactContractError, match="E_LAUNCH_STALE_IMAGE"):
        descriptor.validate_image(replace(image, payload=b"different"))
    with pytest.raises(ArtifactContractError, match="E_LAUNCH_STALE_IMAGE"):
        replace(descriptor, entry_symbol="missing").validate_image(image)
    with pytest.raises(ArtifactContractError, match="E_LAUNCH_STALE_IMAGE"):
        replace(descriptor, abi_id="tessera.matmul.v2").validate_image(image)


@pytest.mark.parametrize(
    "buffers,scalars",
    [
        ({}, {"alpha": 1.0}),
        (_buffers(a=BufferArgument("fp32", (32, 64), "row_major", 256)), {"alpha": 1.0}),
        (_buffers(a=BufferArgument("fp16", (32,), "row_major", 256)), {"alpha": 1.0}),
        (_buffers(a=BufferArgument("fp16", (32, 64), "col_major", 256)), {"alpha": 1.0}),
        (_buffers(a=BufferArgument("fp16", (32, 64), "row_major", 8)), {"alpha": 1.0}),
        (_buffers(), {}),
        (_buffers(), {"alpha": "one"}),
        (_buffers(), {"alpha": 1.0, "extra": 2}),
        (_buffers(a=BufferArgument("fp16", (32, 63), "row_major", 256)), {"alpha": 1.0}),
    ],
)
def test_launch_descriptor_rejects_bad_bindings_before_submission(buffers, scalars) -> None:
    image = _image()
    with pytest.raises(ArtifactContractError) as exc:
        _descriptor(image).validate_invocation(image, buffers, scalars)
    assert exc.value.code == "E_LAUNCH_BINDING_MISMATCH"


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "tessera.launch_descriptor.v0"},
        {"buffers": (
            BufferBinding(0, "a", "input", "fp16", 2, "row_major"),
            BufferBinding(0, "b", "input", "fp16", 2, "row_major"),
        )},
        {"geometry": LaunchGeometry(policy="runtime_default"),
         "shape_guards": (ShapeGuard("missing", 0, "eq", 1),)},
        {"dynamic_local_memory_bytes": -1},
    ],
)
def test_launch_descriptor_rejects_malformed_contract(change) -> None:
    with pytest.raises(ArtifactContractError) as exc:
        _descriptor(_image(), **change)
    assert exc.value.code == "E_LAUNCH_DESCRIPTOR_SCHEMA"


def test_launch_descriptor_rejects_serialized_fingerprint_drift() -> None:
    data = _descriptor(_image()).to_dict()
    data["provenance"] = {"candidate": "tampered"}
    with pytest.raises(ArtifactContractError) as exc:
        LaunchDescriptor.from_dict(data)
    assert exc.value.code == "E_LAUNCH_DESCRIPTOR_SCHEMA"


def test_serialized_contract_rejects_wrong_json_types_without_coercion() -> None:
    image_data = _image().to_dict()
    image_data["architecture"] = 120
    with pytest.raises(ArtifactContractError, match="E_NATIVE_IMAGE_SCHEMA"):
        NativeImageArtifact.from_dict(image_data)

    descriptor_data = _descriptor(_image()).to_dict()
    descriptor_data["dynamic_local_memory_bytes"] = "4096"
    with pytest.raises(ArtifactContractError, match="E_LAUNCH_DESCRIPTOR_SCHEMA"):
        LaunchDescriptor.from_dict(descriptor_data)


def test_resource_and_provenance_metadata_must_be_finite_json() -> None:
    with pytest.raises(ArtifactContractError, match="E_NATIVE_IMAGE_SCHEMA"):
        _image(resource_record=ResourceRecord("ncu", {"occupancy": float("nan")}))
    with pytest.raises(ArtifactContractError, match="E_LAUNCH_DESCRIPTOR_SCHEMA"):
        _descriptor(_image(), provenance={"latency_ms": float("inf")})


def test_malformed_json_roots_fail_with_registered_codes() -> None:
    with pytest.raises(ArtifactContractError, match="E_NATIVE_IMAGE_SCHEMA"):
        NativeImageArtifact.from_json("[]")
    with pytest.raises(ArtifactContractError, match="E_LAUNCH_DESCRIPTOR_SCHEMA"):
        LaunchDescriptor.from_json("not-json")
