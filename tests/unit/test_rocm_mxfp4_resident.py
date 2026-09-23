"""Host-free lifecycle checks for the opt-in packed-folded HIP session."""
from __future__ import annotations

import ctypes
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tessera.compiler.rocm_mxfp4 import pack_e2m1_codes
from tessera.compiler.rocm_mxfp4_packed_folded import (
    PACKED_FOLDED_PHYSICAL_V1,
    PACKED_FOLDED_TARGET_ABI_V1,
    prepare_packed_folded_payload,
)
from tessera.compiler.rocm_mxfp4_resident import PackedFoldedResidentSession


class _FakeFunction:
    def __init__(self, callback: object) -> None:
        self.callback = callback
        self.argtypes: object = None

    def __call__(self, *args: object) -> int:
        return self.callback(*args)  # type: ignore[operator]


class _FakeHip:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.allocations: list[ctypes.Array[ctypes.c_char]] = []
        for name in (
            "hipInit", "hipStreamCreateWithFlags", "hipStreamDestroy",
            "hipStreamSynchronize", "hipModuleLoadData", "hipModuleUnload",
            "hipModuleGetFunction", "hipMalloc", "hipFree", "hipMemcpyAsync",
            "hipModuleLaunchKernel",
        ):
            setattr(self, name, _FakeFunction(lambda *args, name=name: self._call(name, *args)))

    def _call(self, name: str, *args: object) -> int:
        self.calls.append(name)
        if name in ("hipStreamCreateWithFlags", "hipModuleLoadData", "hipModuleGetFunction"):
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(1)
        elif name == "hipMalloc":
            allocation = ctypes.create_string_buffer(int(args[1]))
            self.allocations.append(allocation)
            ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(
                ctypes.addressof(allocation)
            )
        elif name == "hipMemcpyAsync":
            ctypes.memmove(args[0], args[1], int(args[2]))
        return 0


def _fixture() -> tuple[object, object]:
    n, k, m = 48, 64, 65
    payload = prepare_packed_folded_payload(
        pack_e2m1_codes(np.ones((n, k), dtype=np.uint8)),
        np.full((k // 32, n), 127, dtype=np.uint8),
        allow_approximate=True,
    )
    receipt = payload.receipt()
    abi = PACKED_FOLDED_TARGET_ABI_V1
    guards = [
        SimpleNamespace(binding=name, dimension=axis, predicate="eq", value=value)
        for name, axis, value in (
            ("a", 0, m), ("a", 1, k), ("b_packed", 0, n),
            ("b_packed", 1, k // 2), ("a_scale", 0, m),
            ("scale_plane", 0, k // 32 + 1), ("scale_plane", 1, n),
            ("output", 0, m), ("output", 1, n),
        )
    ]
    image = SimpleNamespace(
        target="rocm_gfx1201", architecture="gfx1201", payload=b"fake-hsaco",
        image_digest="fake-image-id", entry_points=(SimpleNamespace(symbol="kernel", abi_id=abi),),
    )
    descriptor = SimpleNamespace(
        abi_id=abi, image_digest=image.image_digest, entry_symbol="kernel",
        provenance={**receipt, "numeric_policy": "folded_row_reference_explicit_approximate",
                    "physical_contract": PACKED_FOLDED_PHYSICAL_V1,
                    "execution_state": "manual_executable_candidate"},
        buffers=tuple(SimpleNamespace(name=name, ordinal=ordinal) for ordinal, name in enumerate(
            ("a", "b_packed", "a_scale", "scale_plane", "output")
        )),
        scalars=tuple(SimpleNamespace(name=name, ordinal=ordinal) for ordinal, name in enumerate(
            ("M", "N", "K"), start=5,
        )),
        shape_guards=guards,
        geometry=SimpleNamespace(grid=(1, 1, 1), workgroup=(256, 1, 1)),
    )
    return SimpleNamespace(image=image, descriptor=descriptor), payload


def test_resident_launch_keeps_weights_module_and_buffers() -> None:
    package, payload = _fixture()
    hip = _FakeHip()
    with patch("tessera.runtime._rocm_live_arch", return_value="gfx1201"):
        with PackedFoldedResidentSession(package, payload, 65, hip=hip) as session:
            a = np.full((65, 64), 0x38, dtype=np.uint8)
            scale = np.ones(65, dtype=np.float32)
            session.upload_activations(a, scale)
            before = len(hip.calls)
            session.launch_resident()
            assert hip.calls[before:] == ["hipModuleLaunchKernel"]
            session.synchronize()
            session.upload_activations(a, scale)
            with pytest.raises(RuntimeError, match="has not launched"):
                session.read_output()
            session.launch_resident()
            session.synchronize()
            assert session.receipt()["kernel_launches"] == 2
            assert hip.calls.count("hipModuleLoadData") == 1
            assert hip.calls.count("hipMalloc") == 5
            assert hip.calls.count("hipMemcpyAsync") == 6  # B + scale plane + 2(A + As)
    assert hip.calls.count("hipFree") == 5
    assert hip.calls.count("hipModuleUnload") == 1


def test_resident_refuses_wrong_device_and_mismatched_payload() -> None:
    package, payload = _fixture()
    hip = _FakeHip()
    with patch("tessera.runtime._rocm_live_arch", return_value="gfx1151"):
        with pytest.raises(RuntimeError, match="selected gfx1201"):
            PackedFoldedResidentSession(package, payload, 65, hip=hip)
    assert not hip.calls
    package.descriptor.provenance["weight_sha256"] = "wrong"
    with pytest.raises(ValueError, match="weight_sha256"):
        PackedFoldedResidentSession(package, payload, 65, hip=hip)
    assert not hip.calls
