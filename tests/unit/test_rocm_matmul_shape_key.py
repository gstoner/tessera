"""gfx1151 register matmul image identity follows its compiled Target directive."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tessera.compiler import rocm_native, scheduled_matmul
from tests.unit.test_scheduled_matmul_consumers import _module


def _target(shape: int, digest: str, *, mt: int = 2) -> str:
    return f"""module attributes {{tessera.arch = "gfx1151"}} {{
  func.func @graph_{shape}() {{
    %c{shape} = arith.constant {shape} : i64
    tessera_rocm.wmma_gemm {{arch = "gfx1151", k = 16 : i64, m = 16 : i64, mt = {mt} : i64, n = 16 : i64, name = "graph_{shape}", nt = 4 : i64, tessera.schedule_hash = "{digest}"}}
    return
  }}
}}
"""


def _project(source: str) -> str:
    return rocm_native._shape_free_target_ir(
        source, family="matmul", directive="tessera_rocm.wmma_gemm"
    )


def test_shape_and_schedule_provenance_do_not_key_the_binary() -> None:
    first = _project(_target(16, "a" * 64))
    second = _project(_target(48, "b" * 64))
    assert first == second
    assert "schedule_hash" not in first and "func.func" not in first
    assert rocm_native._directive_symbol(first, "tessera_rocm.wmma_gemm").startswith(
        "tessera_rocm_matmul_"
    )


def test_physical_matmul_policy_remains_in_the_key() -> None:
    assert _project(_target(16, "a" * 64, mt=2)) != _project(
        _target(16, "a" * 64, mt=4)
    )


def test_matmul_projection_refuses_missing_provenance_or_other_code() -> None:
    with pytest.raises(RuntimeError, match="one Schedule hash"):
        _project(_target(16, "a" * 64).replace(
            ", tessera.schedule_hash = " + chr(34) + "a" * 64 + chr(34), ""
        ))
    with pytest.raises(RuntimeError, match="unaudited Target IR operation"):
        _project(_target(16, "a" * 64).replace(
            "    return", "    %x = arith.addi %c16, %c16 : i64\n    return"
        ))


@pytest.mark.hardware_rocm
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.skipif(
    os.environ.get("TESSERA_ROCM_E2E_DEVICE_TEST") != "1",
    reason="requires explicit gfx1151 device proof gate",
)
def test_three_shapes_share_one_live_gfx1151_image(dtype: str) -> None:
    from tessera import runtime as rt

    assert rt._rocm_live_arch() == rt._rocm_chip() == "gfx1151"
    rng = np.random.default_rng(885)
    storage = np.float16 if dtype == "fp16" else pytest.importorskip("ml_dtypes").bfloat16
    digests = []
    for shape in ((16, 16, 16), (32, 16, 16), (48, 16, 16)):
        m, k, n = shape
        scheduled = scheduled_matmul.lower_scheduled_matmul(
            _module(target="rocm", shape=shape, dtype=dtype), target="rocm_gfx1151"
        )
        package = rocm_native.package_scheduled_matmul(
            scheduled, pipeline_name="tessera-lower-to-rocm"
        )
        a = (rng.normal(size=(m, k)) * 0.2).astype(storage)
        b = (rng.normal(size=(k, n)) * 0.2).astype(storage)
        output = np.zeros((m, n), np.float32)
        artifact = rt.RuntimeArtifact(
            metadata={"target": package.image.target},
            native_image=package.image,
            launch_descriptor=package.descriptor,
            tile_ir=package.tile_ir,
            target_ir=package.target_ir,
        )
        result = rt.launch(
            artifact,
            {"buffers": {"a": a, "b": b, "o": output},
             "scalars": {"M": m, "N": n, "K": k}},
        )
        assert result["ok"] and result["execution_kind"] == "native_gpu", result
        np.testing.assert_allclose(
            output, a.astype(np.float32) @ b.astype(np.float32), rtol=0, atol=2e-2
        )
        digests.append(package.image.image_digest)
        assert package.descriptor.entry_symbol == package.image.entry_points[0].symbol
        assert "tessera.schedule_hash" not in package.target_ir
    assert len(set(digests)) == 1
