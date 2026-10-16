"""Owning gfx1201 proof: physical padding never changes logical math."""
import os
import subprocess
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from tessera import runtime as rt
from tessera.compiler import rocm_mxfp4 as mx
from tessera.compiler.rocm_mxfp4_folded import prepare_folded_weights
from tessera.compiler.rocm_mxfp4_folded_frontend import compile_folded_scaled_matmul

pytestmark = [pytest.mark.hardware_rocm, pytest.mark.skipif(
    os.environ.get("TESSERA_GFX1201_DEVICE_PROOF") != "1", reason="explicit gfx1201 gate")]


@pytest.mark.parametrize("shape", [(65, 48, 64), (256, 128, 128), (257, 129, 192), (256, 8192, 1536)])
def test_padded_input_views_preserve_math_and_one_image(shape):
    assert rt._rocm_live_arch() == "gfx1201"
    m, n, k = shape
    rng = np.random.default_rng(m + n + k)
    a = rng.integers(-2, 3, (m, k)).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    scales = np.exp2(rng.integers(-2, 3, m)).astype(np.float32)
    codes = rng.integers(0, 16, (n, k), dtype=np.uint8)
    refs = rng.integers(125, 130, (k // 32, n), dtype=np.uint8)
    folded = prepare_folded_weights(mx.pack_e2m1_codes(codes), refs, allow_approximate=True)
    reference = (a.view(ml_dtypes.float8_e4m3fn).astype(np.float64)
                 @ folded.weight_bytes.view(ml_dtypes.float8_e4m3fn).astype(np.float64).T)
    reference *= np.exp2(folded.row_reference.astype(np.int64) - 127)[None, :]
    reference *= scales.astype(np.float64)[:, None]
    expected = reference.astype(np.float32).astype(ml_dtypes.bfloat16)
    images = set()
    for lda, ldb in [(k, k), (k + 16, k), (k, k + 32), (k + 16, k + 32)]:
        package = compile_folded_scaled_matmul(a, scales, folded,
            tessera_opt=Path(os.environ["TESSERA_OPT"]), allow_approximate=True,
            input_row_strides=(lda, ldb)).package
        images.add(package.image.payload_digest)
        if shape == (256, 8192, 1536):
            assert package.descriptor.provenance["workgroup_mode"] == "cu"
            assert package.descriptor.provenance["fragment_read_schedule"] == "k32_groups"
        artifact = rt.RuntimeArtifact(metadata={"target": package.image.target},
            native_image=package.image, launch_descriptor=package.descriptor,
            tile_ir=package.tile_ir, target_ir=package.target_ir)
        for poison in (0x7f, 0xff):
            a_storage = np.full((m, lda), poison, np.uint8)
            b_storage = np.full((n, ldb), poison, np.uint8)
            a_storage[:, :k] = a
            b_storage[:, :k] = folded.weight_bytes
            output = np.full((m, n), -101, ml_dtypes.bfloat16)
            result = rt.launch(artifact, {"buffers": {
                "a": a_storage, "b_folded": b_storage, "a_scale": scales,
                "row_reference": folded.row_reference, "output": output},
                "scalars": {"M": m, "N": n, "K": k, "LDA": lda, "LDB": ldb}})
            assert result.get("ok") and result.get("execution_kind") == "native_gpu", result
            np.testing.assert_array_equal(output.view(np.uint16), expected.view(np.uint16))
    assert len(images) == 1


def test_canonicalized_identity_views_execute_on_the_owning_device():
    from tessera.compiler.rocm_mxfp4_folded_frontend import (
        author_folded_scaled_matmul_graph, _lower,
    )
    from tessera.compiler.rocm_mxfp4_folded_carrier import package_folded_scaled_wmma_target_ir
    assert rt._rocm_live_arch() == "gfx1201"
    tool = Path(os.environ["TESSERA_OPT"])
    m, n, k = 65, 48, 64
    graph = author_folded_scaled_matmul_graph(m, n, k, input_row_strides=(k, k))
    canonical = subprocess.run([str(tool), "--canonicalize"], input=graph,
        text=True, capture_output=True, check=True).stdout
    assert "tensor.extract_slice" not in canonical
    # Exact decoded operands are 2 and 1, so every result is K*2.
    a = np.full((m, k), 2, ml_dtypes.float8_e4m3fn).view(np.uint8)
    scales = np.ones(m, np.float32)
    folded = prepare_folded_weights(mx.pack_e2m1_codes(np.full((n, k), 2, np.uint8)),
        np.full((k // 32, n), 127, np.uint8), allow_approximate=True)
    package = package_folded_scaled_wmma_target_ir(
        _lower(tool, canonical, target=False), _lower(tool, canonical, target=True),
        folded, allow_approximate=True)
    output = np.full((m, n), -101, ml_dtypes.bfloat16)
    artifact = rt.RuntimeArtifact(metadata={"target": package.image.target},
        native_image=package.image, launch_descriptor=package.descriptor,
        tile_ir=package.tile_ir, target_ir=package.target_ir)
    result = rt.launch(artifact, {"buffers": {"a": a, "b_folded": folded.weight_bytes,
        "a_scale": scales, "row_reference": folded.row_reference, "output": output},
        "scalars": {"M": m, "N": n, "K": k, "LDA": k, "LDB": k}})
    assert result.get("ok") and result.get("execution_kind") == "native_gpu", result
    np.testing.assert_array_equal(output, np.full((m, n), k * 2, ml_dtypes.bfloat16))
