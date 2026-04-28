"""
Phase 4 — test_tpu_lowering.py

Tests for TPUTargetProfile: validation, constraint injection, and IR attr
serialization.  Does not require a real TPU or the C++ TPU backend binary.
"""
import pytest
from tessera.compiler.tpu_target import TPUTargetProfile, TPUGeneration


class TestTPUTargetProfile:
    def test_default_generation(self):
        t = TPUTargetProfile()
        assert t.generation == TPUGeneration.V4

    def test_mxu_tile_128(self):
        for gen in (TPUGeneration.V3, TPUGeneration.V4,
                    TPUGeneration.V5E, TPUGeneration.V5P):
            t = TPUTargetProfile(generation=gen)
            assert t.mxu_tile == 128

    def test_num_chips_from_mesh(self):
        t = TPUTargetProfile(
            generation=TPUGeneration.V4,
            mesh_axes={"data": 4, "model": 2}
        )
        assert t.num_chips == 8

    def test_single_chip_default(self):
        t = TPUTargetProfile()
        assert t.num_chips == 1

    def test_unknown_generation_raises(self):
        with pytest.raises(ValueError, match="Unknown TPU generation"):
            TPUTargetProfile(generation="v99")

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError, match="Unsupported TPU dtype"):
            TPUTargetProfile(dtype="fp8")


class TestValidateMatmulDims:
    def test_valid_dims_pass(self):
        t = TPUTargetProfile()
        t.validate_matmul_dims(M=512, N=1024, K=4096)  # no exception

    def test_m_not_divisible_raises(self):
        t = TPUTargetProfile()
        with pytest.raises(ValueError, match="M=500"):
            t.validate_matmul_dims(M=500, N=1024, K=4096)

    def test_n_not_divisible_raises(self):
        t = TPUTargetProfile()
        with pytest.raises(ValueError, match="N=100"):
            t.validate_matmul_dims(M=512, N=100, K=128)

    def test_k_not_divisible_raises(self):
        t = TPUTargetProfile()
        with pytest.raises(ValueError, match="K="):
            t.validate_matmul_dims(M=128, N=128, K=64)

    def test_error_message_includes_nearest(self):
        t = TPUTargetProfile()
        with pytest.raises(ValueError, match="Nearest valid value: 128"):
            t.validate_matmul_dims(M=100, N=128, K=128)

    def test_all_128_multiples_pass(self):
        t = TPUTargetProfile()
        for size in (128, 256, 512, 1024, 2048, 4096):
            t.validate_matmul_dims(M=size, N=size, K=size)


class TestTPUAutoConstraints:
    def test_auto_constraints_keys(self):
        t = TPUTargetProfile()
        c = t.auto_constraints()
        assert set(c.keys()) == {"M", "N", "K"}

    def test_auto_constraints_value_is_mxu_tile(self):
        t = TPUTargetProfile()
        c = t.auto_constraints()
        assert c["M"] == t.mxu_tile
        assert c["N"] == t.mxu_tile
        assert c["K"] == t.mxu_tile


class TestTPUMLIRAttrs:
    def test_to_mlir_attrs_contains_generation(self):
        t = TPUTargetProfile(generation=TPUGeneration.V5P)
        attr = t.to_mlir_attrs()
        assert "v5p" in attr

    def test_to_mlir_attrs_contains_mxu_tile(self):
        t = TPUTargetProfile()
        attr = t.to_mlir_attrs()
        assert "128" in attr

    def test_to_mlir_attrs_contains_mesh(self):
        t = TPUTargetProfile(mesh_axes={"data": 4, "model": 2})
        attr = t.to_mlir_attrs()
        assert "data" in attr
        assert "model" in attr

    def test_to_mlir_attrs_is_string(self):
        t = TPUTargetProfile()
        assert isinstance(t.to_mlir_attrs(), str)
