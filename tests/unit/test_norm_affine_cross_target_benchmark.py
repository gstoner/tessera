from benchmarks.norm_affine_cross_target import _artifact


def test_benchmark_artifacts_cover_unary_and_affine_contracts() -> None:
    unary = _artifact("x86", "tessera.rmsnorm", False).metadata
    affine = _artifact("rocm", "tessera.layer_norm", True).metadata
    assert unary["arg_names"] == ["x"]
    assert affine["arg_names"] == ["x", "gamma", "beta"]
    assert affine["compiler_path"] == "rocm_norm_compiled"
