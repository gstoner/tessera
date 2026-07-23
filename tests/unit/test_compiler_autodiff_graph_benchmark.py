from benchmarks.compiler_autodiff_graph import OPS, render_module


def test_benchmark_modules_cover_native_normalization_cohort() -> None:
    assert {"relu", "rmsnorm", "rmsnorm_affine", "layer_norm",
            "layer_norm_affine", "softmax"} <= set(OPS)
    for op in OPS:
        text = render_module(op, "?x128")
        assert f'"tessera.{op.removesuffix("_affine")}"' in text
        assert "tessera.autodiff = \"reverse\"" in text
        assert "tensor<?x128xf32>" in text
