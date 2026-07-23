from benchmarks.norm_backward_cross_target import _artifact, _row


def test_backward_benchmark_artifacts_pin_paired_contracts() -> None:
    rms = _artifact("x86", "tessera.rmsnorm", False).metadata
    layer = _artifact("rocm", "tessera.layer_norm", True).metadata
    assert rms["compiler_path"] == "x86_rmsnorm_bwd_compiled"
    assert rms["arg_names"] == ["x", "dy"]
    assert layer["compiler_path"] == "rocm_layer_norm_bwd_compiled"
    assert layer["arg_names"] == ["x", "gamma", "beta", "dy"]
    assert layer["output_names"] == ["dx", "dgamma", "dbeta"]
    assert layer["autodiff_phase"] == "backward"


def test_rocm_affine_benchmark_records_deterministic_reduction(monkeypatch) -> None:
    import numpy as np

    from benchmarks import norm_backward_cross_target as bench

    def fake_launch(_artifact, _args):
        return {
            "ok": True,
            "output": (
                np.zeros((2, 3), np.float32),
                np.zeros(3, np.float32),
            ),
        }

    monkeypatch.setattr(bench.rt, "launch", fake_launch)
    row = _row("rocm", "tessera.rmsnorm", True, (2, 3), 1, 2)
    assert row["affine_reduction"] == "fixed_row_order_two_pass"
    assert row["affine_bitwise_reproducible"] is True
