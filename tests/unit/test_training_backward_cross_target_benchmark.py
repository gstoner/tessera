from benchmarks.training_backward_cross_target import (
    _loss_artifact,
    _sgd_artifact,
)


def test_training_backward_benchmark_pins_native_routes() -> None:
    loss = _loss_artifact("rocm", "smooth_l1").metadata
    sgd = _sgd_artifact("x86").metadata
    assert loss["compiler_path"] == "rocm_regression_loss_bwd_compiled"
    assert loss["autodiff_phase"] == "backward"
    assert loss["ops"][0]["kwargs"]["beta"] == 0.5
    assert sgd["compiler_path"] == "x86_sgd_bwd_compiled"
    assert sgd["autodiff_phase"] == "backward"
    assert sgd["ops"][0]["kwargs"]["lr"] == 0.125
