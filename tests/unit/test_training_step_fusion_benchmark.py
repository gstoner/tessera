from benchmarks.training_step_fusion_cross_target import (
    _adamw_artifact,
    _fused_artifact,
    _loss_artifact,
    _sgd_artifact,
)


def test_training_fusion_benchmark_pins_measured_routes() -> None:
    fused = _fused_artifact("rocm", "smooth_l1", "sgd").metadata
    fused_adamw = _fused_artifact("x86", "bce", "adamw").metadata
    loss = _loss_artifact("x86", "huber").metadata
    sgd = _sgd_artifact("rocm").metadata
    assert fused["compiler_path"] == "rocm_training_loss_sgd_compiled"
    assert fused["ops"][0]["kwargs"]["parameter"] == 0.5
    assert fused["ops"][0]["kwargs"]["lr"] == 0.125
    assert (
        fused_adamw["compiler_path"]
        == "x86_training_loss_adamw_compiled"
    )
    assert fused_adamw["ops"][0]["kwargs"]["step"] == 7
    assert fused_adamw["ops"][0]["operands"][-2:] == ["moment1", "moment2"]
    assert loss["compiler_path"] == "x86_regression_loss_bwd_compiled"
    assert loss["ops"][0]["kwargs"]["delta"] == 0.75
    assert sgd["compiler_path"] == "rocm_optimizer_compiled"
    assert sgd["ops"][0]["kwargs"]["extras"] == []
    adamw = _adamw_artifact("rocm").metadata
    assert adamw["ops"][0]["kwargs"]["extras"] == ["m", "v"]
    bce = _loss_artifact("rocm", "bce").metadata
    assert bce["compiler_path"] == "rocm_binary_loss_bwd_compiled"
    assert bce["ops"][0]["op_name"] == "tessera.binary_cross_entropy_loss"
