from __future__ import annotations

import pytest

from tessera.compiler.frontend import lower_text_to_graph_ir
from tessera.compiler.schedule_ir import lower_graph_to_schedule_ir
from tessera.compiler.target_ir import (
    TargetFunction,
    TargetIRModule,
    TargetIRVerificationError,
    TargetOp,
    lower_tile_to_target_ir,
)
from tessera.compiler.tile_ir import TileFunction, TileIRModule, TileOp, lower_schedule_to_tile_ir


def test_target_ir_renders_and_verifies_apple_cpu_contracts():
    module = TargetIRModule(
        attrs={"tessera.ir.level": "target", "target": "apple_cpu", "arch": "arm64-apple-silicon"},
        functions=[
            TargetFunction(
                "main",
                target="apple_cpu",
                body=[
                    TargetOp("tessera_apple.cpu.accelerate_gemm", {
                        "source": "tessera.matmul",
                        "result": "C",
                        "ordinal": 0,
                        "framework": "Accelerate",
                        "abi": "cblas_sgemm",
                        "dtype": "f32",
                    })
                ],
            )
        ],
    )
    assert module.verify().ok
    text = module.to_mlir()
    assert "tessera_apple.cpu.accelerate_gemm" in text
    assert 'framework = "Accelerate"' in text


def test_target_ir_verifier_rejects_missing_backend_contract_attrs():
    module = TargetIRModule(
        attrs={"tessera.ir.level": "target", "target": "rocm"},
        functions=[
            TargetFunction("bad", target="rocm", body=[TargetOp("tessera_rocm.mfma", {"source": "tessera.matmul"})])
        ],
    )
    result = module.verify()
    assert not result.ok
    assert "TARGET_IR_MISSING_ATTR" in result.format()
    with pytest.raises(TargetIRVerificationError, match="TARGET_IR_MISSING_ATTR"):
        module.to_mlir()


def test_lower_tile_to_rocm_target_ir_maps_mma_to_mfma_dma_wait():
    tile = TileIRModule(functions=[
        TileFunction(
            "main",
            body=[TileOp("tile.mma", {"source": "tessera.matmul", "result": "C", "ordinal": 0})],
        )
    ])
    target = lower_tile_to_target_ir(tile, target_kind="rocm")
    assert target.verify().ok
    text = target.to_mlir()
    assert 'target = "rocm"' in text
    assert "tessera_rocm.mfma" in text
    assert "tessera_rocm.async_copy" in text
    assert "tessera_rocm.wait" in text


def test_lower_tile_to_cpu_target_ir_maps_mma_and_elementwise_to_x86_numpy_contracts():
    tile = TileIRModule(functions=[
        TileFunction(
            "main",
            body=[
                TileOp("tile.mma", {"source": "tessera.matmul", "result": "C", "ordinal": 0}),
                TileOp("tile.relu", {"source": "tessera.relu", "result": "R", "ordinal": 1}),
            ],
        )
    ])
    target = lower_tile_to_target_ir(tile, target_kind="cpu")
    assert target.verify().ok
    text = target.to_mlir()
    assert 'target = "cpu"' in text
    assert 'arch = "x86_64"' in text
    assert "tessera.cpu.matmul" in text
    assert "tessera.cpu.relu" in text
    assert 'abi = "numpy"' in text


def test_lower_tile_to_nvidia_hopper_target_ir_maps_mma_to_wgmma_tma_mbarrier():
    tile = TileIRModule(functions=[
        TileFunction(
            "main",
            body=[TileOp("tile.mma", {"source": "tessera.matmul", "result": "C", "ordinal": 0})],
            target="nvidia_sm90",
        )
    ])
    target = lower_tile_to_target_ir(tile, target_kind="nvidia_sm90")
    assert target.verify().ok
    text = target.to_mlir()
    assert 'target = "nvidia_sm90"' in text
    assert 'arch = "sm_90a"' in text
    assert "target_features" in text
    assert "launch" in text
    assert "tessera_nvidia.wgmma" in text
    assert "tessera_nvidia.tma_async_copy" in text
    assert "tessera_nvidia.mbarrier" in text


def test_lower_tile_to_nvidia_blackwell_target_ir_maps_mma_to_tcgen05_tmem():
    tile = TileIRModule(functions=[
        TileFunction(
            "main",
            body=[TileOp("tile.mma", {"source": "tessera.matmul", "result": "C", "ordinal": 0})],
            target="nvidia_sm100",
        )
    ])
    target = lower_tile_to_target_ir(tile, target_kind="nvidia_sm100")
    assert target.verify().ok
    text = target.to_mlir()
    assert 'arch = "sm_100a"' in text
    assert "tessera_nvidia.tmem_alloc" in text
    assert "tessera_nvidia.tcgen05_mma" in text
    assert "block_scaled = true" in text


def test_lower_tile_to_apple_gpu_target_ir_maps_fa4_to_metal_contract():
    tile = TileIRModule(functions=[
        TileFunction(
            "flash",
            body=[
                TileOp("tessera.attn.online_softmax", {
                    "source": "tessera.flash_attn",
                    "result": "O",
                    "ordinal": 0,
                    "policy": "safe",
                })
            ],
        )
    ])
    target = lower_tile_to_target_ir(tile, target_kind="apple_gpu")
    assert target.verify().ok
    text = target.to_mlir()
    assert "tessera_apple.gpu.metal_kernel" in text
    assert 'kernel = "flash_attn_contract"' in text
    assert "tessera_apple.gpu.dispatch" in text
    assert 'artifact = "metallib"' in text


def test_frontend_to_rocm_target_ir_reports_kv_cache_diagnostic_through_full_spine():
    graph = lower_text_to_graph_ir("""
    module decode {
      func step(Cache: tensor<?xfp32>, K: tensor<?xfp32>, V: tensor<?xfp32>) -> tensor<?xfp32> {
        C = op.kv_cache_append(Cache, K, V);
        return C;
      }
    }
    """)
    schedule = lower_graph_to_schedule_ir(graph, target_kind="rocm")
    tile = lower_schedule_to_tile_ir(schedule, target_kind="rocm")
    target = lower_tile_to_target_ir(tile, target_kind="rocm")
    assert target.verify().ok
    text = target.to_mlir()
    assert "tessera.target.diagnostic" in text
    assert "KV-cache target lowering is not implemented" in text


def test_frontend_to_apple_cpu_target_ir_maps_matmul_and_softmax():
    graph = lower_text_to_graph_ir("""
    module demo {
      func main(A: tensor<2x3xfp32>, B: tensor<3x2xfp32>) -> tensor<2x2xfp32> {
        C = op.matmul(A, B);
        P = op.softmax(C);
        return P;
      }
    }
    """)
    schedule = lower_graph_to_schedule_ir(graph, target_kind="apple_cpu")
    tile = lower_schedule_to_tile_ir(schedule, target_kind="apple_cpu")
    target = lower_tile_to_target_ir(tile, target_kind="apple_cpu")
    assert target.verify().ok
    text = target.to_mlir()
    assert "tessera_apple.cpu.accelerate_gemm" in text
    assert "tessera_apple.cpu.vector_reduce" in text
    assert "target_features" in text
    assert "launch" in text


def test_frontend_to_nvidia_target_ir_maps_flash_attention_contract_through_full_spine():
    graph = lower_text_to_graph_ir("""
    module demo {
      func flash(Q: tensor<2x4xfp32>, K: tensor<2x4xfp32>, V: tensor<2x4xfp32>) -> tensor<2x4xfp32> {
        O = op.flash_attn(Q, K, V);
        return O;
      }
    }
    """)
    schedule = lower_graph_to_schedule_ir(graph, target_kind="nvidia_sm90")
    tile = lower_schedule_to_tile_ir(schedule, target_kind="nvidia_sm90")
    target = lower_tile_to_target_ir(tile, target_kind="nvidia_sm90")
    assert target.verify().ok
    text = target.to_mlir()
    assert "tessera_nvidia.cuda_kernel" in text
    assert 'kernel = "flash_attn_contract"' in text
    assert 'status = "artifact_only"' in text
