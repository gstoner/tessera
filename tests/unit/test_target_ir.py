from __future__ import annotations

import pytest

from tessera.compiler.frontend import lower_text_to_graph_ir
from tessera.compiler.schedule_ir import lower_graph_to_schedule_ir
from tessera.compiler.target_ir import (
    TargetFunction,
    TargetIRModule,
    TargetIRVerificationError,
    TargetOp,
    annotate_target_ir_with_probes,
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


def test_lower_tile_to_exact_x86_target_uses_native_contracts():
    tile = TileIRModule(functions=[
        TileFunction(
            "main",
            target="x86",
            body=[
                TileOp("tile.mma", {
                    "source": "tessera.matmul", "result": "C", "ordinal": 0,
                }),
                TileOp("tile.kv_cache.read", {
                    "source": "tessera.kv_cache.read", "result": "slice",
                    "ordinal": 1, "effect": "read", "access": "paged_slice",
                    "storage": "paged",
                }),
            ],
        )
    ])
    target = lower_tile_to_target_ir(tile, target_kind="x86")
    assert target.verify().ok
    text = target.to_mlir()
    assert 'target = "x86"' in text
    assert 'execution_mode = "native_cpu"' in text
    assert "tessera_x86.kernel" in text
    assert "tessera_x86.kv_cache_read" in text
    assert 'abi = "libtessera_x86_elementwise"' in text
    assert 'abi = "tessera_x86_kv_cache_read_f32"' in text
    assert 'runtime_lane = "x86_kv_cache_compiled"' in text
    assert 'access = "paged_slice"' in text


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


def test_annotate_target_ir_with_intra_kernel_probes_adds_backend_markers():
    tile = TileIRModule(functions=[
        TileFunction(
            "main",
            body=[TileOp("tile.mma", {"source": "tessera.matmul", "result": "C", "ordinal": 0})],
            target="nvidia_sm90",
        )
    ])
    target = lower_tile_to_target_ir(tile, target_kind="nvidia_sm90")

    annotated = annotate_target_ir_with_probes(target)

    assert annotated.verify().ok
    text = annotated.to_mlir()
    assert "tessera_nvidia.profiler_probe" in text
    assert 'kernel = "matmul"' in text
    assert 'probe_name = "matmul.prologue"' in text
    assert 'backend_correlation_key = "matmul.mainloop"' in text
    assert 'source_op = "matmul"' in text
    assert 'schedule = "target_ir"' in text
    assert 'phase = "prologue"' in text
    assert 'phase = "mainloop"' in text
    assert 'phase = "epilogue"' in text
    assert "profiling_probes" in text


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


def test_lower_tile_to_nvidia_sm120_target_ir_maps_mma_to_warp_level_mma_sync():
    """Consumer Blackwell (sm_120) is NOT a superset of datacenter sm_100: it has
    no tcgen05/TMEM and no Hopper wgmma. tile.mma must lower to warp-level
    `mma.sync` (+ tma_async_copy + mbarrier), never tcgen05_mma/tmem_alloc."""
    tile = TileIRModule(functions=[
        TileFunction(
            "main",
            body=[TileOp("tile.mma", {"source": "tessera.matmul", "result": "C", "ordinal": 0})],
            target="nvidia_sm120",
        )
    ])
    target = lower_tile_to_target_ir(tile, target_kind="nvidia_sm120")
    assert target.verify().ok
    text = target.to_mlir()
    assert 'arch = "sm_120"' in text
    assert "tessera_nvidia.mma_sync" in text
    assert "tessera_nvidia.tma_async_copy" in text
    assert "tessera_nvidia.mbarrier" in text
    # sm_120 has NO tcgen05/TMEM — those are datacenter sm_100a only.
    assert "tessera_nvidia.tcgen05_mma" not in text
    assert "tessera_nvidia.tmem_alloc" not in text


@pytest.mark.parametrize(
    "dtype,shape,dtype_c",
    [("f16", "m16n8k16", "f32"), ("fp16", "m16n8k16", "f32"),
     ("tf32", "m16n8k8", "f32"),
     ("fp8_e4m3", "m16n8k32", "f32"),
     ("fp8_e5m2", "m16n8k32", "f32"), ("int8", "m16n8k32", "s32")],
)
def test_lower_tile_to_nvidia_sm120_selects_dtype_specific_mma_contract(
        dtype, shape, dtype_c):
    tile = TileIRModule(functions=[TileFunction(
        "main", body=[TileOp("tile.mma", {
            "source": "tessera.matmul", "result": "C", "ordinal": 0,
            "dtype": dtype})], target="nvidia_sm120")])
    target = lower_tile_to_target_ir(tile, target_kind="nvidia_sm120")
    assert target.verify().ok
    text = target.to_mlir()
    assert f'shape = "{shape}"' in text
    assert f'dtype_c = "{dtype_c}"' in text


def test_lower_tile_to_nvidia_sm120_keeps_f32_exact_not_implicit_tf32():
    tile = TileIRModule(functions=[TileFunction(
        "main", body=[TileOp("tile.mma", {
            "source": "tessera.matmul", "result": "C", "ordinal": 0,
            "dtype": "f32"})], target="nvidia_sm120")])
    target = lower_tile_to_target_ir(tile, target_kind="nvidia_sm120")
    assert target.verify().ok
    text = target.to_mlir()
    assert "tessera_nvidia.cuda_kernel" in text
    assert 'kernel = "matmul_f32_contract"' in text
    assert 'dtype_ab = "f32"' in text
    assert "tensor_core = false" in text
    assert "tessera_nvidia.mma_sync" not in text
    assert 'dtype_ab = "tf32"' not in text


def test_lower_tile_to_nvidia_sm120_nvfp4_requires_and_preserves_scales():
    attrs = {"source": "tessera.matmul", "result": "C", "ordinal": 0,
             "dtype": "nvfp4", "scale_a": "SFa", "scale_b": "SFb"}
    tile = TileIRModule(functions=[TileFunction(
        "main", body=[TileOp("tile.mma", attrs)], target="nvidia_sm120")])
    target = lower_tile_to_target_ir(tile, target_kind="nvidia_sm120")
    assert target.verify().ok
    text = target.to_mlir()
    assert "tessera_nvidia.nvfp4_block_scale_mma" in text
    assert 'arch = "sm_120a"' in text
    assert 'scale_a = "SFa"' in text and 'scale_b = "SFb"' in text
    assert 'scale_dtype = "ue4m3"' in text and 'scale_vector = "4X"' in text

    missing = TileIRModule(functions=[TileFunction(
        "main", body=[TileOp("tile.mma", {**attrs, "scale_b": ""})],
        target="nvidia_sm120")])
    with pytest.raises(ValueError, match="requires logical scale_a and scale_b"):
        lower_tile_to_target_ir(missing, target_kind="nvidia_sm120")
    assert "tessera_nvidia.tcgen05_mma" not in text
    assert "tessera_nvidia.tmem_alloc" not in text


def test_lower_tile_to_apple_gpu_target_ir_maps_fa4_to_msl_runtime_contract():
    """Phase 8.4.1 — a single-source flash_attn tile module qualifies for the
    runtime path; the lowering emits the MSL kernel + mps_dispatch contract
    with execution_mode='metal_runtime'."""

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
    assert "tessera_apple.gpu.msl_kernel" in text
    assert 'entry_point = "flash_attn_f32"' in text
    assert "tessera_apple.gpu.mps_dispatch" in text
    assert 'execution_mode = "metal_runtime"' in text


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
