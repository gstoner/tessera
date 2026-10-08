"""Host-independent refusal of scale-policy substitutions at package binding."""
import pytest

from tessera.compiler.rocm_fp8_blockscale import BlockScaleShape, _scale_profile, check_blockscale_target_ir
from tessera.compiler.rocm_mxfp8_blockscale import author_mxfp8_graph, MXFP8_CONTRACTS, MXFP8_PACKAGE_ABIS


def texts():
    contract = MXFP8_CONTRACTS["kn"]
    carrier = ('tile.scaled_matmul_kernel physical_contract = "' + contract + '", '
               'combine = "scale_outer_product_then_add", init = "zero", scope = "scale_group", '
               'cross_step_motion = "forbid", tessera.scale_block_n = 1, tessera.schedule_hash = "test"')
    directive = ('tessera_rocm.scaled_wmma_gemm abi = "a_b_lhs_scale_rhs_scale_d_m_n_k", '
        f'physical_contract = "{contract}", package_abi = "{MXFP8_PACKAGE_ABIS[("kn", "f32")]}", '
        'scale_format = "e8m0", partial_combine = "scale_outer_product_then_add", '
        'k_step_schedule = "isolated_scale_group", output = "f32", m = 17, n = 19, k = 64, '
        'instruction_k = 16, scale_k = 32, scale_n = 1, macro_k = 32, block_m = 16, block_n = 16, '
        'staging = "global", warps = 1, pipeline_depth = 1, '
        'numeric_policy = {accum = "f32", storage = "e4m3", execution_mode = "exact_per_block"}, '
        'tessera.schedule_hash = "test"')
    return carrier, directive


@pytest.mark.parametrize("before,after", [
    ('scale_format = "e8m0"', 'scale_format = "fp32"'),
    ('scale_k = 32', 'scale_k = 16'),
    ('scale_n = 1', 'scale_n = 2'),
    ('macro_k = 32', 'macro_k = 64'),
    ('block_m = 16', 'block_m = 32'),
    ('accum = "f32"', 'accum = "f16"'),
    ('execution_mode = "exact_per_block"', 'execution_mode = "approximate"'),
    ('staging = "global"', 'staging = "lds"'),
    ('wide_scale.v1', 'wmma_exact.v1'),
])
def test_mxfp8_target_contract_is_not_normalized(before, after):
    shape = BlockScaleShape(17, 19, 64, 32, 1)
    carrier, directive = texts()
    with pytest.raises(ValueError):
        check_blockscale_target_ir(shape, carrier, directive.replace(before, after), scale_format="e8m0")


def test_mxfp8_does_not_enter_fp32_package_profile():
    carrier, directive = texts()
    shape = BlockScaleShape(17, 19, 64, 32, 1)
    with pytest.raises(ValueError):
        check_blockscale_target_ir(shape, carrier, directive)
    assert check_blockscale_target_ir(shape, carrier, directive, scale_format="e8m0")["macro_k"] == 32


@pytest.mark.parametrize("sk,sn", [(16, 1), (32, 2), (128, 128)])
def test_mxfp8_frontend_refuses_conflicting_groups(sk, sn):
    shape = BlockScaleShape(16, 16, 128, sk, sn)
    with pytest.raises(ValueError):
        author_mxfp8_graph(shape)
    with pytest.raises(ValueError):
        _scale_profile(shape, "e8m0")

@pytest.mark.parametrize("layout,staging,bm,bn,warps,valid", [
    ("nk", "lds", 128, 64, 8, True),
    ("nk", "lds", 128, 128, 8, True),
    ("kn", "lds", 128, 64, 8, False),
    ("nk", "lds", 64, 128, 8, False),
    ("nk", "lds", 128, 64, 4, False),
    ("nk", "lds", 128, 256, 8, False),
    ("nk", "global", 16, 16, True, False),
])
def test_mxfp8_profiles_do_not_admit_unproved_geometry(layout, staging, bm, bn, warps, valid):
    from tessera.compiler.rocm_mxfp8_blockscale import mxfp8_schedule_is_supported
    assert mxfp8_schedule_is_supported(layout=layout, staging=staging, block_m=bm,
        block_n=bn, macro_k=32, warps=warps, pipeline_depth=1) is valid

def test_mxfp8_schedule_intent_is_frontend_metadata_not_a_constructed_kernel():
    shape = BlockScaleShape(200, 2048, 128, 32, 1, "nk", "f32")
    for policy in ("seed", "lds"):
        graph = author_mxfp8_graph(shape, schedule_policy=policy)
        assert f'tessera.rocm.mxfp8_schedule = "{policy}"' in graph
        assert "tessera.scaled_matmul" in graph and "tile." not in graph
    assert "tessera.rocm.mxfp8_schedule" not in author_mxfp8_graph(shape)
    with pytest.raises(ValueError):
        author_mxfp8_graph(shape, schedule_policy="fast")
    with pytest.raises(ValueError):
        author_mxfp8_graph(BlockScaleShape(200, 2048, 128, 32, 1, "kn"), schedule_policy="lds")

@pytest.mark.parametrize("value", ['"invalid"', '7 : i64'])
def test_native_mxfp8_policy_refuses_unknown_or_wrong_type(value):
    import subprocess
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("matching native compiler is required")
    graph = author_mxfp8_graph(BlockScaleShape(200, 2048, 128, 32, 1, "nk"),
                              schedule_policy="lds")
    graph = graph.replace('tessera.rocm.mxfp8_schedule = "lds"',
                          f'tessera.rocm.mxfp8_schedule = {value}')
    done = subprocess.run([str(tool), "-", "--tessera-graph-to-schedule"],
                          input=graph, text=True, capture_output=True)
    assert done.returncode != 0
    assert "ROCM_FP8_BLOCKSCALE_CONTRACT" in done.stderr

@pytest.mark.parametrize("k", [32, 96])
def test_mxfp8_k64_intent_requires_whole_slabs(k):
    with pytest.raises(ValueError, match="whole K64"):
        author_mxfp8_graph(BlockScaleShape(17, 19, k, 32, 1, "nk"),
                          schedule_policy="lds_k64")


def test_native_k64_slab_keeps_semantic_k32_groups():
    from tessera.compiler.rocm_mxfp8_blockscale import lower_mxfp8
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    if find_tessera_opt() is None:
        pytest.skip("matching native compiler is required")
    program = lower_mxfp8(BlockScaleShape(17, 19, 192, 32, 1, "nk"),
                           schedule_policy="lds_k64")
    assert "block_k = 64" in program.schedule_ir
    assert "scale_k = 32" in program.schedule_ir
    assert 'staging = "lds"' in program.schedule_ir
    assert "tessera.scale_block_n = 1" in program.tile_ir


def test_native_seed_policy_decodes_omitted_global_default():
    from tessera.compiler.rocm_mxfp8_blockscale import lower_mxfp8
    from tessera.compiler.rocm_fp8_blockscale import schedule_blockscale_panel
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    if find_tessera_opt() is None:
        pytest.skip("matching native compiler is required")
    program = lower_mxfp8(BlockScaleShape(200, 1024, 4096, 32, 1, "nk"),
                         schedule_policy="seed")
    panel = schedule_blockscale_panel(program.schedule_ir)
    assert (panel.staging, panel.macro_tile_m, panel.macro_tile_n, panel.warps) == ("global", 16, 16, 1)
    assert "tile.scaled_matmul_kernel" in program.tile_ir


@pytest.mark.parametrize("m,n,k,slab", [
    (200,1024,4096,64), (256,1024,2560,64), (256,1024,5120,64),
    (400,512,3584,64), (300,1024,3584,64), (512,1024,2560,64),
    (200,2048,5120,64), (200,1025,3136,64),
    (128,1024,3072,32), (256,512,3072,32), (128,2048,3584,32),
    (256,4096,3072,32), (256,1024,2048,32), (256,1024,5184,32),
    (600,1024,4096,32), (200,1024,2592,32),
])
def test_native_mxfp8_long_k_selection_preserves_controls(m,n,k,slab):
    from tessera.compiler.rocm_mxfp8_blockscale import lower_mxfp8
    from tessera.compiler.scheduled_matmul import find_tessera_opt
    if find_tessera_opt() is None:
        pytest.skip("matching native compiler is required")
    program = lower_mxfp8(BlockScaleShape(m,n,k,32,1,"nk"))
    assert f"block_k = {slab}" in program.schedule_ir
    assert "scale_k = 32" in program.schedule_ir
    assert "tile.scaled_matmul_kernel" in program.tile_ir
