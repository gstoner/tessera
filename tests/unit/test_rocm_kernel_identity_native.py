"""Native MLIR projection coverage; cache test doubles are not compiler proof."""
from __future__ import annotations

import pytest
from tessera.compiler import rocm_native
from tests.unit.test_rocm_matmul_shape_key import _target
from tests.unit.test_rocm_shape_free_cache_key import _target as _reduce_target, _REDUCE_ATTRS

@pytest.fixture(autouse=True)
def native_rocm_compiler():
    tool = rocm_native._tessera_opt()
    if tool is None:
        pytest.skip("requires a ROCm-enabled tessera-opt")
    import subprocess
    help_text = subprocess.run([str(tool), "--help"], capture_output=True, text=True, check=True).stdout
    if "tessera-rocm-project-kernel-identity" not in help_text:
        pytest.skip("requires the native ROCm kernel identity pass")

def _project(source, family="matmul"):
    return rocm_native._shape_free_target_ir(
        source, family=family, directive=rocm_native._SHAPE_FREE_DIRECTIVES[family])

def test_native_shape_symbol_and_ancestry_independence():
    a = _project(_target(16, "a" * 64))
    b = _project(_target(48, "b" * 64))
    assert a == b
    assert "func.func" not in a and "schedule_hash" not in a
    assert rocm_native._directive_symbol(a, "tessera_rocm.wmma_gemm").startswith("tessera_rocm_matmul_")
    # Physical attributes are not shape-bound provenance.
    assert a != _project(_target(16, "a" * 64, mt=4))

def test_native_module_attributes_key_the_image():
    a = _target(16, "a" * 64)
    assert _project(a) != _project(a.replace('tessera.arch = "gfx1151"', 'tessera.arch = "gfx1201"'))

@pytest.mark.parametrize("key,value", [
    ("dtype", '"f16"'), ("kind", '"max"'), ("keepdims", "true"),
    ("nan_mode", '"ignore"'), ("axis", "2 : i64"),
])
def test_native_reduction_physical_attributes_key_the_image(key, value):
    a = _project(_reduce_target((2, 3, 17)), "reduction")
    b = _project(_reduce_target((2, 3, 17), attrs=dict(_REDUCE_ATTRS, **{key: value})), "reduction")
    assert a != b
    assert "func.func" not in b

def test_native_reduction_shape_independence_and_idempotence():
    a = _project(_reduce_target((2, 3, 17)), "reduction")
    assert a == _project(_reduce_target((2, 8, 64), symbol="other"), "reduction")
    assert a == _project(a, "reduction")

@pytest.mark.parametrize("edit,match", [
    (lambda s: s.replace(", tessera.schedule_hash = " + chr(34) + "a" * 64 + chr(34), ""), "Schedule hash"),
    (lambda s: s.replace("    return", "    %x = arith.addi %c16, %c16 : i64\n    return"), "unaudited Target operation"),
    (lambda s: s.replace("a" * 64, "z" * 64), "Schedule hash"),
    (lambda s: s.replace("    return", '    tessera_rocm.wmma_gemm {m = 16 : i64, n = 16 : i64, k = 16 : i64, name = "other"}\n    return'), "exactly one"),
])
def test_native_projection_refuses_unproved_input(edit, match):
    with pytest.raises(RuntimeError, match=match):
        _project(edit(_target(16, "a" * 64)))

def test_driver_refuses_family_directive_mismatch():
    with pytest.raises(ValueError, match="family/directive mismatch"):
        rocm_native._shape_free_target_ir(_target(16, "a" * 64), family="matmul", directive="tessera_rocm.reduce")


@pytest.mark.parametrize("attr", ['bias = true', 'activation = "relu"', 'dtype = "bf16"'])
def test_native_matmul_epilogue_and_storage_key_the_image(attr):
    source = _target(16, "a" * 64)
    changed = source.replace('arch = "gfx1151", k', 'arch = "gfx1151", ' + attr + ', k')
    assert _project(source) != _project(changed)

def test_native_macro_k_and_split_partition_remain_image_identity():
    source = _target(16,"a"*64)
    macro = source.replace("tessera.schedule_hash =", "k_blocks = 2 : i64, tessera.schedule_hash =")
    assert macro!=source
    assert _project(macro)!=_project(source)
    partition = macro.replace("tessera.schedule_hash =",
        'split_k = 8 : i64, split_k_reduction = "ordered", problem_k = 2048 : i64, tessera.schedule_hash =')
    projected = _project(partition)
    assert "problem_k = 2048 : i64" in projected
    assert "k_blocks = 2 : i64" in projected
    assert projected!=_project(partition.replace("problem_k = 2048","problem_k = 4096"))

def _scaled_target(layout="kn", output="f32"):
    from tests.unit.test_rocm_fp8_blockscale import _pair
    return "module {\n" + _pair(layout, output)[1].replace('"h0"', '"'+ "a"*64 + '"') + "\n}"

@pytest.mark.parametrize("layout", ["kn", "nk"])
@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_native_register_blockscale_runtime_shape_identity(layout, output):
    source = _scaled_target(layout, output)
    a = _project(source, "scaled_matmul")
    b = _project(source.replace("m = 64", "m = 17")
        .replace("n = 96", "n = 23").replace("k = 256", "k = 384")
        .replace("a"*64, "b"*64), "scaled_matmul")
    assert a == b
    assert "runtime_shape" in a and "schedule_hash" not in a
    assert "m = 0 : i64" in a and "n = 0 : i64" in a and "k = 0 : i64" in a
    assert a != _project(source.replace("scale_n = 128", "scale_n = 64"), "scaled_matmul")

@pytest.mark.parametrize("old,new", [
    ('staging = "global"', 'staging = "lds"'),
    ('execution_mode = "exact_per_block"', 'execution_mode = "approximate"'),
    ('k = 256', 'k = 257'),
    ('macro_k = 128', 'macro_k = 64'),
    ('warps = 1', 'warps = 2'),
    ('physical_contract = "rocm_fp8_w8a8_blockscale_v1"',
     'physical_contract = "rocm_mxfp4_w4a8_folded_prefill_v1"'),
])
def test_native_blockscale_projection_preserves_static_only_contracts(old, new):
    with pytest.raises(RuntimeError, match="ROCM_FP8_BLOCKSCALE_CONTRACT"):
        _project(_scaled_target().replace(old,new), "scaled_matmul")

def test_native_scaled_raster_is_a_physical_image_key():
    source = _scaled_target()
    grouped = source.replace('name = "w"', 'name = "w", schedule_raster_order = "grouped_m", schedule_raster_group = 3 : i64')
    assert _project(source, "scaled_matmul") != _project(grouped, "scaled_matmul")
    assert _project(grouped, "scaled_matmul") != _project(
        grouped.replace("schedule_raster_group = 3", "schedule_raster_group = 2"),
        "scaled_matmul")
    with pytest.raises(RuntimeError, match="ROCM_FP8_BLOCKSCALE_CONTRACT"):
        _project(grouped.replace('grouped_m', 'unsupported'), "scaled_matmul")

def _scaled_lds_target(m=200, n=8192, k=256, output="f32"):
    from tests.unit.test_rocm_fp8_blockscale import _pair
    target = _pair("nk", output, staging="lds", warps=8, block=(128,128))[1]
    return "module {\n" + target.replace('"h0"', '"'+ "a"*64 + '"').replace(
        "m = 64 : i64", f"m = {m} : i64").replace(
        "n = 96 : i64", f"n = {n} : i64").replace(
        "k = 256 : i64", f"k = {k} : i64") + "\n}"

@pytest.mark.parametrize("output", ["f32","bf16"])
def test_native_lds_blockscale_mn_identity_keeps_k_and_edge_class(output):
    a = _project(_scaled_lds_target(output=output), "scaled_matmul_lds")
    b = _project(_scaled_lds_target(328,10240,output=output), "scaled_matmul_lds")
    assert a == b
    assert "runtime_mn" in a and "runtime_shape" not in a
    assert "m = 0 : i64" in a and "n = 0 : i64" in a
    assert "k = 256 : i64" in a and "whole_m = false" in a and "whole_n = true" in a
    for source in (_scaled_lds_target(k=384,output=output),
                   _scaled_lds_target(m=256,output=output),
                   _scaled_lds_target(n=8191,output=output)):
        assert a != _project(source,"scaled_matmul_lds")

def test_native_lds_projection_refuses_global_and_incomplete_wave_panels():
    with pytest.raises(RuntimeError, match="ROCM_FP8_BLOCKSCALE_CONTRACT"):
        _project(_scaled_target("nk"), "scaled_matmul_lds")
    source = _scaled_lds_target().replace("block_n = 128", "block_n = 16")
    with pytest.raises(RuntimeError, match="ROCM_FP8_BLOCKSCALE_CONTRACT"):
        _project(source, "scaled_matmul_lds")

@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_native_lds_runtime_k_identity_preserves_physical_contract(output):
    def project(source):
        return rocm_native._shape_free_target_ir(source, family="scaled_matmul_lds",
            directive="tessera_rocm.scaled_wmma_gemm", runtime_k=True)
    a = project(_scaled_lds_target(k=1536, output=output))
    assert "runtime_k" in a and "k = 0 : i64" in a
    for k in (128, 384, 2048, 3072):
        assert a == project(_scaled_lds_target(k=k, output=output))
    assert a != project(_scaled_lds_target(m=256, k=1536, output=output))
    assert a != project(_scaled_lds_target(n=8191, k=1536, output=output))
    with pytest.raises(RuntimeError, match="ROCM_FP8_BLOCKSCALE_CONTRACT"):
        project(_scaled_lds_target(k=257, output=output))
    with pytest.raises(ValueError, match="runtime"):
        rocm_native._shape_free_target_ir(_scaled_target(), family="scaled_matmul",
            directive="tessera_rocm.scaled_wmma_gemm", runtime_k=True)


def _mxfp8_target(layout="kn", output="f32"):
    from tessera.compiler.rocm_mxfp8_blockscale import MXFP8_PACKAGE_ABIS, MXFP8_CONTRACTS
    from tessera.compiler.rocm_fp8_blockscale import PACKAGE_ABIS, WEIGHT_LAYOUTS
    return (_scaled_target(layout, output)
        .replace(WEIGHT_LAYOUTS[layout][0], MXFP8_CONTRACTS[layout])
        .replace(PACKAGE_ABIS[(layout, output)], MXFP8_PACKAGE_ABIS[(layout, output)])
        .replace('scale_format = "fp32"', 'scale_format = "e8m0"')
        .replace("scale_k = 128", "scale_k = 32").replace("scale_n = 128", "scale_n = 1")
        .replace("macro_k = 128", "macro_k = 32")
        .replace("block_m = 32", "block_m = 16").replace("block_n = 32", "block_n = 16"))


@pytest.mark.parametrize("layout", ["kn", "nk"])
@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_native_mxfp8_image_keeps_scale_contract_and_erases_mnk(layout, output):
    source = _mxfp8_target(layout, output)
    projected = _project(source, "scaled_matmul")
    assert projected == _project(source.replace("m = 64", "m = 31")
        .replace("n = 96", "n = 7").replace("k = 256", "k = 128"), "scaled_matmul")
    assert "wide_scale.v1" in projected and 'scale_format = "e8m0"' in projected
    assert projected != _project(_scaled_target(layout, output), "scaled_matmul")


@pytest.mark.parametrize("old,new", [
    ('scale_format = "e8m0"', 'scale_format = "fp32"'),
    ("scale_k = 32", "scale_k = 64"),
    ("scale_n = 1", "scale_n = 2"),
    ("macro_k = 32", "macro_k = 64"),
    ("block_m = 16", "block_m = 32"),
    ("k = 256", "k = 255"),
    ("wide_scale.v1", "wmma_exact.v1"),
])
def test_native_mxfp8_identity_refuses_policy_substitution(old, new):
    with pytest.raises(RuntimeError, match="ROCM_FP8_BLOCKSCALE_CONTRACT"):
        _project(_mxfp8_target().replace(old, new), "scaled_matmul")


def _mxfp8_lds_target(m=200, n=8192, k=1024, output="f32"):
    from tessera.compiler.rocm_mxfp8_blockscale import MXFP8_PACKAGE_ABIS, MXFP8_CONTRACTS
    from tessera.compiler.rocm_fp8_blockscale import PACKAGE_ABIS, WEIGHT_LAYOUTS
    return (_scaled_lds_target(m=m, n=n, k=k, output=output)
        .replace(WEIGHT_LAYOUTS["nk"][0], MXFP8_CONTRACTS["nk"])
        .replace(PACKAGE_ABIS[("nk", output)], MXFP8_PACKAGE_ABIS[("nk", output)])
        .replace('scale_format = "fp32"', 'scale_format = "e8m0"')
        .replace("scale_k = 128", "scale_k = 32")
        .replace("scale_n = 128", "scale_n = 1")
        .replace("macro_k = 128", "macro_k = 32"))


@pytest.mark.parametrize("output", ["f32", "bf16"])
def test_mxfp8_lds_native_identity_keeps_profile_and_edge_classes(output):
    def project(source):
        return rocm_native._shape_free_target_ir(source, family="scaled_matmul_lds",
            directive="tessera_rocm.scaled_wmma_gemm", runtime_k=True)
    one = project(_mxfp8_lds_target(output=output))
    assert "runtime_mn" in one and "runtime_k" in one
    assert one == project(_mxfp8_lds_target(m=201, n=8320, k=192, output=output))
    assert one != project(_mxfp8_lds_target(m=256, output=output))
    assert one != project(_mxfp8_lds_target(n=8191, output=output))
    assert one != project(_scaled_lds_target(k=1024, output=output))


@pytest.mark.parametrize("old,new", [
    ('scale_format = "e8m0"', 'scale_format = "fp32"'),
    ("scale_k = 32", "scale_k = 16"),
    ("scale_n = 1", "scale_n = 2"),
    ("block_m = 128", "block_m = 64"),
    ("warps = 8", "warps = 4"),
    ("macro_k = 32", "macro_k = 48"),
    ("pipeline_depth = 1", "pipeline_depth = 2"),
    ('staging = "lds"', 'staging = "global"'),
    ("k = 1024", "k = 1023"),
])
def test_mxfp8_lds_native_identity_refuses_contract_drift(old, new):
    with pytest.raises(RuntimeError, match="ROCM_FP8_BLOCKSCALE_CONTRACT"):
        _project(_mxfp8_lds_target().replace(old, new), "scaled_matmul_lds")


def test_mxfp8_lds_k64_native_identity_distinguishes_supported_slab():
    def project(source):
        return rocm_native._shape_free_target_ir(source, family="scaled_matmul_lds",
            directive="tessera_rocm.scaled_wmma_gemm", runtime_k=True)

    source = _mxfp8_lds_target()
    k32 = project(source)
    k64_source = source.replace("macro_k = 32", "macro_k = 64")
    k64 = project(k64_source)
    assert k32 != k64
    assert "macro_k = 64" in k64
    assert k64 == project(k64_source.replace("k = 1024", "k = 192")
                          .replace("m = 200", "m = 201"))


def test_mxfp8_lds_k64_native_identity_refuses_partial_slab():
    source = _mxfp8_lds_target(k=1056).replace("macro_k = 32", "macro_k = 64")
    with pytest.raises(RuntimeError, match="ROCM_FP8_BLOCKSCALE_CONTRACT"):
        _project(source, "scaled_matmul_lds")
