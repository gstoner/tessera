"""Capacity-proven dynamic GPU buffers and typed multiway CFG execution."""

import json
import platform
import numpy as np
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt, run_tessera_opt


def compiler():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip("native compiler required")
    return tool


@pytest.mark.parametrize("backend", ["nvidia", "rocm"])
def test_dynamic_logical_extents_use_proved_private_capacity(backend):
    text = run_tessera_opt(compiler(), dynamic_source(), "--tessera-native-tape-to-gpu=backend=" + backend)
    assert "memref<36xi8" in text
    assert "memref.view" in text and "memref<?xf32" in text
    assert "memref.dim" in text


@pytest.mark.parametrize("change", ["negative", "large", "mismatched_copy", "loaded"])
def test_unproved_dynamic_storage_refuses(change):
    source = dynamic_source()
    if change == "negative":
        source = source.replace("arith.addi %i, %one", "arith.subi %i, %one")
    if change == "large":
        source = source.replace("%one = arith.constant 1", "%one = arith.constant 1024")
    if change == "mismatched_copy":
        source = source.replace("%b = memref.alloc(%n)", "%b = memref.alloc(%three)")
    if change == "loaded":
        source = source.replace(
            "%n = arith.addi %i, %one : index",
            "%f = memref.load %out[%z,%z] : memref<3x3xf32>\n%k = arith.fptosi %f : f32 to i64\n%n = arith.index_cast %k : i64 to index",
        )
    with pytest.raises(RuntimeError):
        run_tessera_opt(compiler(), source, "--tessera-native-tape-to-gpu")


@pytest.mark.parametrize("cross_block", [False, True])
def test_switch_edges_recover_and_execute_reverse(cross_block):
    from tessera import _jit_boundary as jit
    from tessera.compiler.native_persistent_tape import _attribute, _shape, _dtype

    tool = compiler()
    if platform.machine().lower() not in ("x86_64", "amd64") or jit._find_dylib() is None:
        pytest.skip("native x86 JIT required")
    source = switch_source()
    if cross_block:
        source = source.replace('cf.switch %flag', '%base = "tessera.mul"(%x,%x) : (tensor<4xf32>,tensor<4xf32>) -> tensor<4xf32>\ncf.switch %flag')
        source = source.replace('"tessera.mul"(%a,%a)', '"tessera.mul"(%a,%base)').replace('"tessera.add"(%b,%b)', '"tessera.add"(%b,%base)').replace('"tessera.mul"(%cc,%c)', '"tessera.mul"(%cc,%base)')
    handles = []
    contracts = []
    try:
        for role in ("forward", "backward"):
            text = run_tessera_opt(
                tool, source, "--tessera-autodiff-paired=box-product-scalars=true export-product=" + role
            )
            assert "cf.switch" not in text.split("func.func", 1)[1]
            contracts.append(json.loads(_attribute(text, "tessera.autodiff.product_abi")))
            handles.append(jit.compile_module(text))
        for value in (0.5, 1.5, 2.5):
            x = np.full(4, value, np.float32)
            primal = np.empty_like(x)
            saved = [
                np.zeros(_shape(t), dtype={"fp32": np.float32, "int64": np.int64, "int8": np.int8}[_dtype(t)])
                for t in contracts[0]["results"][1:]
            ]
            jit.invoke(handles[0], contracts[0]["entry"], [x], [primal, *saved])
            expected = x * x if value < 1 else (2 * x if value < 2 else x * x * x)
            if cross_block:
                expected = x**3 if value<1 else (x+x*x if value<2 else x**4)
            np.testing.assert_allclose(primal, expected)
            dx = np.empty_like(x)
            jit.invoke(handles[1], contracts[1]["entry"], [x, np.ones_like(x), *saved], dx)
            expected_dx = 2*x if value<1 else (np.full_like(x,2) if value<2 else 3*x*x)
            if cross_block:
                expected_dx = 3*x*x if value<1 else (1+2*x if value<2 else 4*x**3)
            np.testing.assert_allclose(dx, expected_dx)
    finally:
        for handle in handles:
            jit.destroy(handle)


from benchmarks.record_dynamic_cfg_storage import dynamic_source, switch_source


@pytest.mark.parametrize("replacement", ["0", "-1", "1000001"])
def test_switch_requires_valid_replay_bound(replacement):
    source = switch_source().replace("max_steps = 2 : i64", "max_steps = " + replacement + " : i64")
    with pytest.raises(RuntimeError, match="positive bounded"):
        run_tessera_opt(compiler(), source, "--tessera-autodiff-paired=export-product=forward")


def test_zero_logical_extent_keeps_positive_capacity():
    source=dynamic_source().replace("%n = arith.addi %i, %one", "%n = arith.addi %i, %z")
    text=run_tessera_opt(compiler(),source,'--tessera-native-tape-to-gpu')
    assert 'memref<24xi8' in text
    assert 'memref.dim' in text
