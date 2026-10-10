"""Runtime row grids follow actual bounded sequence sizes, never capacities."""
from dataclasses import replace
import inspect
from types import SimpleNamespace

import pytest

from tessera.compiler.native_gpu_storage import NativeGPUStoragePackage
from tessera.compiler.native_gpu_tensor import GridProduct, IndexSpec, TensorSpec
from tessera.compiler.native_storage_contract import (
    attach_tensor_contract, generate_tensor_binding, read_tensor_contract)


def binding(grid=None, maximum=11):
    specs = (TensorSpec('output', 'fp32', (2, 4, 'query_size', 3), True),
             IndexSpec('query_size', 1, maximum))
    source = attach_tensor_contract('module {\n}', specs,
        grid=(GridProduct((2, 4, 'query_size')), 1, 1) if grid is None else grid,
        block=(128, 1, 1))
    package = NativeGPUStoragePackage('nvidia', 'sm_120', 'entry', 'size',
        ('pointer', 'index'), source, b'image', b'host', 'c' * 64, 'd' * 64, '')
    package = replace(package, binding_digest=package._digest())
    return generate_tensor_binding(package, inspect.signature(lambda output, query_size: None))


@pytest.mark.parametrize('query_size', [1, 3, 7, 11])
def test_checked_grid_uses_actual_sequence_extent(query_size):
    call = binding()
    output = SimpleNamespace(__cuda_array_interface__={
        'version': 3, 'shape': (2, 4, query_size, 3), 'typestr': '<f4',
        'data': (4096, False), 'strides': None})
    raw, _, grid, block, _ = call.prepare(output, query_size)
    assert raw == (4096, query_size)
    assert grid == (8 * query_size, 1, 1)
    assert block == (128, 1, 1)
    assert read_tensor_contract(call.package)['schema'] == 2
    assert generate_tensor_binding(call.package, call.signature).binding_digest == call.binding_digest


@pytest.mark.parametrize('factors', [(), (2,), (True, 'query_size'),
    (0, 'query_size'), (-1, 'query_size'), (2, 'missing'), (2, 3.0),
    (1 << 63, 'query_size')])
def test_invalid_product_rejected_before_loading(factors):
    with pytest.raises(ValueError, match='grid product'):
        binding((GridProduct(factors), 1, 1))


def test_product_capacity_overflow_rejected_before_loading():
    with pytest.raises(ValueError, match='launch envelope'):
        binding(maximum=1 << 30)


def test_product_cannot_be_smuggled_into_schema_one():
    call = binding()
    edited = call.package.arena_ir.replace('\\22schema\\22:2', '\\22schema\\22:1')
    assert edited != call.package.arena_ir
    package = replace(call.package, arena_ir=edited)
    package = replace(package, binding_digest=package._digest())
    with pytest.raises(ValueError, match='grid expression'):
        generate_tensor_binding(package, call.signature)


@pytest.mark.parametrize('query_size', [0, 12, True])
def test_runtime_extent_rejection_precedes_any_residency_access(query_size):
    with pytest.raises(ValueError, match='index bounds'):
        binding()(object(), query_size)


def test_grid_products_not_admitted_as_block_geometry():
    call = binding()
    from tessera.compiler.native_gpu_tensor import NativeTensorCall
    with pytest.raises(ValueError, match='block geometry'):
        NativeTensorCall(call.package, call.signature, call.specs,
                        grid=call.grid, block=(GridProduct((2, 4)), 1, 1))
