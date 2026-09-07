"""Native split-product compilation; device numerics use the owning-host recorder."""
from pathlib import Path
import subprocess
import pytest
import numpy as np
import tessera as ts
from benchmarks.record_persistent_split_tape import source
from tessera.compiler.native_persistent_tape import materialize_persistent_tape, _shape
from tessera.compiler.scheduled_matmul import find_tessera_opt


def test_jit_reverse_trace_materializes_split_products():
    tool=find_tessera_opt()
    llvm=Path('/usr/lib/llvm-23/bin')
    if tool is None or not (llvm/'mlir-opt').exists():
        pytest.skip('requires native LLVM/MLIR toolchain')
    @ts.jit(autodiff='reverse')
    def product(x,w):
        return x*w
    pair=product.compile_persistent_device_tape(np.ones(4,np.float32),np.ones(4,np.float32),
        compiler=tool,llvm_bin=llvm,backend='rocm',chip='gfx1151')
    f,b=pair.validate()
    assert f['inputs']==['tensor<4xf32>','tensor<4xf32>']
    assert b['results']==f['inputs']


def test_nested_split_products_materialize_full_residual_storage(monkeypatch):
    tool=find_tessera_opt()
    llvm=Path('/usr/lib/llvm-23/bin')
    if tool is None or not (llvm/'mlir-opt').exists():
        pytest.skip('requires native LLVM/MLIR toolchain')
    pair=materialize_persistent_tape(source(width=8),compiler=tool,llvm_bin=llvm,backend='rocm',chip='gfx1151')
    f,b=pair.validate()
    assert f['results']==['tensor<8xf32>','tensor<1x8xf32>']
    assert b['inputs'][-1]=='tensor<1x8xf32>'
    assert 'memref.view' in pair.backward.arena_ir
    assert 'tessera.autodiff.temporary_bytes' in pair.backward.arena_ir
    assert pair.forward.binding_digest!=pair.backward.binding_digest
    import tessera.compiler.native_persistent_tape as tape
    read=tape.read_tensor_contract
    def corrupt(package):
        data=read(package)
        data['arguments'][0]['shape']=[4]
        return data
    monkeypatch.setattr(tape,'read_tensor_contract',corrupt)
    with pytest.raises(ValueError,match='tensor binding'):
        pair.validate()


def test_nested_temporary_capacity_refuses_before_packaging():
    tool=find_tessera_opt()
    llvm=Path('/usr/lib/llvm-23/bin')
    if tool is None or not (llvm/'mlir-opt').exists():
        pytest.skip('requires native LLVM/MLIR toolchain')
    with pytest.raises(subprocess.CalledProcessError) as error:
        materialize_persistent_tape(source(width=256),compiler=tool,llvm_bin=llvm,backend='rocm',chip='gfx1151')
    assert '4096 temporary bytes' in error.value.stderr


@pytest.mark.parametrize('text',['tensor<?xf32>','tensor<4xf16>','tensor<0xf32>','tensor<1024x1024xf32>','tensor<9223372036854775807xf32>'])
def test_persistent_slot_limits_refuse(text):
    with pytest.raises(ValueError):
        _shape(text)


@pytest.mark.parametrize('backend',['nvidia','rocm'])
def test_private_temporary_addressing_is_backend_owned(backend):
    tool=find_tessera_opt()
    if tool is None:
        pytest.skip('requires native compiler')
    source='''module attributes {tessera.autodiff.product_abi = "test", tessera.autodiff.product_pair = "test"} {
      func.func @product(%out: memref<4xf32>) {
        %a = memref.alloc() : memref<4xf32>
        memref.copy %a, %out : memref<4xf32> to memref<4xf32>
        return
      }
    }'''
    result=subprocess.check_output([tool,'--allow-unregistered-dialect','--tessera-native-tape-to-gpu=backend='+backend],input=source,text=True)
    assert ('memref<16xi8, 5>' in result)==(backend=='rocm')
    assert ('memref.memory_space_cast' in result)==(backend=='rocm')


def test_zero_trip_reserved_temporaries_still_count_against_capacity():
    tool=find_tessera_opt()
    if tool is None:
        pytest.skip('requires native compiler')
    source='''module attributes {tessera.autodiff.product_abi = "test", tessera.autodiff.product_pair = "test"} {
      func.func @product() {
        %z = arith.constant 0 : index
        %one = arith.constant 1 : index
        scf.for %i = %z to %z step %one {
          %a = memref.alloc() : memref<1024xf32>
          %b = memref.alloc() : memref<1024xf32>
        }
        return
      }
    }'''
    result=subprocess.run([tool,'--tessera-native-tape-to-gpu'],input=source,text=True,capture_output=True)
    assert result.returncode!=0
    assert '4096 temporary bytes' in result.stderr


def test_dynamic_copy_extent_refuses_before_serial_loop_expansion():
    tool=find_tessera_opt()
    if tool is None:
        pytest.skip('requires native compiler')
    source='''module attributes {tessera.autodiff.product_abi = "test", tessera.autodiff.product_pair = "test"} {
      func.func @product(%out: memref<4xf32>) {
        %n = arith.constant 4 : index
        %view = memref.subview %out[0] [%n] [1] : memref<4xf32> to memref<?xf32, strided<[1]>>
        memref.copy %view, %view : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>>
        return
      }
    }'''
    result=subprocess.run([tool,'--tessera-native-tape-to-gpu'],input=source,text=True,capture_output=True)
    assert result.returncode!=0
    assert 'isolated static f32' in result.stderr


from benchmarks.record_tape_checkpoint_execution import mixed_source


def test_mixed_float_products_preserve_storage_types():
    from tessera.compiler.native_storage_contract import read_tensor_contract
    tool = find_tessera_opt()
    llvm = Path('/usr/lib/llvm-23/bin')
    if tool is None or not (llvm/'mlir-opt').exists():
        pytest.skip('requires native LLVM/MLIR toolchain')
    pair = materialize_persistent_tape(mixed_source(), compiler=tool, llvm_bin=llvm,
                                      backend='rocm', chip='gfx1151')
    forward, backward = pair.validate()
    assert forward['inputs'] == ['tensor<4xf32>', 'tensor<4xf64>']
    assert backward['results'] == forward['inputs']
    assert [a['dtype'] for a in read_tensor_contract(pair.forward)['arguments'][:-1]] == [
        'fp32', 'fp64', 'fp32', 'fp64']


def test_f64_temporary_budget_counts_eight_byte_elements():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('requires native compiler')
    text = '''module attributes {tessera.autodiff.product_abi = "test", tessera.autodiff.product_pair = "test"} {
      func.func @product() {
        %a = memref.alloc() : memref<513xf64>
        return
      }
    }'''
    result = subprocess.run([tool, '--tessera-native-tape-to-gpu'], input=text, text=True, capture_output=True)
    assert result.returncode != 0
    assert '4096 temporary bytes' in result.stderr


@pytest.mark.parametrize('loaded', [False, True])
@pytest.mark.parametrize('selected_start', [False, True])
def test_replay_bounds_require_derived_ssa_capacity(loaded, selected_start):
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('requires native compiler')
    bound = '%end = memref.dim %out, %z : memref<4xf32>' if loaded else '%end = arith.subi %four, %i : index'
    text = '''module attributes {tessera.autodiff.product_abi = "test", tessera.autodiff.product_pair = "test"} {
      func.func @product(%out: memref<4xf32>) {
        %z = arith.constant 0 : index
        %one = arith.constant 1 : index
        %four = arith.constant 4 : index
        scf.for %i = %z to %four step %one {
          BOUND
          scf.for %j = %z to %end step %one {
            %a = memref.alloc() : memref<4xf32>
            memref.copy %a, %out : memref<4xf32> to memref<4xf32>
          }
        }
        return
      }
    }'''.replace('BOUND', bound)
    if selected_start:
        text = text.replace('scf.for %j = %z', '%c = arith.cmpi eq, %i, %z : index\n          %start = arith.select %c, %one, %z : index\n          scf.for %j = %start')
    result = subprocess.run([tool, '--tessera-native-tape-to-gpu'], input=text, text=True, capture_output=True)
    if loaded:
        assert result.returncode != 0
    else:
        assert result.returncode == 0, result.stderr
        assert 'tessera.autodiff.temporary_bytes = 256' in result.stdout
        assert 'memref<256xi8>' in result.stdout


def test_counted_while_normalizes_to_persistent_tensor_tape():
    from benchmarks.record_tape_checkpoint_execution import counted_while_source
    tool = find_tessera_opt()
    llvm = Path('/usr/lib/llvm-23/bin')
    if tool is None or not (llvm/'mlir-opt').exists():
        pytest.skip('requires native LLVM/MLIR toolchain')
    pair = materialize_persistent_tape(counted_while_source(), compiler=tool, llvm_bin=llvm,
                                      backend='rocm', chip='gfx1151')
    f, _ = pair.validate()
    assert f['results'] == ['tensor<4xf32>', 'tensor<2x4xf32>']
    assert 'scf.while' not in pair.forward.arena_ir


@pytest.mark.parametrize('change', ['predicate', 'step', 'capacity'])
def test_unproven_while_does_not_enter_device_tape(change):
    from benchmarks.record_tape_checkpoint_execution import counted_while_source
    text = counted_while_source()
    text = {'predicate': text.replace('cmpi slt', 'cmpi sle'),
            'step': text.replace('addi %i, %one', 'addi %i, %zero'),
            'capacity': text.replace('max_iters = 3', 'max_iters = 2')}[change]
    tool = find_tessera_opt()
    llvm = Path('/usr/lib/llvm-23/bin')
    if tool is None or not (llvm/'mlir-opt').exists():
        pytest.skip('requires native LLVM/MLIR toolchain')
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        materialize_persistent_tape(text, compiler=tool, llvm_bin=llvm, backend='rocm', chip='gfx1151')


def test_unsigned_replay_loop_cannot_use_signed_capacity_proof():
    tool = find_tessera_opt()
    if tool is None:
        pytest.skip('requires native compiler')
    text = '''module attributes {tessera.autodiff.product_abi = "test", tessera.autodiff.product_pair = "test"} {
      func.func @product() {
        %z = arith.constant 0 : index
        %one = arith.constant 1 : index
        %negative = arith.constant -1 : index
        scf.for unsigned %i = %z to %negative step %one {
          %a = memref.alloc() : memref<4xf32>
        }
        return
      }
    }'''
    result = subprocess.run([tool, '--tessera-native-tape-to-gpu'], input=text, text=True, capture_output=True)
    assert result.returncode != 0
    assert '4096 temporary bytes' in result.stderr
