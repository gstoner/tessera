"""Native producer and generation ownership regressions; no mock compiler proof."""
from pathlib import Path
import re
import subprocess
import pytest
from tessera.compiler.scheduled_matmul import find_tessera_opt

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def compiler():
    opt = find_tessera_opt()
    if opt is None:
        pytest.skip('native tessera-opt unavailable')
    return str(opt)


def run(compiler, source, *flags):
    return subprocess.run([compiler, '--allow-unregistered-dialect', *flags],
                          input=source, capture_output=True, text=True, timeout=30)


def rotating():
    return (ROOT / 'tests/tessera-ir/phase3/tile_dynamic_gpu_rotating.mlir').read_text()


def groups(text):
    return re.findall(r'tile.buffer_group = (\d+)', text)


def test_rotating_slot_reuses_only_after_final_drain(compiler):
    result = run(compiler, rotating(), '--tessera-tile-buffer-reuse')
    assert result.returncode == 0, result.stderr
    assert groups(result.stdout) == ['0', '0']


@pytest.mark.parametrize('before,after', [
    ('nvgpu.device_async_wait %token', 'nvgpu.device_async_wait %initial_group'),
    ('nvgpu.device_async_wait %last#0', 'nvgpu.device_async_wait %initial_group'),
    ('nvgpu.device_async_wait %token', 'nvgpu.device_async_wait %token { numGroups = 1 : i32 }'),
    ('        gpu.barrier\n        %refill', '        %refill'),
    ('      nvgpu.device_async_wait %last#0\n      gpu.barrier', '      nvgpu.device_async_wait %last#0'),
    ('scf.yield %new_group, %total', 'scf.yield %token, %total'),
])
def test_stale_partial_or_unreleased_generation_cannot_coalesce(compiler, before, after):
    source = rotating()
    # Release is immediately before the next-generation address arithmetic.
    if before == '        gpu.barrier\n        %refill':
        before, after = '        gpu.barrier\n        %next_generation', '        %next_generation'
    assert before in source
    source = source.replace(before, after)
    result = run(compiler, source, '--tessera-tile-buffer-reuse')
    assert result.returncode == 0, result.stderr
    assert groups(result.stdout) == ['0', '1']
    forged = re.sub(r'tile.buffer_group = 1', 'tile.buffer_group = 0', result.stdout)
    rejected = run(compiler, forged, '--tessera-tile-buffer-arena')
    assert rejected.returncode != 0
    assert 'TILE_BARRIER_REUSE_MISSING_BARRIER' in rejected.stderr


def test_apple_materializer_keeps_arena_offsets_and_native_sizer(compiler):
    from tessera.compiler.apple_native_arena import materialize_apple_arena
    source = (ROOT / 'tests/tessera-ir/phase3/tile_dynamic_gpu_nested_device.mlir').read_text()
    artifact = materialize_apple_arena(source, compiler=Path(compiler), llvm_bin=Path('/usr/lib/llvm-23/bin'))
    assert 'threadgroup uchar* arena [[threadgroup(0)]]' in artifact.msl
    assert 'reinterpret_cast<threadgroup float*>(arena +' in artifact.msl
    assert 'threadgroup_barrier(mem_flags::mem_threadgroup)' in artifact.msl
    assert artifact.sizer in artifact.host_llvm_ir
    assert 'for (long' in artifact.msl


def test_apple_materializer_rejects_nvidia_async_protocol(compiler):
    result = run(compiler, rotating(), '--tessera-tile-buffer-reuse', '--tessera-tile-buffer-arena=emit-apple-msl=true')
    assert result.returncode != 0
    assert 'unsupported operation in Apple arena MSL' in result.stderr


def test_native_ad_child_is_generated_from_paired_ssa_and_supports_stop_gradient(compiler):
    source = (ROOT / 'tests/tessera-ir/phase_f4/autodiff_native_storage.mlir').read_text()
    result = run(compiler, source, '--tessera-autodiff-forward=emit-storage-child=true')
    assert result.returncode == 0, result.stderr
    assert 'tessera.native_jvp_pair' in result.stdout
    assert result.stdout.count(' = arith.mulf') == 3
    # A stopped primal remains present while its compiler-generated tangent is zero.
    source = source.replace('tessera.mul %x, %x : (tensor<32xf32>, tensor<32xf32>)',
                            'tessera.stop_gradient %x : (tensor<32xf32>)')
    result = run(compiler, source, '--tessera-autodiff-forward=emit-storage-child=true')
    assert result.returncode == 0, result.stderr
    assert 'arith.constant 0.000000e+00 : f32' in result.stdout
    assert ' = arith.mulf' not in result.stdout


def test_storage_child_cannot_adopt_an_already_authored_jvp(compiler):
    source = (ROOT / 'tests/tessera-ir/phase_f4/autodiff_native_storage.mlir').read_text()
    paired = run(compiler, source, '--tessera-autodiff-forward')
    assert paired.returncode == 0, paired.stderr
    forged = paired.stdout.replace('tessera.autodiff = "forward", ', '')
    result = run(compiler, forged, '--tessera-autodiff-forward=emit-storage-child=true')
    assert result.returncode != 0
    assert 'requires exactly one forward request' in result.stderr


def test_native_ad_subtraction_keeps_operand_order(compiler):
    source = (ROOT / 'tests/tessera-ir/phase_f4/autodiff_native_storage.mlir').read_text()
    source = source.replace('tessera.mul', 'tessera.sub')
    result = run(compiler, source, '--tessera-autodiff-forward=emit-storage-child=true')
    assert result.returncode == 0, result.stderr
    assert result.stdout.count(' = arith.subf') == 2


def test_native_ad_activation_uses_stable_native_math(compiler):
    source = (ROOT / 'tests/tessera-ir/phase_f4/autodiff_native_storage.mlir').read_text()
    source = source.replace('tessera.mul %x, %x : (tensor<32xf32>, tensor<32xf32>)',
                            'tessera.tanh %x : (tensor<32xf32>)')
    result = run(compiler, source, '--tessera-autodiff-forward=emit-storage-child=true')
    assert result.returncode == 0, result.stderr
    assert 'math.exp2' in result.stdout
    assert 'math.copysign' in result.stdout


def slot_alias():
    return (ROOT / 'tests/tessera-ir/phase3/tile_dynamic_gpu_slot_alias.mlir').read_text()


def test_dynamic_slot_permutation_reuses_after_collective_release(compiler):
    result = run(compiler, slot_alias(), '--tessera-tile-buffer-reuse')
    assert result.returncode == 0, result.stderr
    assert groups(result.stdout) == ['0', '1', '0']
    arena = run(compiler, result.stdout, '--tessera-tile-buffer-arena')
    assert arena.returncode == 0, arena.stderr


@pytest.mark.parametrize('before,after', [
    ('scf.yield %write_slot, %read_slot', 'scf.yield %write_slot, %write_slot'),
    ('nvgpu.device_async_wait %group', 'nvgpu.device_async_wait %group { numGroups = 1 : i32 }'),
    ('        gpu.barrier\n        scf.yield', '        scf.yield'),
    ('        nvgpu.device_async_wait %group\n        gpu.barrier', '        nvgpu.device_async_wait %group'),
])
def test_dynamic_slot_alias_rejects_unreleased_or_nonbijective_carries(compiler, before, after):
    source = slot_alias().replace(before, after)
    result = run(compiler, source, '--tessera-tile-buffer-reuse')
    assert result.returncode == 0, result.stderr
    assert groups(result.stdout) == ['0', '1', '2']
    forged = result.stdout.replace('tile.buffer_group = 2', 'tile.buffer_group = 0')
    arena = run(compiler, forged, '--tessera-tile-buffer-arena')
    assert arena.returncode != 0


def test_forward_ad_preserves_requested_tangent_order(compiler):
    source = '''module {
      func.func @difference(%x: tensor<32xf32>, %y: tensor<32xf32>) -> tensor<32xf32>
        attributes {tessera.autodiff = "forward", tessera.autodiff.wrt_indices = [1 : i64, 0 : i64]} {
        %z = tessera.sub %x, %y : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
        return %z : tensor<32xf32>
      }
    }'''
    result = run(compiler, source, '--tessera-autodiff-forward=emit-storage-child=true')
    assert result.returncode == 0, result.stderr
    # Parameters are x, y, dy, dx: the tangent of x-y is dx-dy.
    operations = re.findall(r'arith.subf (%\w+), (%\w+)', result.stdout)
    loads = re.findall(r'(%\w+) = llvm.load ', result.stdout)
    assert operations == [(loads[0], loads[1]), (loads[3], loads[2])]


def test_synchronous_slot_alias_descriptors_follow_arena_space(compiler):
    source = slot_alias()
    start = source.index('        %copy = nvgpu.device_async_copy')
    end = source.index('        gpu.barrier', start)
    source = source[:start] + '        memref.store %initial, %write_slot[%tid] : memref<?xf32, 3>\n' + source[end:]
    source = source.replace('memref<?xf32, 3>', 'memref<?xf32>')
    result = run(compiler, source, '--tessera-tile-buffer-reuse', '--tessera-tile-buffer-arena')
    assert result.returncode == 0, result.stderr
    header = next(line for line in result.stdout.splitlines() if 'scf.for' in line)
    assert header.count('memref<?xf32, #gpu.address_space<workgroup>>') == 2


def pending_swap():
    return (ROOT / 'tests/tessera-ir/phase3/tile_dynamic_gpu_pending_swap.mlir').read_text()


def test_pending_token_and_slot_permute_together(compiler):
    result = run(compiler, pending_swap(), '--tessera-tile-buffer-reuse')
    assert result.returncode == 0, result.stderr
    assert groups(result.stdout) == ['0', '1', '0']
    arena = run(compiler, result.stdout, '--tessera-tile-buffer-arena')
    assert arena.returncode == 0, arena.stderr


@pytest.mark.parametrize('before,after', [
    ('scf.yield %new_group, %total, %write_slot, %read_slot', 'scf.yield %new_group, %total, %read_slot, %write_slot'),
    ('scf.yield %new_group, %total, %write_slot, %read_slot', 'scf.yield %token, %total, %write_slot, %read_slot'),
    ('%source[%refill_position], %write_slot[%tid]', '%source[%refill_position], %read_slot[%tid]'),
    ('nvgpu.device_async_wait %last#0', 'nvgpu.device_async_wait %initial_group'),
    ('nvgpu.device_async_wait %token', 'nvgpu.device_async_wait %token {numGroups = 1 : i32}'),
    ('        gpu.barrier\n        %total', '        %total'),
])
def test_pending_swap_rejects_mismatched_generation_or_release(compiler, before, after):
    result = run(compiler, pending_swap().replace(before, after), '--tessera-tile-buffer-reuse')
    assert result.returncode == 0, result.stderr
    assert groups(result.stdout) == ['0', '1', '2']
    forged = result.stdout.replace('tile.buffer_group = 2', 'tile.buffer_group = 0')
    assert run(compiler, forged, '--tessera-tile-buffer-arena').returncode != 0


@pytest.mark.parametrize('count', [3, 4, 8])
def test_general_released_slot_permutation(compiler, count):
    source = slot_alias()
    additions = ''.join(f'      %extra{i} = memref.alloca(%n) : memref<?xf32, 3>\n      "tile.alloc_shared"(%extra{i}) : (memref<?xf32, 3>) -> ()\n' for i in range(count-2))
    source = source.replace('      %slots:3', additions + '      %slots:' + str(count+1))
    init = ', '.join(f'%slot{i} = %extra{i}' for i in range(count-2))
    source = source.replace('%acc = %initial)', '%acc = %initial, ' + init + ')')
    source = source.replace('memref<?xf32, 3>, f32) {', 'memref<?xf32, 3>, f32' + ', memref<?xf32, 3>'*(count-2) + ') {')
    source = source.replace('scf.yield %write_slot, %read_slot, %total : memref<?xf32, 3>, memref<?xf32, 3>, f32',
        'scf.yield %write_slot, %slot0, %total, ' + ', '.join([f'%slot{i}' for i in range(1,count-2)] + ['%read_slot']) +
        ' : memref<?xf32, 3>, memref<?xf32, 3>, f32' + ', memref<?xf32, 3>'*(count-2))
    result = run(compiler, source, '--tessera-tile-buffer-reuse')
    assert result.returncode == 0, result.stderr
    assert groups(result.stdout) == [str(i) for i in range(count)] + ['0']
    assert run(compiler, result.stdout, '--tessera-tile-buffer-arena').returncode == 0


def test_nested_pending_swap_is_released_before_outer_backedge(compiler):
    source = pending_swap()
    source = source.replace('      %a = memref.alloca', '      scf.for %outer = %zero to %rounds step %one {\n      %a = memref.alloca')
    source = source.replace('      gpu.return', '      }\n      gpu.return')
    result = run(compiler, source, '--tessera-tile-buffer-reuse')
    assert result.returncode == 0, result.stderr
    assert groups(result.stdout) == ['0', '1', '0']
    assert run(compiler, result.stdout, '--tessera-tile-buffer-arena').returncode == 0
    divergent = source.replace('%outer = %zero to %rounds', '%outer = %zero to %tid')
    result = run(compiler, divergent, '--tessera-tile-buffer-reuse')
    assert result.returncode == 0, result.stderr
    assert groups(result.stdout) == ['0', '1', '2']


@pytest.mark.parametrize('kind', ['sum', 'mean'])
def test_native_pair_reduction_is_cooperative(compiler, kind):
    source = '''module attributes {tessera.frontend.authority = "tracer"} {
      func.func @sum(%x: tensor<32xf32>) -> tensor<f32> attributes {tessera.autodiff = "forward"} {
        %r = tessera.reduce %x {kind = "KIND", axis = 0 : i64} : (tensor<32xf32>) -> tensor<f32>
        return %r : tensor<f32>
      }
    }'''.replace('KIND', kind)
    result = run(compiler, source, '--tessera-autodiff-forward=emit-storage-child=true')
    assert result.returncode == 0, result.stderr
    assert 'tessera.native_jvp_output_width = 1' in result.stdout
    assert 'scf.if' in result.stdout and 'gpu.barrier' in result.stdout
    assert result.stdout.count(' = arith.addf') == 10


def test_native_reverse_consumes_explicit_cotangent(compiler):
    source = (ROOT / 'tests/tessera-ir/phase_f4/autodiff_native_storage.mlir').read_text().replace('"forward"', '"reverse"')
    result = run(compiler, source, '--tessera-autodiff-paired=emit-storage-child=true')
    assert result.returncode == 0, result.stderr
    assert 'tessera.native_vjp_pair' in result.stdout
    assert 'tessera.native_vjp_inputs = 2' in result.stdout
    # A previously paired module cannot stand in for a fresh reverse request.
    paired = run(compiler, source, '--tessera-autodiff-paired')
    replay = run(compiler, paired.stdout, '--tessera-autodiff-paired=emit-storage-child=true')
    assert replay.returncode != 0


def pending_ring(slots):
    source = (ROOT / 'tests/tessera-ir/phase3/tile_dynamic_gpu_pending_swap.mlir').read_text()
    if slots == 2:
        return source
    extra = slots - 2
    allocations = ''.join(f'      %extra{i} = memref.alloca(%n) : memref<?xf32, 3>\n      "tile.alloc_shared"(%extra{i}) : (memref<?xf32, 3>) -> ()\n' for i in range(extra))
    source = source.replace('      %seed =', allocations + '      %seed =')
    source = source.replace('%last:4 =', f'%last:{slots+2} =')
    source = source.replace('%write_slot = %other)', '%write_slot = %other, ' + ', '.join(f'%slot{i} = %extra{i}' for i in range(extra)) + ')')
    source = source.replace('f32, memref<?xf32, 3>, memref<?xf32, 3>) {', 'f32, memref<?xf32, 3>, memref<?xf32, 3>' + ', memref<?xf32, 3>'*extra + ') {')
    source = source.replace('scf.yield %new_group, %total, %write_slot, %read_slot : !nvgpu.device.async.token, f32, memref<?xf32, 3>, memref<?xf32, 3>',
        'scf.yield %new_group, %total, %write_slot, ' + ', '.join([f'%slot{i}' for i in range(extra)] + ['%read_slot']) +
        ' : !nvgpu.device.async.token, f32' + ', memref<?xf32, 3>'*slots)
    return source


@pytest.mark.parametrize('slots', [3, 4, 8])
def test_pending_ring_tracks_destination_generation(compiler, slots):
    source = pending_ring(slots)
    good = run(compiler, source, '--tessera-tile-buffer-reuse')
    assert good.returncode == 0, good.stderr
    assert groups(good.stdout) == [str(i) for i in range(slots)] + ['0']
    arena = run(compiler, good.stdout, '--tessera-tile-buffer-arena')
    assert arena.returncode == 0, arena.stderr
    bad = source.replace('scf.yield %new_group, %total, %write_slot, %slot0',
                         'scf.yield %new_group, %total, %slot0, %write_slot')
    result = run(compiler, bad, '--tessera-tile-buffer-reuse')
    assert result.returncode == 0, result.stderr
    assert groups(result.stdout) == [str(i) for i in range(slots+1)]


@pytest.mark.parametrize('kind', ['sum', 'mean'])
def test_reduction_vjp_has_independent_cotangent_extent(compiler, kind):
    source = '''module attributes {tessera.frontend.authority = "tracer"} {
      func.func @reduce(%x: tensor<32xf32>) -> tensor<f32> attributes {tessera.autodiff = "reverse"} {
        %y = tessera.reduce %x {kind = "KIND", axis = 0 : i64} : (tensor<32xf32>) -> tensor<f32>
        return %y : tensor<f32>
      }
    }'''.replace('KIND', kind)
    result = run(compiler, source, '--tessera-autodiff-paired=emit-storage-child=true')
    assert result.returncode == 0, result.stderr
    assert 'native_vjp_input_widths = [32, 1]' in result.stdout
    assert 'native_vjp_output_widths = [1, 32]' in result.stdout
    assert 'llvm.getelementptr %arg1[0]' in result.stdout


def test_saved_reverse_residual_is_forwarded_into_native_child(compiler):
    source = '''module attributes {tessera.frontend.authority = "tracer"} {
      func.func @saved(%x: tensor<32xf32>) -> tensor<32xf32> attributes {tessera.autodiff = "reverse"} {
        %y = tessera.tanh %x {tessera.autodiff.checkpoint_policy = "save", tessera.autodiff.residual_materialized = true, tessera.autodiff.residual_result_indices = array<i64: 0>} : (tensor<32xf32>) -> tensor<32xf32>
        return %y : tensor<32xf32>
      }
    }'''
    result = run(compiler, source, '--tessera-autodiff-paired=emit-storage-child=true')
    assert result.returncode == 0, result.stderr
    assert 'residual_policy' in result.stdout
    assert 'native_vjp_inputs = 2' in result.stdout


def test_native_reverse_does_not_mistake_second_primal_for_residual(compiler):
    source = '''module attributes {tessera.frontend.authority = "tracer"} {
      func.func @two(%x: tensor<32xf32>) -> (tensor<32xf32>, tensor<32xf32>) attributes {tessera.autodiff = "reverse"} {
        %y = tessera.mul %x, %x : (tensor<32xf32>, tensor<32xf32>) -> tensor<32xf32>
        return %x, %y : tensor<32xf32>, tensor<32xf32>
      }
    }'''
    result = run(compiler, source, '--tessera-autodiff-paired=emit-storage-child=true')
    assert result.returncode != 0
    assert 'one primal result' in result.stderr
