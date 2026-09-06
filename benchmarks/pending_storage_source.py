"""Matched outstanding-copy cohorts for native ownership and device probes."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def pending_cohort(count):
    """Independent read/refill pairs, each carrying its own completion token."""
    source = (ROOT / 'tests/tessera-ir/phase3/tile_dynamic_gpu_pending_swap.mlir').read_text()
    allocation = source[source.index('      %a = memref.alloca'):source.index('      %last:4')]
    refill = source[source.index('        %refill ='):source.index('        gpu.barrier\n        %total')]
    for i in range(1, count):
        rename = lambda text: re.sub(r'%(a|other|seed|initial_group|refill|new_group|read|read_slot|write_slot)(?![\w])',
                                    lambda match: match.group(0) + str(i), text)
        source = source.replace('      %last:', rename(allocation) + '      %last:', 1)
        source = source.replace('        %next_generation', f'        nvgpu.device_async_wait %token{i}\n        %next_generation', 1)
        source = source.replace('        gpu.barrier\n        %total', rename(refill) + '        gpu.barrier\n        %total', 1)
    # Every token must complete before the shared publication barrier.
    for i in range(1, count):
        source = source.replace(f'        nvgpu.device_async_wait %token{i}\n', '')
    source = source.replace('        nvgpu.device_async_wait %token\n',
                            '        nvgpu.device_async_wait %token\n' + ''.join(f'        nvgpu.device_async_wait %token{i}\n' for i in range(1, count)))
    extra_types = ', !nvgpu.device.async.token, memref<?xf32, 3>, memref<?xf32, 3>' * (count - 1)
    source = source.replace('%last:4', f'%last:{4 + 3 * (count - 1)}')
    source = source.replace('%write_slot = %other)', '%write_slot = %other' + ''.join(
        f', %token{i} = %initial_group{i}, %read_slot{i} = %a{i}, %write_slot{i} = %other{i}' for i in range(1, count)) + ')')
    source = source.replace('memref<?xf32, 3>, memref<?xf32, 3>) {', 'memref<?xf32, 3>, memref<?xf32, 3>' + extra_types + ') {')
    source = source.replace('scf.yield %new_group, %total, %write_slot, %read_slot :',
        'scf.yield %new_group, %total, %write_slot, %read_slot' + ''.join(
            f', %new_group{i}, %write_slot{i}, %read_slot{i}' for i in range(1, count)) + ' :')
    source = source.replace('f32, memref<?xf32, 3>, memref<?xf32, 3>\n', 'f32, memref<?xf32, 3>, memref<?xf32, 3>' + extra_types + '\n')
    source = source.replace('      nvgpu.device_async_wait %last#0\n', '      nvgpu.device_async_wait %last#0\n' + ''.join(
        f'      nvgpu.device_async_wait %last#{4 + 3 * (i - 1)}\n' for i in range(1, count)))
    # Distinct input slab for every outstanding generation stream.
    source = source.replace('      %len = arith.index_cast %all_length',
        f'      %cohorts = arith.constant {count} : index\n      %cohort_length = arith.muli %all_length, %cohorts : index\n      %len = arith.index_cast %cohort_length')
    for i in range(1, count):
        source = source.replace(f'      %seed{i} =',
            f'      %cohort{i} = arith.constant {i} : index\n      %offset{i} = arith.muli %all_length, %cohort{i} : index\n      %position{i} = arith.addi %position, %offset{i} : index\n      %seed{i} =')
        source = source.replace(f'%seed{i} = nvgpu.device_async_copy %source[%position]', f'%seed{i} = nvgpu.device_async_copy %source[%position{i}]')
        source = source.replace(f'        %refill{i} =', f'        %refill_position{i} = arith.addi %refill_position, %offset{i} : index\n        %refill{i} =')
        source = source.replace(f'%refill{i} = nvgpu.device_async_copy %source[%refill_position]', f'%refill{i} = nvgpu.device_async_copy %source[%refill_position{i}]')
    sums = ''.join(f'        %sum{i} = arith.addf ' + ('%read' if i == 1 else f'%sum{i-1}') + f', %read{i} : f32\n' for i in range(1, count))
    source = source.replace('        %total = arith.addf %acc, %read : f32', sums + f'        %total = arith.addf %acc, %sum{count-1} : f32')
    return source
