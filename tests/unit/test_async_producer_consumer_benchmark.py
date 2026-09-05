"""The ablation must add a wait without changing the producer/consumer body."""
import pytest

from benchmarks.nvidia.benchmark_async_producer_consumer import serialize_prefetch


SOURCE = '''module {
  nvvm.cp.async.commit.group
  nvvm.cp.async.wait.group 0
  scf.for %i = %lo to %hi step %step {
    nvvm.cp.async.commit.group
    nvvm.mma.sync
    nvvm.cp.async.wait.group 0
  }
}
'''


def test_serialized_control_retains_all_original_instructions():
    result = serialize_prefetch(SOURCE)
    assert result == SOURCE.replace('    nvvm.cp.async.commit.group\n',
                                    '    nvvm.cp.async.commit.group\n    nvvm.cp.async.wait.group 0\n')


@pytest.mark.parametrize('source', [SOURCE.replace('    nvvm.cp.async.commit.group\n', ''),
                                    SOURCE.replace('  nvvm.cp.async.wait.group 0\n', ''),
                                    SOURCE + 'nvvm.cp.async.commit.group\n',
                                    serialize_prefetch(SOURCE)])
def test_ablation_rejects_changed_native_protocol(source):
    with pytest.raises(ValueError):
        serialize_prefetch(source)
