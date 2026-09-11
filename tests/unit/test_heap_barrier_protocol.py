import pytest
from dataclasses import replace
from tessera.compiler.heap_protocol_model import State, explore, transitions
from tessera.compiler.heap_barrier_contract import read_heap_contract
from tessera.compiler.gpu_heap_collection import emit_pool


def test_all_bounded_interleavings_preserve_readers():
    result = explore()
    assert result['states'] > 100
    assert result['counterexample'] is None
    # Guard the model: missing reclamation dependencies must have a witness.
    broken = explore(unsafe_reuse=True)
    assert broken['counterexample'][-1] == 'reclaim'
    assert any(a.startswith('acquire') for a in broken['counterexample'])


def test_pending_and_uncertain_readers_are_not_host_scope_completion():
    for status in (1, 2, 3):
        state = State('retired', 1, False, ((status, 1), (0, 0)))
        assert 'reclaim' not in dict(transitions(state))
    assert not dict(transitions(State('free', 2)))  # generation exhaustion
    assert 'publish' not in dict(transitions(State('reserved', 1)))


@pytest.mark.parametrize('mode', ['allocate', 'graph_checked', 'retire', 'reclaim'])
def test_protocol_is_serialized_and_rejects_silent_semantic_changes(mode):
    source, _ = emit_pool(3, 8, mode)
    data = read_heap_contract(source)
    assert data['mode'] == mode and data['slots'] == 3
    with pytest.raises(ValueError, match='protocol'):
        read_heap_contract(source.replace('exclusive_stream_epoch', 'device_atomic'))
    with pytest.raises(ValueError, match='protocol'):
        read_heap_contract(source.replace('schema\\22:1', 'schema\\22:true'))
    with pytest.raises(ValueError, match='exactly one'):
        read_heap_contract(source + source)


def test_graph_update_validation_precedes_all_graph_stores():
    source, specs = emit_pool(3, 8, 'graph_checked')
    assert specs[-2].name == 'status'
    gate = source.index('scf.if %valid')
    assert 'llvm.store' not in source[:gate]


def test_retired_slot_cannot_be_rooted_in_model():
    state = replace(State(), phase='retired', generation=1)
    assert 'root_update' not in dict(transitions(state))
