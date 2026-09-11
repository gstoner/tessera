from tessera.compiler.incremental_heap_model import explore_graph, GraphState, successors


def test_incremental_metadata_and_reader_interleavings():
    result = explore_graph()
    assert result['states'] > 1000 and result['counterexample'] is None
    assert explore_graph(omit_barrier=True)['reason'] == 'incremental marking invariant violated'
    assert explore_graph(omit_pins=True)['reason'] == 'reader storage reclaimed'


def test_retirement_admits_old_reader_but_never_a_new_retired_handle():
    state = GraphState(roots=(1, 0), colors=(2, 0), marking=True, readers=(0, 1))
    retired = dict(successors(state))['retire']
    assert retired.lifecycle == (1, 2)
    assert 'pin_1' not in dict(successors(retired))
    assert 'reuse_1' not in dict(successors(retired))
