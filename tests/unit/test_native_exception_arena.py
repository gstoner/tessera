from tessera.compiler.native_exception_arena import NativeExceptionArena


def test_arena_grows_and_collects_unreachable_chain():
    arena = NativeExceptionArena(2, 4)
    leaf = arena.allocate(1, b'leaf')
    root = arena.allocate(2, b'root', edges=(leaf,), root=True)
    assert arena.abi.capacity >= 2
    arena.release(root)
    assert arena.collect() == 2
    assert arena.allocate(3, b'x', root=True) == 0


def test_arena_rejects_dangling_native_edges():
    arena = NativeExceptionArena()
    try:
        arena.allocate(1, b'x', edges=(7,))
    except ValueError as error:
        assert 'live handles' in str(error)
    else:
        raise AssertionError('dangling edge accepted')


def test_exported_pointers_pin_storage_and_block_collection_and_growth():
    import ctypes as ct
    import pytest
    from tessera.compiler.native_exception_arena import _Node
    arena = NativeExceptionArena(1, 4)
    root = arena.allocate(7, b'live', root=True)
    with arena.abi as first, arena.abi as second:
        nodes, payload = first.nodes, first.payload
        with pytest.raises(RuntimeError, match='ABI readers'):
            arena.allocate(2, b'growth', root=True)
        arena.release(root)
        with pytest.raises(RuntimeError, match='ABI readers'):
            arena.collect()
        first.close()
        assert second.nodes == nodes and second.payload == payload
        assert ct.cast(nodes, ct.POINTER(_Node))[root].kind == 7
        assert ct.string_at(payload, 4) == b'live'
        with pytest.raises(RuntimeError, match='ABI readers'):
            arena.allocate(2, b'growth', root=True)
    assert arena.collect() == 1
    arena.allocate(2, b'growth', root=True)
    with pytest.raises(ValueError, match='closed'):
        _ = first.nodes


def test_partial_collection_reuses_coalesced_payload_holes_without_moving_roots():
    import ctypes as ct
    arena = NativeExceptionArena(4, 16)
    left = arena.allocate(1, b'left', root=True)
    a = arena.allocate(2, b'aaaa', root=True)
    b = arena.allocate(2, b'bbbb', root=True)
    right = arena.allocate(1, b'end!', root=True)
    arena.release(a); arena.release(b)
    assert arena.collect() == 2
    for _ in range(200):
        transient = arena.allocate(2, b'12345678', root=True)
        with arena.abi as abi:
            assert abi.payload_capacity == 16 and abi.capacity == 4
            assert ct.string_at(abi.payload, 4) == b'left'
            assert ct.string_at(abi.payload + 12, 4) == b'end!'
        arena.release(transient)
        assert arena.collect() == 1
    assert left in arena._live and right in arena._live


def test_export_keeps_arena_alive_until_reader_completes():
    import gc
    import weakref
    arena = NativeExceptionArena()
    arena.allocate(1, b'kept', root=True)
    reference = weakref.ref(arena)
    lease = arena.abi
    del arena
    gc.collect()
    assert reference() is not None
    lease.close()
    gc.collect()
    assert reference() is None


def test_actual_cycle_lives_through_root_and_is_collected_after_release():
    import ctypes as ct
    import pytest
    from tessera.compiler.native_exception_arena import _Node
    arena = NativeExceptionArena(3, 12)
    first = arena.allocate(1, b'one', root=True)
    second = arena.allocate(2, b'two', edges=(first,))
    arena.set_edges(first, edges=(second,))
    arena.retain(second)
    arena.release(first)
    assert arena.collect() == 0
    with arena.abi as abi:
        nodes = ct.cast(abi.nodes, ct.POINTER(_Node))
        assert nodes[first].cause == second and nodes[second].cause == first
        with pytest.raises(RuntimeError, match='ABI readers'):
            arena.set_edges(first, edges=())
    with pytest.raises(ValueError, match='live handles'):
        arena.set_edges(first, edges=(99,))
    arena.release(second)
    assert arena.collect() == 2
    assert arena._payload_used == 0
    with pytest.raises(ValueError, match='live handle'):
        arena.retain(first)
