from tessera.compiler.native_exception_arena import NativeExceptionArena


def test_arena_grows_and_collects_unreachable_cycles():
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
