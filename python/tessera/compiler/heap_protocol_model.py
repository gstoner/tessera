"""Finite interleaving model for the v1 stream-ordered heap protocol.

One slot, two readers, two generations; graph reachability is abstracted by a
root bit. This checks lifetime ordering, not hardware memory-model correctness
or general collector termination. Closed host scopes remain pending readers.
"""
from collections import deque
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class State:
    phase: str = 'free'
    generation: int = 0
    rooted: bool = False
    readers: tuple = ((0, 0), (0, 0))  # status (open/pending/uncertain), generation


def transitions(state, *, unsafe_reuse=False):
    idle = all(status == 0 for status, _ in state.readers)
    if state.phase == 'free' and idle and state.generation < 2:
        yield 'reserve', replace(state, phase='reserved', generation=state.generation + 1)
    if state.phase == 'reserved':
        yield 'initialize', replace(state, phase='initialized')
        yield 'abort', replace(state, phase='free')
    if state.phase == 'initialized':
        yield 'publish', replace(state, phase='live', rooted=True)
    if state.phase == 'live' and idle:
        yield 'root_update', replace(state, rooted=not state.rooted)
        if not state.rooted:
            yield 'retire', replace(state, phase='retired')
    if state.phase == 'retired' and (idle or unsafe_reuse):
        yield 'reclaim', replace(state, phase='free')
    for i, (status, generation) in enumerate(state.readers):
        updates = []
        if status == 0 and state.phase in ('live', 'retired'):
            updates = [('acquire', (1, state.generation))]
        elif status == 1:
            updates = [('close', (2, generation)), ('record_failure', (3, generation))]
        elif status == 2:
            updates = [('device_complete', (0, 0))]
        elif status == 3:
            updates = [('explicit_completion', (0, 0))]
        for action, reader in updates:
            readers = list(state.readers)
            readers[i] = reader
            yield f'{action}_{i}', replace(state, readers=tuple(readers))


def explore(*, unsafe_reuse=False):
    """Return state/edge counts and the shortest unsafe trace, if one exists."""
    start = State()
    seen = {start}
    queue: deque[tuple[State, tuple[str, ...]]] = deque([(start, ())])
    edges = 0
    while queue:
        state, trace = queue.popleft()
        if any(status and (generation != state.generation or state.phase not in ('live', 'retired'))
               for status, generation in state.readers):
            return dict(states=len(seen), transitions=edges, counterexample=list(trace))
        if state.rooted and state.phase != 'live':
            raise AssertionError('reachable storage is not live')
        for action, nxt in transitions(state, unsafe_reuse=unsafe_reuse):
            edges += 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, trace + (action,)))
    return dict(states=len(seen), transitions=edges, counterexample=None)
