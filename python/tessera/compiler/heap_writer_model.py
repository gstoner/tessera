"""Reservation model for future racing metadata writers, not a native protocol.

Compare atomic try-reservation to a split validate/write. The model assumes
atomic linearization; scope, weak memory and device progress remain separate.
"""
from collections import deque
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class WriterState:
    owner: int = -1
    pc: tuple = (0, 0)  # initial/checked/reserved/initialized/published/refused
    initialized: bool = False
    published: bool = False


def explore_writers(*, split_reservation=False):
    start = WriterState()
    queue: deque[tuple[WriterState, tuple[str, ...]]] = deque([(start, ())])
    seen = {start}
    transitions = 0
    while queue:
        s, trace = queue.popleft()
        claimants = [i for i, pc in enumerate(s.pc) if pc in (2, 3, 4)]
        if any(i != s.owner for i in claimants) or (s.published and not s.initialized):
            return dict(states=len(seen), transitions=transitions, counterexample=list(trace))
        for i, pc in enumerate(s.pc):
            owner, initialized, published = s.owner, s.initialized, s.published
            if pc == 0:
                if s.owner != -1:
                    nxt, action = 5, 'refused'
                elif split_reservation:
                    nxt, action = 1, 'checked'
                else:
                    owner, nxt, action = i, 2, 'try_reserve'
            elif pc == 1:
                owner, nxt, action = i, 2, 'unchecked_reserve'
            elif pc == 2:
                initialized, nxt, action = True, 3, 'initialize'
            elif pc == 3:
                published, nxt, action = True, 4, 'publish'
            else:
                continue
            candidate = replace(s, owner=owner, initialized=initialized, published=published,
                                pc=s.pc[:i] + (nxt,) + s.pc[i + 1:])
            transitions += 1
            if candidate not in seen:
                seen.add(candidate)
                queue.append((candidate, trace + (f'{i}:{action}',)))
    return dict(states=len(seen), transitions=transitions, counterexample=None)


def explore_retirement(*, omit_gate=False):
    """One white object: graph publication races final retirement.

    Acquire/release are assumed linearizable; this is not a weak-memory model.
    """
    # gate, writer pc, retire pc, live, rooted, grey
    start = (-1, 0, 0, True, False, False)
    queue: deque[tuple[tuple, tuple[str, ...]]] = deque([(start, ())])
    seen = {start}
    transitions = 0
    while queue:
        state, trace = queue.popleft()
        gate, w, r, live, rooted, grey = state
        if rooted and not live:
            return dict(states=len(seen), transitions=transitions, counterexample=list(trace))
        for actor in (0, 1):
            g, wp, rp, alive, root, mark = state
            pc = wp if actor == 0 else rp
            if pc == 0:
                if not omit_gate and g != -1:
                    nxt, action = 5, 'busy'
                else:
                    g, nxt, action = actor, 1, 'acquire'
            elif pc == 1:
                allowed = alive if actor == 0 else alive and not mark and not root
                nxt, action = (2 if allowed else 4), 'validate'
            elif pc == 2:
                if actor == 0:
                    mark = True
                else:
                    alive = False
                nxt, action = 3, 'shade' if actor == 0 else 'retire'
            elif pc == 3:
                if actor == 0:
                    root = True
                nxt, action = 4, 'publish'
            elif pc == 4:
                g, nxt, action = -1, 5, 'release'
            else:
                continue
            candidate = (g, nxt if actor == 0 else wp, nxt if actor == 1 else rp, alive, root, mark)
            transitions += 1
            if candidate not in seen:
                seen.add(candidate)
                queue.append((candidate, trace + (f'{actor}:{action}',)))
    return dict(states=len(seen), transitions=transitions, counterexample=None)
