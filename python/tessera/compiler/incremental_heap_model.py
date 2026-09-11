"""Two-object exploration of incremental updates and pinned retirement.

Metadata actions are linearized. Payload readers can remain admitted through
retirement. This is not a weak-memory or multiple-writer model.
"""
from collections import deque
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class GraphState:
    lifecycle: tuple = (1, 1)  # free/live/retired
    roots: tuple = (1, 0)
    edges: tuple = (-1, -1)
    colors: tuple = (0, 0)  # white/grey/black
    readers: tuple = (0, 0)  # absent/open/pending/uncertain
    marking: bool = False


def _set(values, i, value):
    return values[:i] + (value,) + values[i + 1:]


def successors(s, *, omit_barrier=False, omit_pins=False):
    if not s.marking:
        yield 'begin', replace(s, marking=True, colors=tuple(int(bool(r)) for r in s.roots))
    for i in range(2):
        if s.lifecycle[i] == 1:
            roots = _set(s.roots, i, 1 - s.roots[i])
            colors = s.colors
            if s.marking and roots[i] and not omit_barrier and not colors[i]:
                colors = _set(colors, i, 1)
            yield f'root_{i}', replace(s, roots=roots, colors=colors)
            for target in (-1, 0, 1):
                if target >= 0 and s.lifecycle[target] != 1:
                    continue
                edges = _set(s.edges, i, target)
                colors = list(s.colors)
                if s.marking and not omit_barrier:
                    for source, dest in enumerate(edges):
                        if s.lifecycle[source] == 1 and dest >= 0 and colors[dest] == 0:
                            colors[dest] = 1
                yield f'edge_{i}_{target}', replace(s, edges=edges, colors=tuple(colors))
            if not s.readers[i]:
                yield f'pin_{i}', replace(s, readers=_set(s.readers, i, 1))
        if s.readers[i] == 1:
            yield f'close_{i}', replace(s, readers=_set(s.readers, i, 2))
            yield f'failed_record_{i}', replace(s, readers=_set(s.readers, i, 3))
        elif s.readers[i] in (2, 3):
            yield f'proven_complete_{i}', replace(s, readers=_set(s.readers, i, 0))
        if s.marking and s.colors[i] == 1:
            colors = _set(s.colors, i, 2)
            target = s.edges[i]
            if target >= 0 and colors[target] == 0:
                colors = _set(colors, target, 1)
            yield f'scan_{i}', replace(s, colors=colors)
        if not s.marking and s.lifecycle[i] == 2 and (not s.readers[i] or omit_pins):
            yield f'reuse_{i}', replace(s, lifecycle=_set(s.lifecycle, i, 0))
    closed = (1 not in s.colors and all(not r or s.colors[i] == 2 for i, r in enumerate(s.roots))
              and all(s.colors[i] != 2 or target < 0 or s.colors[target] == 2
                      for i, target in enumerate(s.edges)))
    if s.marking and closed:
        yield 'retire', replace(s, marking=False, lifecycle=tuple(
            2 if live == 1 and s.colors[i] == 0 else live for i, live in enumerate(s.lifecycle)))


def explore_graph(*, omit_barrier=False, omit_pins=False):
    start = GraphState()
    seen = {start}
    queue: deque[tuple[GraphState, tuple[str, ...]]] = deque([(start, ())])
    edges = 0
    while queue:
        s, trace = queue.popleft()
        reason = None
        if any(reader and s.lifecycle[i] == 0 for i, reader in enumerate(s.readers)):
            reason = 'reader storage reclaimed'
        if s.marking and (any(r and s.colors[i] == 0 for i, r in enumerate(s.roots)) or
                          any(s.colors[i] == 2 and target >= 0 and s.colors[target] == 0
                              for i, target in enumerate(s.edges))):
            reason = 'incremental marking invariant violated'
        if reason:
            return dict(states=len(seen), transitions=edges, counterexample=list(trace), reason=reason)
        for action, nxt in successors(s, omit_barrier=omit_barrier, omit_pins=omit_pins):
            edges += 1
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, trace + (action,)))
    return dict(states=len(seen), transitions=edges, counterexample=None, reason=None)
