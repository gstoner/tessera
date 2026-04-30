from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DraftNode:
    tokens: tuple[str, ...]
    score: float


class DraftModel:
    vocab = (" efficient", " robust", " sparse", " verified", " cached")

    def expand(self, prefix: tuple[str, ...], branching: int) -> list[DraftNode]:
        nodes = []
        for idx, token in enumerate(self.vocab[:branching]):
            nodes.append(DraftNode(prefix + (token,), 1.0 / (idx + 1)))
        return nodes


class TargetModel:
    def accepts(self, node: DraftNode) -> bool:
        joined = "".join(node.tokens)
        return "verified" in joined or "cached" in joined or len(node.tokens) == 1


def speculative_step(prompt: str, draft: DraftModel, target: TargetModel, depth: int, branching: int) -> list[str]:
    frontier = [DraftNode((), 1.0)]
    accepted: list[str] = []
    for _ in range(depth):
        candidates = []
        for node in frontier:
            candidates.extend(draft.expand(node.tokens, branching))
        candidates.sort(key=lambda n: n.score, reverse=True)
        accepted_nodes = [node for node in candidates if target.accepts(node)]
        if not accepted_nodes:
            break
        best = accepted_nodes[0]
        accepted = list(best.tokens)
        frontier = accepted_nodes[:branching]
    return accepted
