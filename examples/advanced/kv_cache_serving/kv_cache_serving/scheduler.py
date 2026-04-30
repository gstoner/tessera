from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Request:
    request_id: str
    tenant: str
    prefix_id: str
    context_tokens: int


@dataclass(frozen=True)
class Placement:
    request_id: str
    worker: int
    route: str
    prefix_id: str


def route_requests(requests: list[Request], decode_workers: int) -> list[Placement]:
    prefix_owner: dict[str, int] = {}
    placements: list[Placement] = []
    for req in requests:
        if req.prefix_id in prefix_owner:
            worker = prefix_owner[req.prefix_id]
            route = "decode_cache_hit"
        else:
            worker = len(prefix_owner) % decode_workers
            prefix_owner[req.prefix_id] = worker
            route = "prefill_then_decode"
        placements.append(Placement(req.request_id, worker, route, req.prefix_id))
    return placements
