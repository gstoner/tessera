"""KV-cache compression + disaggregated serving: plan *and* execute.

``compression``/``scheduler`` model the routing and memory-accounting decisions;
``serving`` runs those decisions against the real ``tessera.cache`` block-paged
cache manager (page allocation, prefix reuse, ragged decode, eviction/reclaim).
"""

from .compression import CachePolicy, estimate_request_cache
from .scheduler import Placement, Request, route_requests
from .serving import ServingSummary, run_serving_demo

__all__ = [
    "CachePolicy",
    "estimate_request_cache",
    "Placement",
    "Request",
    "route_requests",
    "ServingSummary",
    "run_serving_demo",
]
