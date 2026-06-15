"""DLOP-Bench-style long-tail operator fusion benchmark."""

from .core import (
    LONGTAIL_OPS,
    DlopLongtailConfig,
    LongTailOp,
    build_report,
    run_core,
    synthesized_fusion_rows,
    telemetry,
)

__all__ = [
    "LONGTAIL_OPS",
    "DlopLongtailConfig",
    "LongTailOp",
    "build_report",
    "run_core",
    "synthesized_fusion_rows",
    "telemetry",
]
