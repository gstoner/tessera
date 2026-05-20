"""Optional LOTUS adapter for the DeepScholar-Bench concept.

The active, dependency-light benchmark is
``tessera_deepscholar_model.py``.  This module is deliberately only an
adapter shell: it imports cleanly without LOTUS or pandas and raises a
clear error when the optional integration is requested without those
extras installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


try:  # pragma: no cover - optional research dependency.
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:  # pragma: no cover - optional research dependency.
    import lotus
except Exception:  # pragma: no cover
    lotus = None  # type: ignore[assignment]


class OptionalDependencyError(RuntimeError):
    """Raised when the LOTUS adapter is used without research extras."""


def _require_optional_stack() -> None:
    missing = []
    if pd is None:
        missing.append("pandas")
    if lotus is None:
        missing.append("lotus")
    if missing:
        raise OptionalDependencyError(
            "DeepScholar LOTUS integration requires optional packages: "
            + ", ".join(missing)
        )


@dataclass
class TesseraLotusDeepScholar:
    """Minimal adapter boundary for future LOTUS work.

    The previous version imported non-existent ``tessera.models`` APIs at
    construction time.  This version keeps import-time clean and leaves a
    narrow, explicit place to attach the real LOTUS semantic operators when
    the research stack is available.
    """

    config: dict[str, Any]

    def __post_init__(self) -> None:
        _require_optional_stack()
        if hasattr(lotus, "settings"):
            lotus.settings.configure(**self.config.get("lotus_settings", {}))

    async def research_synthesis_pipeline(
        self,
        query: str,
        arxiv_papers: Any,
        exemplar_related_work: str | None = None,
    ) -> dict[str, Any]:
        _require_optional_stack()
        if not hasattr(arxiv_papers, "sem_filter"):
            raise OptionalDependencyError(
                "LOTUS semantic dataframe accessors are not registered; "
                "install/configure LOTUS before running the research pipeline"
            )
        relevant = arxiv_papers.sem_filter(
            f"The paper abstract is relevant to: {query}"
        )
        return {
            "query": query,
            "relevant_count": int(len(relevant)),
            "has_exemplar": exemplar_related_work is not None,
            "status": "lotus_adapter_smoke",
        }


__all__ = [
    "OptionalDependencyError",
    "TesseraLotusDeepScholar",
]
