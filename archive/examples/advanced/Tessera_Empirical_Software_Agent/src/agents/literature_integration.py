"""Placeholders for literature/idea ingestion (papers, textbooks, search results).

In production, you would plumb this to your curated corpora or search connectors and
distill into short 'idea cards' that the LLM can reference during rewriting.
"""
from dataclasses import dataclass
from typing import List

@dataclass
class IdeaCard:
    title: str
    summary: str
    citation: str

def load_idea_bank(task_name: str) -> List[IdeaCard]:
    return [IdeaCard(title="Baseline heuristics", summary="Try a strong baseline + safety fallbacks.", citation="internal")]
