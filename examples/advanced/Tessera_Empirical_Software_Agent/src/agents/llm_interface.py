"""LLM provider shim.

Fill in with your provider (Gemini/OpenAI/self-hosted). The interface below is intentionally
simple so you can swap providers without touching the rest of the stack.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Proposal:
    title: str
    patch_plan: str  # natural-language patch plan
    files: Dict[str, str]  # optional whole-file replacements or snippets
    metadata: Dict[str, Any]

class LLMInterface:
    def propose(self, context: str, k: int = 4) -> List[Proposal]:
        raise NotImplementedError

    def critique(self, code: str, logs: str) -> str:
        """Return a short critique/safety check for the next step."""
        raise NotImplementedError
