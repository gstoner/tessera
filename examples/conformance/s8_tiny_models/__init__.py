"""S8 tiny-model conformance manifest.

These examples are intentionally small enough for unit tests but shaped like
real model families so compiler changes can be validated without waiting for
Phase G hardware execution.
"""

from .models import TinyModelSpec, manifest, surfaces_covered

__all__ = ["TinyModelSpec", "manifest", "surfaces_covered"]

