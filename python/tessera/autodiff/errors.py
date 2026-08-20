"""Autodiff exception types — a leaf module with no intra-package imports.

``tape.py`` imports ``vjp.py`` and ``vjp.py`` imports ``degeneracy.py``, so any
module the rules need must sit *below* that cycle. Keeping the exception
hierarchy here lets both ends import it without ordering games; ``tape.py``
re-exports :class:`TesseraAutodiffError` so every existing
``from .tape import TesseraAutodiffError`` keeps naming the same class object.
"""

from __future__ import annotations

__all__ = [
    "TesseraAutodiffError",
    "TesseraDegeneracyError",
    "TesseraDegeneracyWarning",
]


class TesseraAutodiffError(RuntimeError):
    """Raised on misuse of the autodiff machinery (no VJP, scalar shape, etc.)."""


class TesseraDegeneracyError(TesseraAutodiffError):
    """A factorization derivative was requested where it does not exist.

    Subclasses :class:`TesseraAutodiffError` so existing
    ``except TesseraAutodiffError`` handlers keep working. Raised by the
    declared-policy guards in ``degeneracy.py`` — see that module for the
    ``degeneracy_policy`` semantic key and why the guard fails closed.
    """


class TesseraDegeneracyWarning(UserWarning):
    """A factorization derivative exists here but has lost most of its digits.

    Raised (as a warning, not an error) in the band between "numerically
    singular" — where the derivative does not exist and the guard fails closed —
    and "well conditioned". Decision #21a puts the boundary there deliberately:
    non-existence is a *semantic* condition and must fail closed, while poor
    conditioning is an *accuracy* condition, which may proceed but must say so.
    Promote it with ``warnings.simplefilter("error", TesseraDegeneracyWarning)``.
    """
