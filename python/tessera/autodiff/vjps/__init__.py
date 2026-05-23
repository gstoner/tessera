"""Arch-7 (2026-05-22) — VJP family subpackage scaffold.

Today `python/tessera/autodiff/vjp.py` is 4262 LOC carrying every
VJP registration in one file.  The plan is to split it by primitive
family (tensor algebra, attention, reductions, losses, quantization,
spectral, sparse, linalg, GA / EBM / complex, recurrent, collectives,
quant STE) into family modules under this package.

This file is the **scaffold landing** (Arch-7).  The submodule
files are created today but mostly empty — they import
:func:`tessera.autodiff.vjp.register_vjp` and demonstrate the
registration pattern, but the bulk-code migration is a per-family
follow-up sprint.

Migration rules
---------------

1. Each family file (`tensor_algebra.py`, `attention.py`, etc.) is
   imported by ``vjp.py`` AFTER ``register_vjp`` is defined, so the
   submodule's import-side-effect registrations land in the global
   ``_VJPS`` dict.

2. A submodule registers via the canonical pattern::

      from ..vjp import register_vjp
      import numpy as np

      def _vjp_foo(dout, x, **kw):
          ...
          return (dx,)

      register_vjp("foo", _vjp_foo)

3. The drift gate at ``tests/unit/test_autodiff_split_scaffold.py``
   asserts the total VJP count stays at-or-above the baseline.  When
   you move 30 VJPs from ``vjp.py`` to ``vjps/tensor_algebra.py``,
   the count stays the same.  When you accidentally drop one in
   transit, the gate catches it.

4. New VJPs should land in the appropriate family module — NOT in
   ``vjp.py``.

Status
------

This subpackage is the **migration target**, not yet the migration
source.  ``vjp.py`` still owns the 241 registrations today.  Future
sprints flip families one at a time; each migration is a
mechanical move + a counter check.
"""

from __future__ import annotations

# Each family submodule is imported here for side effects (their
# register_vjp() calls).  When a family is migrated, uncomment its
# import line.  Today they're all empty so no-op imports are fine.

# from . import tensor_algebra        # noqa: F401 — pending migration
# from . import attention             # noqa: F401 — pending migration
# from . import reductions            # noqa: F401 — pending migration
# from . import losses                # noqa: F401 — pending migration
# from . import quantization          # noqa: F401 — pending migration
# from . import spectral              # noqa: F401 — pending migration
# from . import sparse                # noqa: F401 — pending migration
# from . import linalg                # noqa: F401 — pending migration
# from . import ga_ebm                # noqa: F401 — pending migration
# from . import complex_family        # noqa: F401 — pending migration
# from . import recurrent             # noqa: F401 — pending migration
# from . import collectives           # noqa: F401 — pending migration
