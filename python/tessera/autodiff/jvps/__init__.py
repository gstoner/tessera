"""Arch-7 (2026-05-22) — JVP family subpackage scaffold.

Mirrors `vjps/__init__.py` for the forward-mode JVP registry.  See
that file for migration rules.

Today `python/tessera/autodiff/jvp.py` is 3150 LOC carrying every
JVP registration in one file.  This package is the migration target;
the actual code movement is a per-family follow-up sprint.

Drift gate at ``tests/unit/test_autodiff_split_scaffold.py`` locks
the registration count so a partial migration that drops JVPs in
transit fails.
"""

from __future__ import annotations

# Each family submodule is imported here for side effects (their
# register_jvp() calls).  When a family is migrated, uncomment its
# import line.

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
