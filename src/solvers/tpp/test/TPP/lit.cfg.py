"""Lit configuration for the TPP solver fixtures.

Mirrors `tests/tessera-ir/lit.cfg.py` — discovers `*.mlir` files
and substitutes `tessera-opt` so the fixture RUN lines work with
or without the binary on PATH.
"""

import os
import lit.formats

config.name = "Tessera-TPP"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = ['.mlir']
config.environment['PATH'] = os.environ.get('PATH', '')
config.substitutions.append(('tessera-opt', 'tessera-opt -allow-unregistered-dialect'))
