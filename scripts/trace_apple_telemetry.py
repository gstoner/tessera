"""Opt-in pytest plugin: attribute process-wide telemetry changes to test order.

Usage in the owning host: pytest -p scripts.trace_apple_telemetry <ordered tests>
Reads only an already-loaded runtime; never loads Metal to inspect a CPU test.
It records after fixture teardown, and does not reset the state or mask a leak.
"""
import json
import os
import sys
from pathlib import Path

import pytest


def _snapshot():
    dispatch = sys.modules.get('tessera._apple_gpu_dispatch')
    handle = getattr(dispatch, '_handle', None)
    if handle is None:
        return None
    probe = getattr(handle, 'tessera_apple_gpu_dispatch_telemetry_enabled', None)
    if probe is None:
        return None
    return {'enabled': bool(probe()), 'image': str(getattr(dispatch, '_dylib_path', None) or getattr(handle, '_name', None))}


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    before = _snapshot()
    yield
    after = _snapshot()
    if before != after:
        record = {'test': item.nodeid, 'before': before, 'after': after}
        path = Path(os.environ.get('TESSERA_TELEMETRY_TRACE', '/tmp/tessera-telemetry-transitions.jsonl'))
        with path.open('a') as output:
            output.write(json.dumps(record, sort_keys=True) + '\n')
