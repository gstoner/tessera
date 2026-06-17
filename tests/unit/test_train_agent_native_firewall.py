"""Agent-native firewall for tessera.train.

This invariant is cross-cutting (not specific to any one model), so it lives in
its own file. ``tessera.train`` must not pull in the compiler audit/registry
machinery — ``primitive_coverage`` / ``op_catalog`` / ``backend_manifest`` are
the implicit indirection PithTrain measures as costly for agents; they belong
behind ``@tessera.jit``, not in the training read-path. See
``python/tessera/train/__init__.py`` ("The agent-native firewall").
"""

from __future__ import annotations


def test_agent_native_firewall():
    import sys

    forbidden = {
        "tessera.compiler.primitive_coverage",
        "tessera.compiler.op_catalog",
        "tessera.compiler.backend_manifest",
    }
    # Drop any that a *prior* test already imported, then import train fresh.
    for name in list(sys.modules):
        if name in forbidden:
            del sys.modules[name]
    for name in [n for n in list(sys.modules) if n.startswith("tessera.train")]:
        del sys.modules[name]

    import tessera.train  # noqa: F401

    leaked = forbidden & set(sys.modules)
    assert not leaked, f"tessera.train leaked compiler registry imports: {leaked}"
