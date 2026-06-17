"""Lock the lazy top-level export of ``tessera.train``.

``tessera.train`` is bound lazily via the PEP 562 ``__getattr__`` in
``python/tessera/__init__.py`` (see also the agent-native firewall in
``test_train_agent_native_firewall.py``). These tests pin both halves of the
contract: the surface IS reachable as an attribute, AND a bare ``import
tessera`` does NOT eagerly pull the (heavier) training subpackage in.
"""

from __future__ import annotations

import subprocess
import sys


def test_train_reachable_as_attribute():
    import tessera

    # Lazy attribute access resolves the subpackage and its curated surface.
    assert tessera.train.MoERouter.__name__ == "MoERouter"
    assert "train" in dir(tessera)
    assert "MoERouter" in tessera.train.__all__


def test_from_import_resolves():
    from tessera import train

    assert hasattr(train, "Qwen3MoEModel")
    assert hasattr(train, "grpo_step")


def test_bare_import_tessera_does_not_eagerly_load_train():
    """The cheap-import property: ``import tessera`` must not drag in the
    training subpackage or its model files. Run in a fresh interpreter so a
    prior in-session access of ``tessera.train`` can't mask the result."""
    code = (
        "import sys, tessera\n"
        "loaded = [m for m in sys.modules if m == 'tessera.train' "
        "or m.startswith('tessera.train.')]\n"
        "print(';'.join(sorted(loaded)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = [m for m in proc.stdout.strip().split(";") if m]
    assert not leaked, f"import tessera eagerly loaded train modules: {leaked}"


def test_unknown_attribute_still_raises():
    import tessera

    try:
        tessera.this_attribute_does_not_exist
    except AttributeError:
        return
    raise AssertionError("expected AttributeError for an unknown attribute")
