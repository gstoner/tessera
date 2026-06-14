#!/usr/bin/env python3
"""Verifiable success check for the add-moe-model skill.

Usage:
    PYTHONPATH=python python3 .../add-moe-model/scripts/verify.py <name>

<name> is the model module under tessera.train.models (e.g. "qwen3_moe").
The module must export a ``*Config`` and a ``*Model`` whose ``forward(ids)``
returns ``(logits, aux)`` with ``logits`` shaped ``(B, S, vocab_size)`` and
``aux`` a dict of finite scalar losses.

Prints PASS and exits 0 on success; prints FAIL: <reason> and exits 1 otherwise.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from typing import NoReturn

import numpy as np


def _fail(msg: str) -> NoReturn:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def main(name: str) -> None:
    try:
        mod = importlib.import_module(f"tessera.train.models.{name}")
    except Exception as e:  # noqa: BLE001
        _fail(f"import tessera.train.models.{name}: {e!r}")

    cfg_cls = next((o for n, o in vars(mod).items()
                    if n.endswith("Config") and inspect.isclass(o)), None)
    model_cls = next((o for n, o in vars(mod).items()
                      if n.endswith("Model") and inspect.isclass(o)), None)
    if cfg_cls is None or model_cls is None:
        _fail(f"module must export a *Config and a *Model (found {cfg_cls}, {model_cls})")

    # Tiny config: rely on dataclass defaults; shrink if the field exists.
    try:
        cfg = cfg_cls()
    except Exception as e:  # noqa: BLE001
        _fail(f"instantiate {cfg_cls.__name__}(): {e!r}")

    try:
        model = model_cls(cfg)
    except Exception as e:  # noqa: BLE001
        _fail(f"instantiate {model_cls.__name__}(cfg): {e!r}")

    B, S = 2, 8
    vocab = getattr(cfg, "vocab_size", None)
    if not vocab:
        _fail("config has no vocab_size")
    ids = np.random.default_rng(0).integers(0, vocab, size=(B, S))

    try:
        out = model.forward(ids)
    except Exception as e:  # noqa: BLE001
        _fail(f"forward(ids): {e!r}")

    if not (isinstance(out, tuple) and len(out) == 2):
        _fail("forward must return (logits, aux)")
    logits, aux = out
    logits = np.asarray(logits)
    if logits.shape != (B, S, vocab):
        _fail(f"logits shape {logits.shape} != {(B, S, vocab)}")
    if not np.all(np.isfinite(logits)):
        _fail("logits contain non-finite values")
    if not isinstance(aux, dict):
        _fail("aux must be a dict of scalar losses")
    for k, v in aux.items():
        if not np.isfinite(float(v)):
            _fail(f"aux loss {k!r} is non-finite: {v}")

    print(f"PASS: {name} — logits {logits.shape}, aux {sorted(aux)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: verify.py <model_module_name>")
        raise SystemExit(2)
    main(sys.argv[1])
