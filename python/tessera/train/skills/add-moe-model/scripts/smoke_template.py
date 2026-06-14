"""Smoke-test template for a newly-ported tessera.train model.

Copy to ``tests/unit/test_train_<name>.py``, set ``MODEL_MODULE`` to your model's
module name under ``tessera.train.models``, and rename the test. Mirrors
``tests/unit/test_train_qwen3_moe.py`` and the skill's ``verify.py`` check.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np

# EDIT: the module name under tessera.train.models (e.g. "qwen3_moe", "moba").
MODEL_MODULE = "qwen3_moe"


def _load():
    mod = importlib.import_module(f"tessera.train.models.{MODEL_MODULE}")
    cfg_cls = next(o for n, o in vars(mod).items()
                   if n.endswith("Config") and inspect.isclass(o))
    model_cls = next(o for n, o in vars(mod).items()
                     if n.endswith("Model") and inspect.isclass(o))
    return cfg_cls, model_cls


def test_model_forward_shapes_and_finite_aux():
    cfg_cls, model_cls = _load()
    cfg = cfg_cls()
    model = model_cls(cfg)
    ids = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 8))
    logits, aux = model.forward(ids)

    assert np.asarray(logits).shape == (2, 8, cfg.vocab_size)
    assert np.all(np.isfinite(np.asarray(logits)))
    assert all(np.isfinite(float(v)) for v in aux.values())
