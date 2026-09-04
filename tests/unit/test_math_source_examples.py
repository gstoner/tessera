"""The tutorials' printed numerical claims are executable acceptance checks."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    spec = importlib.util.spec_from_file_location("math_source_example", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    # dataclasses consult the defining module while decorating.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tensor_calculus_tutorial():
    result = load("examples/tensor_calculus/tensor_calculus_tutorial.py").run()
    assert set(result) == {"cartesian", "cylindrical", "spherical"}


def test_pde_learning_tutorial():
    result = load("examples/pde_learning/pde_learning_tutorial.py").run()
    assert len(result["pinn_allen_cahn"]["rms_errors"]) == 3


def test_ann_calculus_design_spike():
    assert load("tools/ann_calculus_spike.py").run() == 15
