import os, json, yaml, pathlib, time
import numpy as np
import pytest

from tessera_numerics import tessera_adapter as A

def pytest_addoption(parser):
    parser.addoption("--config", action="store", default=None, help="Path to config.yaml overrides")
    parser.addoption("--seed", action="store", default="1234", help="RNG seed")
    parser.addoption("--report", action="store", default="reports/numerics_summary.json", help="Report path")

@pytest.fixture(scope="session")
def cfg(request):
    here = pathlib.Path(__file__).resolve().parent.parent
    cfg_path = here / "config.yaml"
    with open(cfg_path, "r") as f:
        base = yaml.safe_load(f)
    override = request.config.getoption("--config")
    if override:
        with open(override, "r") as f:
            ov = yaml.safe_load(f)
        base.update(ov or {})
    return base

@pytest.fixture(scope="session", autouse=True)
def seeded(request):
    seed = int(request.config.getoption("--seed"))
    A.rng(seed)
    np.random.seed(seed)
    yield

@pytest.fixture(scope="session", autouse=True)
def report_writer(request):
    report_path = pathlib.Path(request.config.getoption("--report"))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"backend": A.backend_name(), "start": time.time(), "results": []}
    yield data
    data["end"] = time.time()
    with open(report_path, "w") as f:
        json.dump(data, f, indent=2)

@pytest.fixture
def rec(report_writer):
    def _rec(name, payload):
        report_writer["results"].append({"name": name, **payload})
    return _rec
