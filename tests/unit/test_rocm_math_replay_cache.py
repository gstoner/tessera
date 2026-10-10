"""Warm native math replay retains original Graph and ABI rejection checks."""
from dataclasses import replace
import pytest
from tessera.compiler import rocm_math_native as native, rocm_pass_cache as cache
from tests.unit.test_rocm_math_native_package import module

pytestmark = pytest.mark.compiler_route

@pytest.fixture
def math_recipe(production_compiler, monkeypatch):
    recipe = native.lower_math_graph(module(), "rocm_gfx1151")
    execute = native.run_tessera_opt
    calls = []
    def observed(*args):
        calls.append(args[2])
        return execute(*args)
    monkeypatch.setattr(native, "run_tessera_opt", observed)
    cache.clear()
    def prime():
        recipe.validate()
        assert calls == ["--tessera-graph-to-schedule", "--tessera-schedule-to-tile"]
        calls.clear()
        return recipe, calls
    yield prime
    cache.clear()

def test_repeated_math_ancestry_reuses_native_passes(math_recipe):
    recipe, calls = math_recipe()
    recipe.validate()
    assert calls == []

@pytest.mark.parametrize("field", ["graph_ir", "schedule_ir", "tile_ir", "target"])
def test_cached_math_ancestry_rejects_changed_contract(math_recipe, field):
    recipe, calls = math_recipe()
    changed = ("rocm_gfx1201" if field == "target" else
               getattr(recipe, field).replace("sqrt", "exp"))
    with pytest.raises((ValueError, RuntimeError)):
        replace(recipe, **{field: changed}).validate()
    # A refused request cannot poison the existing valid recipe.
    recipe.validate()
    count = len(calls)
    recipe.validate()
    assert len(calls) == count

def test_math_graph_change_requires_new_native_projection(math_recipe):
    recipe, calls = math_recipe()
    replace(recipe, graph_ir=recipe.graph_ir+"\n").validate()
    assert calls == ["--tessera-graph-to-schedule"]

def test_native_graph_errors_are_replayed_instead_of_cached(math_recipe):
    recipe, calls = math_recipe()
    invalid = replace(recipe, graph_ir="module { invalid_graph }")
    for _ in range(2):
        with pytest.raises(RuntimeError):
            invalid.validate()
    assert calls == ["--tessera-graph-to-schedule"]*2
