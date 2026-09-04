import numpy as np
import pytest
from tessera.compiler import evaluator
from tessera.autodiff.ann_laws import ann_identity_checks


@pytest.mark.parametrize('native', [(True, True), (True, False), (False, True), (False, False)])
def test_pair_requires_both_native_programs(monkeypatch, native):
    first, second = object(), object()
    seen = []
    def run(target, program, args):
        seen.append(program)
        flag = native[0 if program is first else 1]
        return np.array([1., 2.]) if flag else None, flag
    monkeypatch.setattr(evaluator, 'run_native', run)
    verdict = evaluator.program_pair_equivalence('x86', first, second, (), ())
    assert seen == [first, second]
    assert verdict.relation == ('equivalent' if all(native) else 'inconclusive')


def test_pair_detects_composition_bias_mutation(monkeypatch):
    rng = np.random.default_rng(33)
    x, a, b, c, d = rng.normal(size=(7, 2)), rng.normal(size=(2, 3)), rng.normal(size=3), rng.normal(size=(3, 4)), rng.normal(size=4)
    original = lambda value: (value @ a + b) @ c + d
    composed = lambda value: value @ (a @ c) + b @ c + d
    mutated = lambda value: composed(value) + 1
    # Synthetic native provenance validates adapter logic, not hardware execution.
    monkeypatch.setattr(evaluator, 'run_native', lambda target, fn, args: (fn(*args), True))
    assert evaluator.program_pair_equivalence('x86', original, composed, (x,), (x,)).relation == 'equivalent'
    assert evaluator.program_pair_equivalence('x86', original, mutated, (x,), (x,)).relation == 'divergent'


def test_pair_refuses_invalid_tolerance():
    with pytest.raises(ValueError, match='finite'):
        evaluator.program_pair_equivalence('x86', None, None, (), (), rtol=float('nan'))


def test_ann_reference_laws_are_separate():
    results = ann_identity_checks()
    assert len(results) == 2
    assert all(result.status == 'pass' and result.registry == 'ann_calculus' for result in results)
