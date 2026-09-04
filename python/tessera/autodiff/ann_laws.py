"""MSW-9 reference laws registered independently of native evaluator evidence."""
import numpy as np


def ann_identity_checks():
    from .laws import LawResult

    rng = np.random.default_rng(901)
    residuals: dict[str, list[float]] = {'affine_composition': [], 'relu_identity_extension': []}
    for n, hidden, out in ((2, 5, 3), (4, 3, 7), (1, 1, 1)):
        x = rng.normal(size=(11, n))
        w1, b1 = rng.normal(size=(n, hidden)), rng.normal(size=hidden)
        w2, b2 = rng.normal(size=(hidden, out)), rng.normal(size=out)
        oracle = (x @ w1 + b1) @ w2 + b2
        merged = x @ (w1 @ w2) + b1 @ w2 + b2
        residuals['affine_composition'].append(float(np.max(np.abs(oracle - merged))) / max(1, float(np.max(np.abs(oracle)))))
        y = x @ w1 + b1
        extension = np.maximum(y, 0) - np.maximum(-y, 0)
        residuals['relu_identity_extension'].append(float(np.max(np.abs(y - extension))))
    return [LawResult(name, 'ann_calculus', 'realization',
                      'pass' if max(values) < 1e-12 else 'fail', len(values), max(values),
                      'reference algebra; no native promotion')
            for name, values in residuals.items()]
