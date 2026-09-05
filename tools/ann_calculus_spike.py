"""MSW-9 executable design spike. Reference oracle, not a compiler transform."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Network:
    # Row-vector convention: x @ W + b, activation between affine layers.
    layers: tuple
    activation: str = "relu"

    def __post_init__(self):
        if self.activation not in ("relu", "swish") or not self.layers:
            raise ValueError("spike accepts only nonempty affine/ReLU or affine/Swish chains")
        for i, (w, b) in enumerate(self.layers):
            if w.ndim != 2 or b.shape != (w.shape[1],) or (i and self.layers[i - 1][0].shape[1] != w.shape[0]):
                raise ValueError("incompatible affine chain")

    @property
    def widths(self):
        return (self.layers[0][0].shape[0],) + tuple(w.shape[1] for w, b in self.layers)

    @property
    def parameters(self):
        # Dense slots, including fixed zero blocks; not unique trainable buffers.
        return sum((a + 1) * b for a, b in zip(self.widths, self.widths[1:]))

    def __call__(self, x):
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i + 1 < len(self.layers):
                x = np.maximum(x, 0) if self.activation == "relu" else x / (1 + np.exp(-x))
        return x


def compose(first, second):
    if first.activation != second.activation or first.widths[-1] != second.widths[0]:
        raise ValueError("composition contract mismatch")
    a, b = first.layers[-1]
    c, d = second.layers[0]
    merged = (a @ c, b @ c + d)
    result = Network(first.layers[:-1] + (merged,) + second.layers[1:], first.activation)
    expected = (
        first.parameters + second.parameters - (a.size + b.size + c.size + d.size) + (a.shape[0] + 1) * c.shape[1]
    )
    check_accounting(result, expected, len(first.layers) + len(second.layers) - 1)
    return result


def check_accounting(network, expected_parameters, expected_length):
    if network.parameters != expected_parameters or len(network.layers) != expected_length:
        raise ValueError("ANN accounting disagrees with the derived affine chain")


def parallel(first, second):
    if len(first.layers) != len(second.layers) or first.activation != second.activation:
        raise ValueError("parallelization requires equal lengths and activations")
    layers = []
    for (a, b), (c, d) in zip(first.layers, second.layers):
        w = np.zeros((a.shape[0] + c.shape[0], a.shape[1] + c.shape[1]))
        w[: a.shape[0], : a.shape[1]] = a
        w[a.shape[0] :, a.shape[1] :] = c
        layers.append((w, np.concatenate((b, d))))
    return Network(tuple(layers), first.activation)


def identity(n):
    return Network(
        (
            (np.concatenate((np.eye(n), -np.eye(n)), axis=1), np.zeros(2 * n)),
            (np.concatenate((np.eye(n), -np.eye(n)), axis=0), np.zeros(n)),
        )
    )


def extend(network, length):
    if network.activation != "relu" or length < len(network.layers):
        raise ValueError("extension requires ReLU and a nondecreasing length")
    while len(network.layers) < length:
        network = compose(network, identity(network.widths[-1]))
    return network


def summed(first, second):
    if first.widths[0] != second.widths[0] or first.widths[-1] != second.widths[-1]:
        raise ValueError("sum needs matching input/output widths")
    n, m = first.widths[0], first.widths[-1]
    duplicate = Network(((np.concatenate((np.eye(n), np.eye(n)), axis=1), np.zeros(2 * n)),), first.activation)
    add = Network(((np.concatenate((np.eye(m), np.eye(m)), axis=0), np.zeros(m)),), first.activation)
    return compose(compose(duplicate, parallel(first, second)), add)


def run():
    rng = np.random.default_rng(900)
    probes = 0
    for activation in ("relu", "swish"):

        def make(widths):
            return Network(
                tuple((rng.normal(size=(a, b)) * 0.2, rng.normal(size=b) * 0.2) for a, b in zip(widths, widths[1:])),
                activation,
            )

        f, g, h = make((3, 5, 4)), make((4, 6, 2)), make((2, 4, 3))
        x = rng.normal(size=(32, 3))
        np.testing.assert_allclose(compose(f, g)(x), g(f(x)), atol=1e-12)
        probes += 1
        np.testing.assert_allclose(compose(compose(f, g), h)(x), compose(f, compose(g, h))(x), atol=1e-12)
        probes += 1
        q = make((3, 7, 4))
        p = parallel(f, q)
        np.testing.assert_allclose(p(np.concatenate((x, x), axis=1)), np.concatenate((f(x), q(x)), axis=1), atol=1e-12)
        probes += 1
        np.testing.assert_allclose(summed(f, q)(x), f(x) + q(x), atol=1e-12)
        probes += 1
        if activation == "relu":
            np.testing.assert_allclose(identity(3)(x), x, atol=1e-12)
            probes += 1
            np.testing.assert_allclose(extend(f, 5)(x), f(x), atol=1e-12)
            probes += 1
        try:
            check_accounting(p, p.parameters + 1, len(p.layers))
        except ValueError:
            probes += 1
        else:
            raise AssertionError("parameter-count mutation was accepted")
        corrupted = compose(f, g)
        w, b = corrupted.layers[0]
        corrupted.layers = ((w, b + 1),) + corrupted.layers[1:]
        assert not np.allclose(corrupted(x), g(f(x)))
        probes += 1
    try:
        Network(((np.eye(2), np.zeros(2)),), "attention")
    except ValueError:
        probes += 1
    else:
        raise AssertionError("out-of-fragment activation was accepted")
    print(f"ANN design spike: {probes} checks passed (reference only)")
    return probes


if __name__ == "__main__":
    run()
