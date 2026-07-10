"""Apple GPU 0-move + sort lane — pad / roll / flip / tile / repeat / stack +
sort / argsort.

Apple ships no device gather/sort kernel, so the layout ops (host index-map +
numpy gather) and sort/argsort (numpy stable sort) run on the CPU reference the
x86/ROCm device kernels are matched against. Reachable via
`compiler_path="apple_gpu_shape_compiled"`; execution_kind=reference_cpu.
Validated vs numpy — parity with test_x86_strided_compiled / test_x86_sort_compiled.
"""

from __future__ import annotations

import numpy as np

from tessera import runtime as rt

X = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
Y = np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32)


def _run(op, kwargs, *arrs):
    names = [f"a{i}" for i in range(len(arrs))]
    art = rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_shape_compiled",
        "executable": True, "execution_kind": "reference_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": dict(kwargs)}]})
    res = rt.launch(art, tuple(arrs))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_shape_compiled"
    assert res["execution_kind"] == "reference_cpu"
    return np.asarray(res["output"])


def test_roll():
    np.testing.assert_array_equal(_run("tessera.roll", {"shift": 2, "axis": 1}, X),
                                  np.roll(X, 2, axis=1))
    np.testing.assert_array_equal(_run("tessera.roll", {"shift": -3}, X),
                                  np.roll(X, -3))


def test_flip():
    np.testing.assert_array_equal(_run("tessera.flip", {"axis": 0}, X), np.flip(X, 0))
    np.testing.assert_array_equal(_run("tessera.flip", {"axis": 1}, X), np.flip(X, 1))


def test_tile_and_repeat():
    np.testing.assert_array_equal(_run("tessera.tile", {"reps": [2, 3]}, X),
                                  np.tile(X, [2, 3]))
    np.testing.assert_array_equal(_run("tessera.repeat", {"repeats": 2, "axis": 1}, X),
                                  np.repeat(X, 2, axis=1))


def test_pad():
    np.testing.assert_array_equal(
        _run("tessera.pad", {"pad_width": ((1, 2), (0, 3)), "mode": "constant",
                             "constant_values": 0.0}, X),
        np.pad(X, ((1, 2), (0, 3)), mode="constant"))
    np.testing.assert_array_equal(
        _run("tessera.pad", {"pad_width": ((1, 1), (1, 1)), "mode": "edge"}, X),
        np.pad(X, ((1, 1), (1, 1)), mode="edge"))


def test_stack():
    a = np.random.default_rng(2).standard_normal((3, 4)).astype(np.float32)
    np.testing.assert_array_equal(_run("tessera.stack", {"axis": 0}, X, a),
                                  np.stack([X, a], axis=0))
    np.testing.assert_array_equal(_run("tessera.stack", {"axis": 1}, X, a),
                                  np.stack([X, a], axis=1))


def test_sort():
    np.testing.assert_allclose(_run("tessera.sort", {"axis": 1}, X),
                               np.sort(X, axis=1), atol=0)
    np.testing.assert_allclose(_run("tessera.sort", {"axis": 0}, X),
                               np.sort(X, axis=0), atol=0)
    np.testing.assert_allclose(_run("tessera.sort", {"axis": 1, "descending": True}, X),
                               np.sort(X, axis=1)[:, ::-1], atol=0)


def test_argsort():
    np.testing.assert_array_equal(_run("tessera.argsort", {"axis": 1}, X),
                                  np.argsort(X, axis=1, kind="stable"))
    np.testing.assert_array_equal(_run("tessera.argsort", {"axis": 0}, X),
                                  np.argsort(X, axis=0, kind="stable"))
