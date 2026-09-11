#!/usr/bin/env python3
"""Owning-device proof for reusable heap slots and resident automatic SSD VJP."""

import argparse
import ctypes as ct
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.gpu_heap_collection import materialize_pool  # noqa: E402
from tessera.compiler.resident_ssd import ResidentSSDProgram  # noqa: E402
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd  # noqa: E402


class Memory:
    def __init__(self, device):
        self.device, self.pointers = device, []

    def put(self, value):
        p = ct.c_void_p()
        self.device.check(self.device.alloc(ct.byref(p), value.nbytes))
        self.pointers.append(p)
        view = SimpleNamespace(
            __cuda_array_interface__=dict(version=3, shape=value.shape, typestr=value.dtype.str, data=(p.value, False))
        )
        self.write(view, value)
        return view

    def write(self, view, value):
        p = ct.c_void_p(view.__cuda_array_interface__["data"][0])
        d = self.device
        d.check(d.htod(p, value.ctypes.data, value.nbytes) if d.cuda else d.copy(p, value.ctypes.data, value.nbytes, 1))

    def get(self, view):
        spec = view.__cuda_array_interface__
        value = np.empty(spec["shape"], dtype=spec["typestr"])
        d, p = self.device, ct.c_void_p(spec["data"][0])
        d.check(d.dtoh(value.ctypes.data, p, value.nbytes) if d.cuda else d.copy(value.ctypes.data, p, value.nbytes, 2))
        return value

    def close(self):
        for p in reversed(self.pointers):
            self.device.check(self.device.free(p))
        self.pointers.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, choices=["nvidia", "rocm"])
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    device = Device(args.backend)
    options = dict(
        compiler=args.compiler,
        llvm_bin=Path("/usr/lib/llvm-23/bin"),
        backend=args.backend,
        chip="sm_120" if device.cuda else "gfx1151",
    )
    allocation = materialize_pool(3, 4, "allocate", **options)
    collection = materialize_pool(3, 4, "collect", **options)
    memory = Memory(device)
    results = []
    try:
        state = memory.put(np.zeros((3, 3), np.int64))
        roots = memory.put(np.zeros(3, np.int64))
        edges = memory.put(np.tile(np.array([-1, 0, -1, 0], np.int64), (3, 1)))
        payload = memory.put(np.full((3, 4), -9, np.float32))
        incoming = memory.put(np.arange(4, dtype=np.float32))
        status = memory.put(np.full(3, -9, np.int64))
        marks = memory.put(np.full(3, -9, np.int64))
        with closing(allocation.bind()) as alloc, closing(collection.bind()) as collect:
            for i in range(3):
                alloc(state, roots, edges, payload, incoming, status, 4, 1)
                np.testing.assert_array_equal(memory.get(status), [0, i, 1])
            alloc(state, roots, edges, payload, incoming, status, 4, 1)
            assert memory.get(status)[0] == 1
            links = np.array([[1, 1, -1, 0], [0, 1, -1, 0], [-1, 0, -1, 0]], np.int64)
            memory.write(edges, links)
            memory.write(roots, np.array([1, 0, 0], np.int64))
            collect(state, roots, edges, marks, status, 1)
            assert memory.get(status)[1] == 1
            alloc(state, roots, edges, payload, incoming, status, 2, 1)
            np.testing.assert_array_equal(memory.get(status), [0, 2, 2])
            results.append("cycle reachability and partial payload-slot reuse")
            # An old generation must never keep a different occupant alive.
            links[0, 2:] = [2, 1]
            memory.write(edges, links)
            before = memory.get(state)
            collect(state, roots, edges, marks, status, 1)
            assert memory.get(status)[0] == 2
            np.testing.assert_array_equal(memory.get(state), before)
            results.append("stale edge refuses before sweep")
            links[0, 3] = 2
            memory.write(edges, links)
            memory.write(roots, np.zeros(3, np.int64))
            collect(state, roots, edges, marks, status, 1)
            assert memory.get(status)[1] == 3
            for iteration in range(8):
                alloc(state, roots, edges, payload, incoming, status, 4, 1)
                assert memory.get(status)[2] == iteration + 2
                memory.write(roots, np.zeros(3, np.int64))
                collect(state, roots, edges, marks, status, 1)
                assert memory.get(status)[1] == 1
            results.append("unrooted cycles and repeated generation reuse")
            malformed = memory.get(state)
            malformed[0] = [0, 4, 1]
            memory.write(state, malformed)
            collect(state, roots, edges, marks, status, 1)
            assert memory.get(status)[0] == 2
            np.testing.assert_array_equal(memory.get(state), malformed)
            results.append("invalid live generation refuses transactionally")
        rng = np.random.default_rng(743)
        shapes = [(3, 2, 2), (3, 2), (3, 2, 2), (3, 2, 2), (2, 2, 2)]
        values = [rng.uniform(-0.4, 0.4, s).astype(np.float32) for s in shapes]

        def forward(inputs):
            x, d, b, c, state = inputs
            state = state.copy()
            out = []
            for t in range(3):
                state = d[t, :, None, None] * state + b[t, :, :, None] * x[t, :, None, :]
                out.append((c[t, :, :, None] * state).sum(axis=1))
            return np.array(out)

        seed = rng.uniform(-0.4, 0.4, (3, 2, 2)).astype(np.float32)
        views = [memory.put(v) for v in values]
        dy = memory.put(seed)
        logical = lower_scheduled_ssd(3, 2, 2, 2, 2, compiler=args.compiler)
        errors = []
        with ResidentSSDProgram(logical, **options) as program:
            views[0]._tessera_reader_stream = 1
            try:
                program.capture(*views)
            except ValueError as error:
                assert "stream ownership" in str(error)
            else:
                raise AssertionError("borrowed reader accepted by synchronous capture")
            del views[0]._tessera_reader_stream
            with program.capture(*views) as frame:
                # Capture owns a device snapshot; changing caller storage must
                # not affect subsequent differentiation.
                memory.write(views[0], np.zeros_like(values[0]))
                grads = frame.backward(dy)
                np.testing.assert_allclose(memory.get(frame.value), forward(values), rtol=1e-5, atol=1e-6)
                for i, value in enumerate(values):
                    numerical = np.empty_like(value)
                    for index in np.ndindex(value.shape):
                        plus, minus = [v.astype(np.float64) for v in values], [v.astype(np.float64) for v in values]
                        plus[i][index] += 1e-5
                        minus[i][index] -= 1e-5
                        numerical[index] = np.sum((forward(plus) - forward(minus)) * seed) / 2e-5
                    got = memory.get(grads[i])
                    np.testing.assert_allclose(got, numerical, rtol=2e-5, atol=2e-6)
                    errors.append(float(np.max(np.abs(got - numerical))))
                held = frame.value
            try:
                held.__cuda_array_interface__
            except ValueError:
                pass
            else:
                raise AssertionError("closed resident value exposed")
            memory.write(views[0], values[0])
            with program.value_and_grad(*views, cotangent=dy) as automatic:
                np.testing.assert_allclose(memory.get(automatic.value), forward(values), rtol=1e-5, atol=1e-6)
                assert len(automatic.gradients) == 5
            assert not program.frames
        args.output.write_text(
            json.dumps(
                dict(
                    schema=1,
                    backend=args.backend,
                    chip=options["chip"],
                    execution="native_gpu",
                    compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                    allocation_binding=allocation.package.binding_digest,
                    collection_binding=collection.package.binding_digest,
                    forward_binding=program.forward.package.binding_digest,
                    backward_binding=program.reverse.package.binding_digest,
                    heap_checks=results,
                    resident_gradient_max_abs_errors=errors,
                    promotion_eligible=False,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        memory.close()


if __name__ == "__main__":
    import subprocess

    try:
        main()
    except subprocess.CalledProcessError as error:
        print(error.stderr, file=sys.stderr)
        raise
