#!/usr/bin/env python3
"""CUDA/HIP stream-owned byte graphs and composed resident SSD adjoints."""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks.record_pool_resident_ssd import Memory  # noqa: E402
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.resident_object_pool import ResidentObjectPool  # noqa: E402
from tessera.compiler.resident_ssd import ResidentSSDProgram  # noqa: E402
from tessera.compiler.scheduled_ssd import lower_scheduled_ssd  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["nvidia", "rocm"])
    p.add_argument("--compiler", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    d = Device(args.backend)
    P = ct.c_void_p

    def bind(cu, hip, types):
        fn = getattr(d.lib, cu if d.cuda else hip)
        fn.argtypes, fn.restype = types, ct.c_int
        return fn

    create = bind("cuStreamCreate", "hipStreamCreateWithFlags", [ct.POINTER(P), ct.c_uint])
    destroy = bind("cuStreamDestroy_v2", "hipStreamDestroy", [P])
    copy = bind("cuMemcpyDtoDAsync_v2", "hipMemcpyDtoDAsync", [P, P, ct.c_size_t, P])
    streams = [P(), P(), P()]
    for stream in streams:
        d.check(create(ct.byref(stream), 1))
    s0, s1, s2 = [s.value for s in streams]
    options = dict(
        compiler=args.compiler,
        llvm_bin=Path("/usr/lib/llvm-23/bin"),
        backend=args.backend,
        chip="sm_120" if d.cuda else "gfx1151",
    )
    memory = Memory(d)
    try:
        record = np.zeros(64, np.int8)
        encoded = json.dumps(dict(tag="record", value=[1, "hello", False])).encode()
        record[: len(encoded)] = np.frombuffer(encoded, np.int8)
        incoming = memory.put(record)
        readback = memory.put(np.zeros((4, 64), np.int8))
        with ResidentObjectPool(4, 64, 4, stream=s0, **options) as pool:
            for _ in range(4):
                pool.allocate(s0, incoming, len(encoded)).wait()
            roots = memory.put(np.array([1, 0, 0, 0], np.int64))
            links = np.tile(np.array([-1, 0] * 4, np.int64), (4, 1))
            links[0, 6:8] = [1, 1]
            links[1, 0:2] = [0, 1]
            edges = memory.put(links)
            pool.set_graph(s0, roots, edges).wait()
            with pool.read(s1) as borrowed:
                try:
                    pool.collect(s2)
                except ValueError:
                    pass
                else:
                    raise AssertionError("collection raced an active reader lease")
                d.check(
                    copy(
                        P(readback.__cuda_array_interface__["data"][0]),
                        P(borrowed[3].__cuda_array_interface__["data"][0]),
                        256,
                        P(s1),
                    )
                )
            pool.collect(s2).wait()
            with pool.read(s0) as borrowed:
                np.testing.assert_array_equal(memory.get(borrowed[0])[:, 2], [1, 1, 0, 0])
                assert memory.get(borrowed[-1])[1] == 2
            for row in memory.get(readback):
                assert json.loads(row[: len(encoded)].tobytes()) == json.loads(encoded)
            memory.write(roots, np.zeros(4, np.int64))
            pool.set_graph(s0, roots, edges).wait()
            pool.collect(s1).wait()
            pool.allocate(s2, incoming, len(encoded)).wait()
            with pool.read(s0) as borrowed:
                np.testing.assert_array_equal(memory.get(borrowed[-1])[:3], [0, 0, 2])
            heap_digest = pool.collect_program.package.binding_digest
        rng = np.random.default_rng(745)
        values = [
            rng.uniform(-0.4, 0.4, s).astype(np.float32) for s in [(3, 2, 2), (3, 2), (3, 2, 2), (3, 2, 2), (2, 2, 2)]
        ]
        inputs = [memory.put(v) for v in values]
        seed = rng.uniform(-0.4, 0.4, (3, 2, 2)).astype(np.float32)
        dy = memory.put(seed)
        logical = lower_scheduled_ssd(3, 2, 2, 2, 2, compiler=args.compiler)
        with ResidentSSDProgram(logical, **options) as program:
            # Capture before asynchronous composition; no context wait is
            # permitted from first backward through both reader submissions.
            first, second = program.capture(*inputs), program.capture(*inputs)

            def forbidden():
                raise AssertionError("asynchronous AD synchronized the context")

            sync_first, sync_second = first.sync, second.sync
            first.sync = second.sync = forbidden
            copies = [memory.put(np.zeros_like(v)) for v in values]
            g1 = first.backward_async(s0, dy)
            g2 = g1.backward_into(second, s1, indices=(0,))
            with g2.read(s2) as outputs:
                for destination, source, value in zip(copies, outputs, values, strict=True):
                    d.check(
                        copy(
                            P(destination.__cuda_array_interface__["data"][0]),
                            P(source.__cuda_array_interface__["data"][0]),
                            value.nbytes,
                            P(s2),
                        )
                    )
            g1.retire(s2)
            g2.retire(s0)
            g1.wait()
            g2.wait()
            first.sync, second.sync = sync_first, sync_second
            # Independent synchronous VJP composition is the numerical oracle.
            with program.capture(*inputs) as reference1:
                gradients = reference1.backward(dy)
                with program.capture(*inputs) as reference2:
                    expected = reference2.backward(gradients[0])
                    errors = []
                    for got, want in zip(copies, expected, strict=True):
                        a, b = memory.get(got), memory.get(want)
                        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-7)
                        errors.append(float(np.max(np.abs(a - b))))
            first.close()
            second.close()
        args.output.write_text(
            json.dumps(
                dict(
                    schema=1,
                    backend=args.backend,
                    chip=options["chip"],
                    execution="native_gpu",
                    compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                    heap_binding=heap_digest,
                    heap_allocation_binding=pool.allocate_program.package.binding_digest,
                    heap_graph_binding=pool.graph_program.package.binding_digest,
                    forward_binding=program.forward.package.binding_digest,
                    backward_binding=program.reverse.package.binding_digest,
                    byte_record_graph=True,
                    reader_ordered_collection=True,
                    async_composed_gradient_max_errors=errors,
                    promotion_eligible=False,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        d.check(d.sync())
        memory.close()
        for stream in streams:
            d.check(destroy(stream))


if __name__ == "__main__":
    import subprocess

    try:
        main()
    except subprocess.CalledProcessError as e:
        print(e.stderr, file=sys.stderr)
        raise
