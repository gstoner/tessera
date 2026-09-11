#!/usr/bin/env python3
"""CUDA/HIP stream-owned byte graphs and composed resident SSD adjoints."""

import argparse
import ctypes as ct
import hashlib
import json
import platform
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks.record_pool_resident_ssd import Memory  # noqa: E402
from benchmarks.record_device_ring_protocol import Device  # noqa: E402
from tessera.compiler.resident_object_pool import ResidentObjectPool  # noqa: E402
from tessera.compiler.resident_ssd import ResidentSSDProgram  # noqa: E402
from tessera.control import vjp
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
            empty_roots = memory.put(np.zeros(4, np.int64))
            pool.begin_collection(s0, marker_stream=s1)
            # Removing roots while the private snapshot is marked must retain
            # snapshot survivors for this cycle, then reclaim them next cycle.
            pool.set_graph(s2, empty_roots, edges)
            pool.finish_collection(s2).wait()
            with pool.read(s0) as borrowed:
                np.testing.assert_array_equal(memory.get(borrowed[0])[:, 2], [1, 1, 0, 0])
                assert memory.get(borrowed[-1])[1] == 2
            bad_seed = memory.put(np.array([0, 0, 1, 0], np.int64))
            seed_status = memory.put(np.zeros(3, np.int64))
            # A dead slot must never be traversed through an unchecked seed.
            pool._seeded.submit(s0, pool._state, pool._roots, pool._edges,
                                pool._marks, pool._status, bad_seed, seed_status, 0, 4, 1).ticket.wait()
            with pool.read(s0) as borrowed:
                assert memory.get(borrowed[-1])[0] == 2
                np.testing.assert_array_equal(memory.get(borrowed[0])[:, 2], [1, 1, 0, 0])
            for row in memory.get(readback):
                assert json.loads(row[: len(encoded)].tobytes()) == json.loads(encoded)
            memory.write(roots, np.zeros(4, np.int64))
            pool.set_graph(s0, roots, edges).wait()
            pool.collect(s1).wait()
            pool.allocate(s2, incoming, len(encoded)).wait()
            with pool.read(s0) as borrowed:
                np.testing.assert_array_equal(memory.get(borrowed[-1])[:3], [0, 0, 2])
            heap_digest = pool.collect_program.package.binding_digest

        class Record:
            pass

        root = Record()
        root.children = [root]
        with ResidentObjectPool.from_objects(root, stream=s0, **options) as discovered:
            discovered.collect(s1).wait()
            with discovered.read(s2) as records:
                np.testing.assert_array_equal(memory.get(records[0])[:, 2], [1, 1])
        rng = np.random.default_rng(745)
        values = [
            rng.uniform(-0.4, 0.4, s).astype(np.float32) for s in [(3, 2, 2), (3, 2), (3, 2, 2), (3, 2, 2), (2, 2, 2)]
        ]
        inputs = [memory.put(v) for v in values]
        seed = rng.uniform(-0.4, 0.4, (3, 2, 2)).astype(np.float32)
        dy = memory.put(seed)
        logical = lower_scheduled_ssd(3, 2, 2, 2, 2, compiler=args.compiler)
        with ResidentSSDProgram(logical, **options) as program:
            copies = [memory.put(np.zeros_like(v)) for v in values]
            program._forward._bound = program.forward.package.bind()

            def forbidden():
                raise AssertionError("asynchronous capture/AD synchronized the context")

            saved_sync = program._forward._bound._sync
            program._forward._bound._sync = forbidden
            value, pullback = vjp(program, *inputs, stream=s0)
            first, second = pullback.frame, program.capture_async(s1, *inputs)
            g1 = pullback(dy)
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
            first.retire_async(s2)
            second.retire_async(s0)
            first._retirement.wait()
            second._retirement.wait()
            assert first.poll_close() and second.poll_close()
            program._forward._bound._sync = saved_sync
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
                    host_kernel=platform.release(),
                    backend=args.backend,
                    chip=options["chip"],
                    execution="native_gpu",
                    compiler_sha256=hashlib.sha256(args.compiler.read_bytes()).hexdigest(),
                    heap_binding=heap_digest,
                    snapshot_marker_binding=pool._marker.package.binding_digest,
                    seeded_collector_binding=pool._seeded.package.binding_digest,
                    discovered_collector_binding=discovered.collect_program.package.binding_digest,
                    heap_allocation_binding=pool.allocate_program.package.binding_digest,
                    heap_graph_binding=pool.graph_program.package.binding_digest,
                    forward_binding=program.forward.package.binding_digest,
                    backward_binding=program.reverse.package.binding_digest,
                    byte_record_graph=True,
                    reader_ordered_collection=True,
                    snapshot_marking_with_mutation=True,
                    public_async_vjp=True,
                    whole_frame_async_retirement=True,
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
