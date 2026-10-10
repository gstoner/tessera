"""HIP graph replay of checked resident packages; includes GPU graph dispatch.

Graphs borrow package buffers/modules and the clock marker. Close this owner
before closing either dependency. Capture/instantiation are outside timing.
"""
from __future__ import annotations
import ctypes as C
import time
from tessera.compiler.profiler_timing import wall_clock_ticks_to_ns

class FoldedGraphWindows:
    def __init__(self, clock):
        self.clock, self.hip = clock, clock.hip
        self.stream = C.c_void_p()
        self.graphs = {}
        signatures = {
            "hipStreamCreateWithFlags": [C.POINTER(C.c_void_p), C.c_uint],
            "hipStreamDestroy": [C.c_void_p],
            "hipStreamSynchronize": [C.c_void_p],
            "hipStreamBeginCapture": [C.c_void_p, C.c_int],
            "hipStreamEndCapture": [C.c_void_p, C.POINTER(C.c_void_p)],
            "hipGraphGetNodes": [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)],
            "hipGraphInstantiateWithFlags": [C.POINTER(C.c_void_p), C.c_void_p, C.c_ulonglong],
            "hipGraphLaunch": [C.c_void_p, C.c_void_p],
            "hipGraphExecDestroy": [C.c_void_p],
            "hipGraphDestroy": [C.c_void_p],
        }
        for name, args in signatures.items():
            function = getattr(self.hip, name)
            function.argtypes, function.restype = args, C.c_int
        self._check(self.hip.hipStreamCreateWithFlags(C.byref(self.stream), 1), "stream creation")

    def _check(self, status, action):
        if status:
            raise RuntimeError(f"HIP graph {action} failed rc={status}")

    def _capture(self, engine, launches, bracketed):
        key = (id(engine), launches, bracketed)
        if key in self.graphs:
            return self.graphs[key]
        if launches < len(engine.copies) or launches % len(engine.copies):
            raise ValueError("graph must cover an integral rotation of resident copies")
        graph, executable = C.c_void_p(), C.c_void_p()
        capturing = False
        start = time.perf_counter_ns()
        self._check(self.hip.hipDeviceSynchronize(), "pre-capture completion")
        engine._next = 0
        try:
            self._check(self.hip.hipStreamBeginCapture(self.stream, 0), "begin capture")
            capturing = True
            if bracketed:
                self.clock._marker(self.stream)
            for _ in range(launches):
                engine.launch_on_stream(self.stream)
            if bracketed:
                self.clock._marker(self.stream)
            status = self.hip.hipStreamEndCapture(self.stream, C.byref(graph))
            capturing = False
            self._check(status, "end capture")
            nodes = C.c_size_t()
            self._check(self.hip.hipGraphGetNodes(graph, None, C.byref(nodes)), "node census")
            expected = launches + (2 if bracketed else 0)
            if nodes.value != expected:
                raise RuntimeError(f"captured {nodes.value} nodes, expected {expected}")
            self._check(self.hip.hipGraphInstantiateWithFlags(C.byref(executable), graph, 0),
                        "instantiate")
            record = (graph, executable, nodes.value, (time.perf_counter_ns()-start)/1e6)
            self.graphs[key] = record
            return record
        except BaseException:
            if capturing:
                self.hip.hipStreamEndCapture(self.stream, C.byref(graph))
            if executable.value:
                self.hip.hipGraphExecDestroy(executable)
            if graph.value:
                self.hip.hipGraphDestroy(graph)
            raise

    def window(self, engine, launches, *, bracketed):
        graph, executable, nodes, build_ms = self._capture(engine, launches, bracketed)
        clock, hip = self.clock, self.hip
        clock.host_span[0], clock.host_span[1] = (1 << 64)-1, 0
        host = C.c_void_p(C.addressof(clock.host_span))
        self._check(hip.hipMemcpy(clock.span, host, 16, 1), "span reset")
        self._check(hip.hipDeviceSynchronize(), "pre-replay completion")
        start = time.perf_counter_ns()
        self._check(hip.hipEventRecord(clock.events[0], self.stream), "start event")
        self._check(hip.hipGraphLaunch(executable, self.stream), "replay")
        self._check(hip.hipEventRecord(clock.events[1], self.stream), "stop event")
        self._check(hip.hipEventSynchronize(clock.events[1]), "replay completion")
        host_ms = (time.perf_counter_ns()-start)/1e6
        elapsed = C.c_float()
        self._check(hip.hipEventElapsedTime(C.byref(elapsed), *clock.events), "event elapsed")
        sample = dict(launches=launches, bracketed=bracketed,
                      dispatch_mode="hip_graph_replay", host_graph_launches=1,
                      graph_node_count=nodes, graph_build_wall_ms=build_ms,
                      event_window_ms=float(elapsed.value), host_window_ms=host_ms)
        if bracketed:
            self._check(hip.hipMemcpy(host, clock.span, 16, 2), "span read")
            begin, end = map(int, clock.host_span)
            if begin == (1 << 64)-1 or end <= begin or elapsed.value <= 0:
                raise RuntimeError("graph replay clock marker was not written")
            device_ms = wall_clock_ticks_to_ns(end-begin, clock.rate_khz)/1e6
            sample.update(device_window_ms=device_ms,
                          device_event_disagreement=abs(device_ms-elapsed.value)/elapsed.value)
        return sample

    def close(self):
        if self.stream.value:
            self._check(self.hip.hipStreamSynchronize(self.stream), "close completion")
        for graph, executable, _, _ in self.graphs.values():
            self._check(self.hip.hipGraphExecDestroy(executable), "destroy executable")
            self._check(self.hip.hipGraphDestroy(graph), "destroy graph")
        self.graphs.clear()
        if self.stream.value:
            self._check(self.hip.hipStreamDestroy(self.stream), "destroy stream")
            self.stream.value = None
