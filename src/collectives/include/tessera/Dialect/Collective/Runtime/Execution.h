#pragma once
#include "tessera/Dialect/Collective/Runtime/Policy.h"
#include "tessera/Dialect/Collective/Runtime/TokenLimiter.h"
#include "tessera/Dialect/Collective/Runtime/PerfettoTrace.h"
#include "tessera/Dialect/Collective/Runtime/Adapters.h"
#include <memory>
#include <mutex>

namespace tessera { namespace collective {

struct ChunkDesc {
  const void* ptr;
  uint64_t bytes;
  int device = 0;
  int stream = 0;
  bool intraNode = true;
};

class ExecRuntime {
public:
  ExecRuntime(int maxInflight, Policy pol, int pidBase=0)
    : limiter_(maxInflight), policy_(pol), pidBase_(pidBase) {}

  void setNCCL(std::unique_ptr<NCCLAdapter> a) {
    std::lock_guard<std::mutex> g(mu_);
    nccl_ = std::shared_ptr<NCCLAdapter>(std::move(a));
  }
  void setRCCL(std::unique_ptr<RCCLAdapter> a) {
    std::lock_guard<std::mutex> g(mu_);
    rccl_ = std::shared_ptr<RCCLAdapter>(std::move(a));
  }
  PerfettoTraceWriter& trace() { return trace_; }

  // Core API: gated submission with policy + tracing
  void submit(const ChunkDesc& d,
              std::shared_ptr<ExecRuntime> keepAlive = nullptr) {
    auto algo = policy_.chooseAlgo(d.bytes);
    auto path = policy_.choosePath(d.intraNode, topo_);

    limiter_.acquire();
    int pid = pidBase_ + d.device;
    int tid = d.stream;
    trace_.begin(
      "CommChunk", "comm", pid, tid,
      {
        {"schema", "tessera.telemetry.v1"},
        {"source", "collective_runtime"},
        {"op", "collective"},
        {"kernel_id", "collective_chunk"},
        {"device", std::to_string(d.device)},
        {"stream", std::to_string(d.stream)},
        {"status", "ok"},
        {"path", (path==Path::NVLINK?"NVLINK": path==Path::RDMA?"RDMA":"PCIE")},
        {"algo", (algo==Algo::Ring?"ring": algo==Algo::Tree?"tree": algo==Algo::Hier?"hier":"auto")},
      },
      {
        {"memory_bytes", static_cast<double>(d.bytes)},
      });
    trace_.counter("chunk_bytes", static_cast<double>(d.bytes), pid, tid);
    trace_.annotate("path", (path==Path::NVLINK?"NVLINK": path==Path::RDMA?"RDMA":"PCIE"));

    // Snapshot strong adapter references so a concurrent replacement cannot
    // destroy an adapter while its asynchronous submission is in flight.
    std::shared_ptr<NCCLAdapter> nccl_snap;
    std::shared_ptr<RCCLAdapter> rccl_snap;
    {
      std::lock_guard<std::mutex> g(mu_);
      nccl_snap = nccl_;
      rccl_snap = rccl_;
    }

    // The C singleton path supplies ``keepAlive`` so shutdown cannot destroy
    // the limiter/trace state before the completion callback runs. Direct
    // stack-owned users retain the established contract: the caller keeps the
    // ExecRuntime alive until its adapter has completed every submission.
    auto done = [this, pid, tid, keepAlive = std::move(keepAlive),
                 nccl_snap, rccl_snap]() {
      trace_.end("CommChunk", "comm", pid, tid);
      limiter_.release();
    };

    // Prefer NCCL/RCCL if enabled; otherwise immediate completion.
    if (nccl_snap)      nccl_snap->submitChunkAsync(d.ptr, d.bytes, d.device, d.stream, done);
    else if (rccl_snap) rccl_snap->submitChunkAsync(d.ptr, d.bytes, d.device, d.stream, done);
    else                done();
  }

  // Exposed to C hooks
  void setMaxInflight(int n) { limiter_.set(n); }

private:
  TokenLimiter limiter_;
  Policy policy_;
  Topology topo_;
  PerfettoTraceWriter trace_;
  std::shared_ptr<NCCLAdapter> nccl_;
  std::shared_ptr<RCCLAdapter> rccl_;
  std::mutex mu_;
  int pidBase_;
};

// ---- C hooks to wire from lowered QoS ops ----
extern "C" {
  void tessera_qos_limit_set(int tokens);
  void tessera_qos_acquire();
  void tessera_qos_release();
  void tessera_submit_chunk_async(const void* ptr, uint64_t bytes, int device, int stream);
  void tessera_trace_write(const char* path);
  // Explicit teardown for embedded / notebook / reload-test lifecycles.
  // Safe to call when the runtime was never initialized (no-op).
  // Subsequent ``tessera_*`` calls re-initialize on demand.
  //
  // Concurrency: this entry point is safe to call while another
  // thread is inside ``tessera_submit_chunk_async`` — each asynchronous
  // completion callback holds a ``shared_ptr`` to the runtime so actual
  // destruction is deferred until the last callback completes. Callers
  // must not, however, rely on any specific runtime *identity*
  // surviving across a shutdown: the next submit may construct a
  // fresh runtime with a freshly-configured ``TokenLimiter``.
  void tessera_shutdown_runtime();
}

}} // ns
