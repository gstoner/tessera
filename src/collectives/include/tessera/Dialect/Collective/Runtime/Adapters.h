#pragma once
//===- Adapters.h - NCCL/RCCL collective adapter interface ----------------===//
//
// CollectiveAdapter: common interface implemented by NCCLAdapter (NVIDIA) and
// RCCLAdapter (AMD ROCm).  Both adapters expose the four canonical collectives
// (all_reduce, reduce_scatter, all_gather, all_to_all) plus a low-level
// chunk-async path used by the CollectiveScheduler.
//
// When built without NCCL/RCCL (CPU-only mode, tests), both adapters fall
// back to the in-memory mock path: all operations complete synchronously on
// the calling thread using a shared memory model.
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tessera {
namespace collective {

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

using Callback = std::function<void()>;

enum class ReduceOp { SUM, MAX, MIN, PROD };

enum class WireDType { FP32, BF16, FP16, FP8, INT8 };

struct CollectiveSpec {
  int    world_size   = 1;
  int    local_rank   = 0;
  size_t chunk_bytes  = 512 * 1024;  // 512 KiB default (NVLink bandwidth sweet spot)
  WireDType wire_dtype = WireDType::BF16;
};

// ─────────────────────────────────────────────────────────────────────────────
// CollectiveAdapter — base interface
// ─────────────────────────────────────────────────────────────────────────────

struct CollectiveAdapter {
  virtual ~CollectiveAdapter() = default;

  /// Returns true if the underlying library (NCCL/RCCL) is available.
  virtual bool enabled() const = 0;

  /// All-reduce: reduce across all ranks; every rank gets the result.
  /// buf must be pre-allocated with count * sizeof(element).
  virtual void all_reduce(void* buf, size_t count, ReduceOp op,
                          const CollectiveSpec& spec) = 0;

  /// Reduce-scatter: reduce then scatter one shard per rank.
  /// src: full tensor (count elements)
  /// dst: shard for this rank (count / world_size elements)
  virtual void reduce_scatter(const void* src, void* dst, size_t count,
                              ReduceOp op, const CollectiveSpec& spec) = 0;

  /// All-gather: each rank contributes src (count/world_size elements);
  /// dst receives the full concatenated tensor (count elements).
  virtual void all_gather(const void* src, void* dst, size_t count,
                          const CollectiveSpec& spec) = 0;

  /// All-to-all: each rank sends a distinct shard to every other rank.
  /// send_buf: world_size consecutive slices of shard_bytes each
  /// recv_buf: world_size consecutive slots of shard_bytes each
  virtual void all_to_all(const void* send_buf, void* recv_buf,
                           size_t shard_bytes, const CollectiveSpec& spec) = 0;

  /// Low-level async chunk submission (used by CollectiveScheduler).
  /// On completion, cb() is invoked on an unspecified thread.
  virtual void submitChunkAsync(const void* buf, size_t bytes,
                                int device, int stream,
                                Callback cb) = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// NCCLAdapter — NVIDIA NCCL backend
// ─────────────────────────────────────────────────────────────────────────────

struct NCCLAdapter : CollectiveAdapter {
  /// Returns true when built with NCCL (TESSERA_ENABLE_CUDA=ON and nccl found).
  bool enabled() const override {
#ifdef TESSERA_HAS_NCCL
    return true;
#else
    return false;
#endif
  }

  void all_reduce(void* buf, size_t count, ReduceOp op,
                  const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_NCCL
    // ncclAllReduce(buf, buf, count, ncclFloat32, toNcclOp(op), comm_, stream_);
    // cudaStreamSynchronize(stream_);
    (void)buf; (void)count; (void)op; (void)spec;
#else
    _mock_all_reduce(buf, count, op, spec);
#endif
  }

  void reduce_scatter(const void* src, void* dst, size_t count,
                      ReduceOp op, const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_NCCL
    // ncclReduceScatter(src, dst, count / spec.world_size, ncclFloat32,
    //                  toNcclOp(op), comm_, stream_);
    // cudaStreamSynchronize(stream_);
    (void)src; (void)dst; (void)count; (void)op; (void)spec;
#else
    _mock_reduce_scatter(src, dst, count, op, spec);
#endif
  }

  void all_gather(const void* src, void* dst, size_t count,
                  const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_NCCL
    // ncclAllGather(src, dst, count / spec.world_size, ncclFloat32, comm_, stream_);
    // cudaStreamSynchronize(stream_);
    (void)src; (void)dst; (void)count; (void)spec;
#else
    _mock_all_gather(src, dst, count, spec);
#endif
  }

  void all_to_all(const void* send_buf, void* recv_buf,
                   size_t shard_bytes, const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_NCCL
    // NCCL 2.7+: ncclGroupStart / ncclSend / ncclRecv / ncclGroupEnd
    (void)send_buf; (void)recv_buf; (void)shard_bytes; (void)spec;
#else
    _mock_all_to_all(send_buf, recv_buf, shard_bytes, spec);
#endif
  }

  void submitChunkAsync(const void* buf, size_t bytes, int device, int stream,
                        Callback cb) override {
    (void)buf; (void)bytes; (void)device; (void)stream;
    if (cb) cb();  // mock: complete immediately
  }

private:
  // ── In-memory mock implementations (CPU, for testing) ────────────────────

  static void _mock_all_reduce(void* buf, size_t count, ReduceOp op,
                                const CollectiveSpec& spec) {
    // Single-process mock: no-op (already "reduced" on one process)
    (void)buf; (void)count; (void)op; (void)spec;
  }

  static void _mock_reduce_scatter(const void* src, void* dst, size_t count,
                                    ReduceOp op, const CollectiveSpec& spec) {
    // Copy this rank's shard from src to dst
    size_t shard = count / (size_t)spec.world_size;
    size_t offset = (size_t)spec.local_rank * shard;
    std::memcpy(dst, static_cast<const char*>(src) + offset * sizeof(float),
                shard * sizeof(float));
    (void)op;
  }

  static void _mock_all_gather(const void* src, void* dst, size_t count,
                                const CollectiveSpec& spec) {
    // Copy src into rank's slot in dst
    size_t shard = count / (size_t)spec.world_size;
    size_t offset = (size_t)spec.local_rank * shard;
    std::memcpy(static_cast<char*>(dst) + offset * sizeof(float),
                src, shard * sizeof(float));
  }

  static void _mock_all_to_all(const void* send_buf, void* recv_buf,
                                size_t shard_bytes, const CollectiveSpec& spec) {
    // Single-process: copy rank's own shard from send to recv position
    size_t offset = (size_t)spec.local_rank * shard_bytes;
    std::memcpy(static_cast<char*>(recv_buf) + offset,
                static_cast<const char*>(send_buf) + offset, shard_bytes);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// RCCLAdapter — AMD ROCm RCCL backend
// ─────────────────────────────────────────────────────────────────────────────

struct RCCLAdapter : CollectiveAdapter {
  bool enabled() const override {
#ifdef TESSERA_HAS_RCCL
    return true;
#else
    return false;
#endif
  }

  void all_reduce(void* buf, size_t count, ReduceOp op,
                  const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_RCCL
    // rcclAllReduce(buf, buf, count, rcclFloat, toRcclOp(op), comm_, stream_);
    // hipStreamSynchronize(stream_);
    (void)buf; (void)count; (void)op; (void)spec;
#else
    (void)buf; (void)count; (void)op; (void)spec;
#endif
  }

  void reduce_scatter(const void* src, void* dst, size_t count,
                      ReduceOp op, const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_RCCL
    (void)src; (void)dst; (void)count; (void)op; (void)spec;
#else
    size_t shard = count / (size_t)spec.world_size;
    size_t offset = (size_t)spec.local_rank * shard;
    std::memcpy(dst, static_cast<const char*>(src) + offset * sizeof(float),
                shard * sizeof(float));
    (void)op;
#endif
  }

  void all_gather(const void* src, void* dst, size_t count,
                  const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_RCCL
    (void)src; (void)dst; (void)count; (void)spec;
#else
    size_t shard = count / (size_t)spec.world_size;
    size_t offset = (size_t)spec.local_rank * shard;
    std::memcpy(static_cast<char*>(dst) + offset * sizeof(float),
                src, shard * sizeof(float));
#endif
  }

  void all_to_all(const void* send_buf, void* recv_buf,
                   size_t shard_bytes, const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_RCCL
    (void)send_buf; (void)recv_buf; (void)shard_bytes; (void)spec;
#else
    size_t offset = (size_t)spec.local_rank * shard_bytes;
    std::memcpy(static_cast<char*>(recv_buf) + offset,
                static_cast<const char*>(send_buf) + offset, shard_bytes);
#endif
  }

  void submitChunkAsync(const void* buf, size_t bytes, int device, int stream,
                        Callback cb) override {
    (void)buf; (void)bytes; (void)device; (void)stream;
    if (cb) cb();
  }
};

}} // namespace tessera::collective
