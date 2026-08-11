#pragma once
//===- Adapters.h - NCCL/RCCL collective adapter interface ----------------===//
//
// CollectiveAdapter: common interface implemented by NCCLAdapter (NVIDIA) and
// RCCLAdapter (AMD ROCm).  Both adapters expose the four canonical collectives
// (all_reduce, reduce_scatter, all_gather, all_to_all) plus a low-level
// chunk-async path driven by ExecRuntime::submit (Execution.h). A richer
// overlap scheduler on top of that path is unimplemented — see the draft in
// src/collectives/docs/Tessera_Collectives_Overlap_Design.md §4.
//
// When built without NCCL/RCCL (CPU-only mode, tests), both adapters fall
// back to the in-memory mock path: all operations complete synchronously on
// the calling thread using a shared memory model.
//
//===----------------------------------------------------------------------===//

#include <functional>
#if defined(TESSERA_HAS_NCCL) || defined(TESSERA_HAS_RCCL)
#include <dlfcn.h>
#endif
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tessera/Dialect/Collective/Runtime/AdapterVersionPin.h"

#if defined(TESSERA_HAS_NCCL) && defined(TESSERA_HAS_RCCL)
#error "A collective runtime target must select exactly one NCCL-compatible ABI"
#endif

namespace tessera {
namespace collective {

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

using Callback = std::function<void()>;

enum class ReduceOp { SUM, MAX, MIN, PROD };

enum class WireDType { FP32, BF16, FP16, FP8, INT8 };

// Element size for the wire dtype. The mock collective paths operate on the
// wire layout, so hardcoding sizeof(float) gave wrong offsets/sizes for
// non-FP32 mock tests (Decision #6: mock paths must be correct).
inline size_t wireDTypeBytes(WireDType dt) {
  switch (dt) {
    case WireDType::FP32: return 4;
    case WireDType::BF16: return 2;
    case WireDType::FP16: return 2;
    case WireDType::FP8:  return 1;
    case WireDType::INT8: return 1;
  }
  return 4;
}

struct CollectiveSpec {
  int    world_size   = 1;
  int    local_rank   = 0;
  size_t chunk_bytes  = 512 * 1024;  // 512 KiB default (NVLink bandwidth sweet spot)
  WireDType wire_dtype = WireDType::BF16;
};

struct CommunicatorProperties {
  bool query_available = false;
  int rank = -1;
  int world_size = 0;
  int device = -1;
  bool device_api_support = false;
  bool multimem_support = false;
  int gin_type = 0;
  int lsa_team_count = 0;
  bool host_rma_support = false;
  int railed_gin_type = 0;
};

#if defined(TESSERA_NCCL_DEVICE_CORE_HEADER) || \
    defined(TESSERA_NCCL_DEVICE_CORE_TMP_HEADER)
// Host-only ABI mirror. Including nccl_device/core*.h in an ordinary C++
// translation unit pulls AMD device builtins into the host compiler. RCCL's
// query contract is size/version initialized, so this exact 2.30 layout can be
// passed through a dynamically discovered C symbol without compiling device
// implementation headers into the runtime library.
struct NativeCommPropertiesV23004 {
  size_t size;
  unsigned int magic;
  unsigned int version;
  int rank;
  int nRanks;
  int cudaDev;
  int nvmlDev;
  bool deviceApiSupport;
  bool multimemSupport;
  int ginType;
  int nLsaTeams;
  bool hostRmaSupport;
  int railedGinType;
};

inline CommunicatorProperties queryCommunicatorProperties(ncclComm_t comm,
                                                           const char *backend) {
  using QueryFn = ncclResult_t (*)(ncclComm_t, NativeCommPropertiesV23004 *);
  auto query = reinterpret_cast<QueryFn>(
      dlsym(RTLD_DEFAULT, "ncclCommQueryProperties"));
  if (!query)
    throw std::runtime_error(std::string(backend) +
                             " communicator property query symbol is unavailable");
  NativeCommPropertiesV23004 native{};
  native.size = sizeof(native);
  native.magic = 0xcafebeefU;
  native.version = NCCL_VERSION_CODE;
  const ncclResult_t status = query(comm, &native);
  if (status != ncclSuccess)
    throw std::runtime_error(std::string(backend) +
                             " communicator property query: " +
                             ncclGetErrorString(status));
  CommunicatorProperties result;
  result.query_available = true;
  result.rank = native.rank;
  result.world_size = native.nRanks;
  result.device = native.cudaDev;
  result.device_api_support = native.deviceApiSupport;
  result.multimem_support = native.multimemSupport;
  result.gin_type = native.ginType;
  result.lsa_team_count = native.nLsaTeams;
  result.host_rma_support = native.hostRmaSupport;
  result.railed_gin_type = native.railedGinType;
  return result;
}
#endif

#if (defined(TESSERA_HAS_NCCL) || defined(TESSERA_HAS_RCCL)) && \
    defined(NCCL_VERSION_CODE) && NCCL_VERSION_CODE >= 22800
/// Owns a collective window registration, not the underlying allocation.
/// Explicit close reports deregistration errors; destruction is best effort.
class NativeRegisteredWindow {
public:
  NativeRegisteredWindow() = default;
  NativeRegisteredWindow(ncclComm_t comm, ncclWindow_t window, void *buffer,
                         size_t bytes, int flags, const char *backend)
      : comm_(comm), window_(window), buffer_(buffer), bytes_(bytes),
        flags_(flags), backend_(backend) {}
  NativeRegisteredWindow(const NativeRegisteredWindow &) = delete;
  NativeRegisteredWindow &operator=(const NativeRegisteredWindow &) = delete;
  NativeRegisteredWindow(NativeRegisteredWindow &&other) noexcept {
    *this = std::move(other);
  }
  NativeRegisteredWindow &operator=(NativeRegisteredWindow &&other) noexcept {
    if (this == &other) return *this;
    closeNoThrow();
    comm_ = other.comm_;
    window_ = other.window_;
    buffer_ = other.buffer_;
    bytes_ = other.bytes_;
    flags_ = other.flags_;
    backend_ = other.backend_;
    other.comm_ = nullptr;
    other.window_ = nullptr;
    other.buffer_ = nullptr;
    other.bytes_ = 0;
    other.flags_ = 0;
    return *this;
  }
  ~NativeRegisteredWindow() { closeNoThrow(); }

  bool valid() const { return comm_ != nullptr && window_ != nullptr; }
  void *buffer() const { return buffer_; }
  size_t bytes() const { return bytes_; }
  int flags() const { return flags_; }
  ncclWindow_t nativeHandle() const { return window_; }

  void close() {
    if (!valid()) return;
    const ncclResult_t result = ncclCommWindowDeregister(comm_, window_);
    if (result != ncclSuccess)
      throw std::runtime_error(std::string(backend_) +
                               " window deregistration: " +
                               ncclGetErrorString(result));
    clear();
  }

private:
  void clear() noexcept {
    comm_ = nullptr;
    window_ = nullptr;
    buffer_ = nullptr;
    bytes_ = 0;
    flags_ = 0;
  }
  void closeNoThrow() noexcept {
    if (valid()) (void)ncclCommWindowDeregister(comm_, window_);
    clear();
  }

  ncclComm_t comm_ = nullptr;
  ncclWindow_t window_ = nullptr;
  void *buffer_ = nullptr;
  size_t bytes_ = 0;
  int flags_ = 0;
  const char *backend_ = "NCCL-compatible";
};
#endif

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

  /// Low-level async chunk submission (called by ExecRuntime::submit,
  /// Execution.h).
  /// On completion, cb() is invoked on an unspecified thread.
  virtual void submitChunkAsync(const void* buf, size_t bytes,
                                int device, int stream,
                                Callback cb) = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// NCCLAdapter — NVIDIA NCCL backend
// ─────────────────────────────────────────────────────────────────────────────

struct NCCLAdapter : CollectiveAdapter {
#ifdef TESSERA_HAS_NCCL
  NCCLAdapter(ncclComm_t comm, cudaStream_t stream)
      : comm_(comm), stream_(stream) {}
#endif
  NCCLAdapter() = default;

  CommunicatorProperties communicatorProperties() const {
    CommunicatorProperties result;
#if defined(TESSERA_HAS_NCCL) && \
    (defined(TESSERA_NCCL_DEVICE_CORE_HEADER) || \
     defined(TESSERA_NCCL_DEVICE_CORE_TMP_HEADER))
    requireReady();
    result = queryCommunicatorProperties(comm_, "NCCL");
#endif
    return result;
  }

#if defined(TESSERA_HAS_NCCL) && NCCL_VERSION_CODE >= 22800
  NativeRegisteredWindow registerSymmetricWindow(void *buffer, size_t bytes,
                                                  bool strictOrdering) const {
    return registerWindow(buffer, bytes, strictOrdering, "NCCL");
  }
#endif

  /// Returns true when built with NCCL (TESSERA_ENABLE_CUDA=ON and nccl found).
  bool enabled() const override {
#ifdef TESSERA_HAS_NCCL
    return comm_ != nullptr && stream_ != nullptr;
#else
    return false;
#endif
  }

  void all_reduce(void* buf, size_t count, ReduceOp op,
                  const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_NCCL
    requireReady();
    checkNccl(ncclAllReduce(buf, buf, count, toNcclDType(spec.wire_dtype),
                            toNcclOp(op), comm_, stream_),
              "all_reduce");
    synchronize();
#else
    _mock_all_reduce(buf, count, op, spec);
#endif
  }

  void reduce_scatter(const void* src, void* dst, size_t count,
                      ReduceOp op, const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_NCCL
    requireDivisible(count, spec.world_size, "reduce_scatter");
    requireReady();
    checkNccl(ncclReduceScatter(src, dst, count / spec.world_size,
                                toNcclDType(spec.wire_dtype), toNcclOp(op),
                                comm_, stream_),
              "reduce_scatter");
    synchronize();
#else
    _mock_reduce_scatter(src, dst, count, op, spec);
#endif
  }

  void all_gather(const void* src, void* dst, size_t count,
                  const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_NCCL
    requireDivisible(count, spec.world_size, "all_gather");
    requireReady();
    checkNccl(ncclAllGather(src, dst, count / spec.world_size,
                            toNcclDType(spec.wire_dtype), comm_, stream_),
              "all_gather");
    synchronize();
#else
    _mock_all_gather(src, dst, count, spec);
#endif
  }

  void all_to_all(const void* send_buf, void* recv_buf,
                   size_t shard_bytes, const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_NCCL
    requireReady();
    const size_t elementBytes = wireDTypeBytes(spec.wire_dtype);
    if (shard_bytes % elementBytes != 0)
      throw std::invalid_argument("all_to_all shard bytes must align to dtype");
    const size_t count = shard_bytes / elementBytes;
    const auto dtype = toNcclDType(spec.wire_dtype);
    checkNccl(ncclGroupStart(), "all_to_all group start");
    for (int peer = 0; peer < spec.world_size; ++peer) {
      const auto *send = static_cast<const char *>(send_buf) + peer * shard_bytes;
      auto *recv = static_cast<char *>(recv_buf) + peer * shard_bytes;
      checkNccl(ncclSend(send, count, dtype, peer, comm_, stream_),
                "all_to_all send");
      checkNccl(ncclRecv(recv, count, dtype, peer, comm_, stream_),
                "all_to_all receive");
    }
    checkNccl(ncclGroupEnd(), "all_to_all group end");
    synchronize();
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
#ifdef TESSERA_HAS_NCCL
#if NCCL_VERSION_CODE >= 22800
  NativeRegisteredWindow registerWindow(void *buffer, size_t bytes,
                                        bool strictOrdering,
                                        const char *backend) const {
    requireReady();
    validateWindow(buffer, bytes);
    const int flags = NCCL_WIN_COLL_SYMMETRIC |
                      (strictOrdering ? NCCL_WIN_STRICT_ORDERING : 0);
    ncclWindow_t window = nullptr;
    checkNccl(ncclCommWindowRegister(comm_, buffer, bytes, &window, flags),
              "symmetric window registration");
    return NativeRegisteredWindow(comm_, window, buffer, bytes, flags, backend);
  }

  static void validateWindow(void *buffer, size_t bytes) {
    if (!buffer || bytes == 0)
      throw std::invalid_argument("collective window requires a non-empty buffer");
  }
#endif
  static ncclDataType_t toNcclDType(WireDType dtype) {
    switch (dtype) {
      case WireDType::FP32: return ncclFloat32;
      case WireDType::BF16: return ncclBfloat16;
      case WireDType::FP16: return ncclFloat16;
      case WireDType::INT8: return ncclInt8;
      case WireDType::FP8:
        throw std::invalid_argument(
            "FP8 collective storage must distinguish e4m3 from e5m2");
    }
    throw std::invalid_argument("unsupported NCCL wire dtype");
  }

  static ncclRedOp_t toNcclOp(ReduceOp op) {
    switch (op) {
      case ReduceOp::SUM: return ncclSum;
      case ReduceOp::MAX: return ncclMax;
      case ReduceOp::MIN: return ncclMin;
      case ReduceOp::PROD: return ncclProd;
    }
    throw std::invalid_argument("unsupported NCCL reduction");
  }

  static void checkNccl(ncclResult_t result, const char *where) {
    if (result != ncclSuccess)
      throw std::runtime_error(std::string("NCCL ") + where + ": " +
                               ncclGetErrorString(result));
  }

  static void requireDivisible(size_t count, int worldSize,
                               const char *where) {
    if (worldSize < 1 || count % static_cast<size_t>(worldSize) != 0)
      throw std::invalid_argument(std::string(where) +
                                  " count must divide world size");
  }

  void requireReady() const {
    if (!enabled())
      throw std::runtime_error("NCCL adapter requires a communicator and stream");
  }

  void synchronize() const {
    const cudaError_t result = cudaStreamSynchronize(stream_);
    if (result != cudaSuccess)
      throw std::runtime_error(std::string("CUDA stream synchronization: ") +
                               cudaGetErrorString(result));
  }

  ncclComm_t comm_ = nullptr;
  cudaStream_t stream_ = nullptr;
#endif

  // ── In-memory mock implementations (CPU, for testing) ────────────────────

  static void _mock_all_reduce(void* buf, size_t count, ReduceOp op,
                                const CollectiveSpec& spec) {
    // Single-process mock: no-op (already "reduced" on one process)
    (void)buf; (void)count; (void)op; (void)spec;
  }

  static void _mock_reduce_scatter(const void* src, void* dst, size_t count,
                                    ReduceOp op, const CollectiveSpec& spec) {
    // Copy this rank's shard from src to dst
    size_t esz = wireDTypeBytes(spec.wire_dtype);
    size_t shard = count / (size_t)spec.world_size;
    size_t offset = (size_t)spec.local_rank * shard;
    std::memcpy(dst, static_cast<const char*>(src) + offset * esz,
                shard * esz);
    (void)op;
  }

  static void _mock_all_gather(const void* src, void* dst, size_t count,
                                const CollectiveSpec& spec) {
    // Copy src into rank's slot in dst
    size_t esz = wireDTypeBytes(spec.wire_dtype);
    size_t shard = count / (size_t)spec.world_size;
    size_t offset = (size_t)spec.local_rank * shard;
    std::memcpy(static_cast<char*>(dst) + offset * esz,
                src, shard * esz);
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
#ifdef TESSERA_HAS_RCCL
  RCCLAdapter(ncclComm_t comm, hipStream_t stream)
      : comm_(comm), stream_(stream) {}
#endif
  RCCLAdapter() = default;

  CommunicatorProperties communicatorProperties() const {
    CommunicatorProperties result;
#if defined(TESSERA_HAS_RCCL) && \
    (defined(TESSERA_NCCL_DEVICE_CORE_HEADER) || \
     defined(TESSERA_NCCL_DEVICE_CORE_TMP_HEADER))
    requireReady();
    result = queryCommunicatorProperties(comm_, "RCCL");
#endif
    return result;
  }

#if defined(TESSERA_HAS_RCCL) && NCCL_VERSION_CODE >= 22800
  NativeRegisteredWindow registerSymmetricWindow(void *buffer, size_t bytes,
                                                  bool strictOrdering) const {
    requireReady();
    validateWindow(buffer, bytes);
    const int flags = NCCL_WIN_COLL_SYMMETRIC |
                      (strictOrdering ? NCCL_WIN_STRICT_ORDERING : 0);
    ncclWindow_t window = nullptr;
    checkRccl(ncclCommWindowRegister(comm_, buffer, bytes, &window, flags),
              "symmetric window registration");
    return NativeRegisteredWindow(comm_, window, buffer, bytes, flags, "RCCL");
  }
#endif

#if defined(TESSERA_HAS_RCCL) && NCCL_VERSION_CODE >= 23000
  void putSignalAsync(const void *source, size_t count, WireDType dtype, int peer,
                      const NativeRegisteredWindow &peerWindow,
                      size_t peerWindowOffset, int signalIndex = 0,
                      int context = 0) const {
    requireReady();
    if (!peerWindow.valid())
      throw std::invalid_argument("RCCL put-signal requires a registered peer window");
    checkRccl(ncclPutSignal(source, count, toRcclDType(dtype), peer,
                            peerWindow.nativeHandle(), peerWindowOffset,
                            signalIndex, context, /*flags=*/0, comm_, stream_),
              "put signal");
  }

  void putSignal(const void *source, size_t count, WireDType dtype, int peer,
                 const NativeRegisteredWindow &peerWindow,
                 size_t peerWindowOffset, int signalIndex = 0,
                 int context = 0) const {
    putSignalAsync(source, count, dtype, peer, peerWindow, peerWindowOffset,
                   signalIndex, context);
    synchronize();
  }

  void signalAsync(int peer, int signalIndex = 0, int context = 0) const {
    requireReady();
    checkRccl(ncclSignal(peer, signalIndex, context, /*flags=*/0,
                         comm_, stream_),
              "signal");
  }

  void signal(int peer, int signalIndex = 0, int context = 0) const {
    signalAsync(peer, signalIndex, context);
    synchronize();
  }

  void waitSignalAsync(int peer, int operationCount = 1,
                       int signalIndex = 0, int context = 0) const {
    requireReady();
    if (operationCount < 1)
      throw std::invalid_argument("RCCL wait-signal count must be positive");
    ncclWaitSignalDesc_t descriptor{};
    descriptor.opCnt = operationCount;
    descriptor.peer = peer;
    descriptor.sigIdx = signalIndex;
    descriptor.ctx = context;
    checkRccl(ncclWaitSignal(1, &descriptor, comm_, stream_), "wait signal");
  }

  void waitSignal(int peer, int operationCount = 1,
                  int signalIndex = 0, int context = 0) const {
    waitSignalAsync(peer, operationCount, signalIndex, context);
    synchronize();
  }
#endif

  bool enabled() const override {
#ifdef TESSERA_HAS_RCCL
    return comm_ != nullptr && stream_ != nullptr;
#else
    return false;
#endif
  }

  void all_reduce(void* buf, size_t count, ReduceOp op,
                  const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_RCCL
    requireReady();
    checkRccl(ncclAllReduce(buf, buf, count, toRcclDType(spec.wire_dtype),
                            toRcclOp(op), comm_, stream_),
              "all_reduce");
    synchronize();
#else
    (void)buf; (void)count; (void)op; (void)spec;
#endif
  }

  void reduce_scatter(const void* src, void* dst, size_t count,
                      ReduceOp op, const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_RCCL
    requireDivisible(count, spec.world_size, "reduce_scatter");
    requireReady();
    checkRccl(ncclReduceScatter(src, dst, count / spec.world_size,
                                toRcclDType(spec.wire_dtype), toRcclOp(op),
                                comm_, stream_),
              "reduce_scatter");
    synchronize();
#else
    size_t esz = wireDTypeBytes(spec.wire_dtype);
    size_t shard = count / (size_t)spec.world_size;
    size_t offset = (size_t)spec.local_rank * shard;
    std::memcpy(dst, static_cast<const char*>(src) + offset * esz,
                shard * esz);
    (void)op;
#endif
  }

  void all_gather(const void* src, void* dst, size_t count,
                  const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_RCCL
    requireDivisible(count, spec.world_size, "all_gather");
    requireReady();
    checkRccl(ncclAllGather(src, dst, count / spec.world_size,
                            toRcclDType(spec.wire_dtype), comm_, stream_),
              "all_gather");
    synchronize();
#else
    size_t esz = wireDTypeBytes(spec.wire_dtype);
    size_t shard = count / (size_t)spec.world_size;
    size_t offset = (size_t)spec.local_rank * shard;
    std::memcpy(static_cast<char*>(dst) + offset * esz,
                src, shard * esz);
#endif
  }

  void all_to_all(const void* send_buf, void* recv_buf,
                   size_t shard_bytes, const CollectiveSpec& spec) override {
#ifdef TESSERA_HAS_RCCL
    requireReady();
    const size_t elementBytes = wireDTypeBytes(spec.wire_dtype);
    if (shard_bytes % elementBytes != 0)
      throw std::invalid_argument("all_to_all shard bytes must align to dtype");
    const size_t count = shard_bytes / elementBytes;
    const auto dtype = toRcclDType(spec.wire_dtype);
    checkRccl(ncclGroupStart(), "all_to_all group start");
    for (int peer = 0; peer < spec.world_size; ++peer) {
      const auto *send = static_cast<const char *>(send_buf) + peer * shard_bytes;
      auto *recv = static_cast<char *>(recv_buf) + peer * shard_bytes;
      checkRccl(ncclSend(send, count, dtype, peer, comm_, stream_),
                "all_to_all send");
      checkRccl(ncclRecv(recv, count, dtype, peer, comm_, stream_),
                "all_to_all receive");
    }
    checkRccl(ncclGroupEnd(), "all_to_all group end");
    synchronize();
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

private:
#ifdef TESSERA_HAS_RCCL
#if NCCL_VERSION_CODE >= 22800
  static void validateWindow(void *buffer, size_t bytes) {
    if (!buffer || bytes == 0)
      throw std::invalid_argument("collective window requires a non-empty buffer");
  }
#endif
  static ncclDataType_t toRcclDType(WireDType dtype) {
    switch (dtype) {
      case WireDType::FP32: return ncclFloat32;
      case WireDType::BF16: return ncclBfloat16;
      case WireDType::FP16: return ncclFloat16;
      case WireDType::INT8: return ncclInt8;
      case WireDType::FP8:
        throw std::invalid_argument(
            "FP8 collective storage must distinguish e4m3 from e5m2");
    }
    throw std::invalid_argument("unsupported RCCL wire dtype");
  }

  static ncclRedOp_t toRcclOp(ReduceOp op) {
    switch (op) {
      case ReduceOp::SUM: return ncclSum;
      case ReduceOp::MAX: return ncclMax;
      case ReduceOp::MIN: return ncclMin;
      case ReduceOp::PROD: return ncclProd;
    }
    throw std::invalid_argument("unsupported RCCL reduction");
  }

  static void checkRccl(ncclResult_t result, const char *where) {
    if (result != ncclSuccess)
      throw std::runtime_error(std::string("RCCL ") + where + ": " +
                               ncclGetErrorString(result));
  }

  static void requireDivisible(size_t count, int worldSize,
                               const char *where) {
    if (worldSize < 1 || count % static_cast<size_t>(worldSize) != 0)
      throw std::invalid_argument(std::string(where) +
                                  " count must divide world size");
  }

  void requireReady() const {
    if (!enabled())
      throw std::runtime_error("RCCL adapter requires a communicator and stream");
  }

  void synchronize() const {
    const hipError_t result = hipStreamSynchronize(stream_);
    if (result != hipSuccess)
      throw std::runtime_error(std::string("HIP stream synchronization: ") +
                               hipGetErrorString(result));
  }

  ncclComm_t comm_ = nullptr;
  hipStream_t stream_ = nullptr;
#endif
};

}} // namespace tessera::collective
