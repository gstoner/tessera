// Generation-owned device results from compiler-authored paired tensor programs.
namespace {
struct TensorResultLease {
  CUcontext context = nullptr;
  unsigned long long identity = 0;
  CUdeviceptr base = 0;
  CUstream stream = nullptr;
  size_t bytes = 0;
  uint64_t ownerHandle = 0;
  bool materialized = false;
  std::array<TesseraNvidiaMatmulHostView, 2> outputs{};
  std::vector<CUevent> consumers;
  ~TensorResultLease() {
    if (getpid() != process || !context || cuCtxPushCurrent(context) != CUDA_SUCCESS) return;
    unsigned long long live = 0;
    if (cuCtxGetId(context, &live) == CUDA_SUCCESS && live == identity) {
      for (CUevent event : consumers) { cuEventSynchronize(event); cuEventDestroy(event); }
      if (stream) { cuStreamSynchronize(stream); cuStreamDestroy(stream); }
      if (base) cuMemFree(base);
    }
    CUcontext prior = nullptr;
    cuCtxPopCurrent(&prior);
  }
};
std::map<uint64_t, std::unique_ptr<TensorResultLease>> tensorResults;

bool tensorResultContext(TensorResultLease &lease) {
  CUcontext context = nullptr;
  unsigned long long identity = 0;
  if (!ok(cuCtxGetCurrent(&context), "get tensor result context")) return false;
  if (!context && !ok(cuCtxSetCurrent(lease.context), "restore tensor result context")) return false;
  if ((context && context != lease.context) ||
      !ok(cuCtxGetId(lease.context, &identity), "tensor result context identity") ||
      identity != lease.identity)
    return bad("tensor result context changed"), false;
  return true;
}
}

extern "C" int tessera_nvidia_tensor_program_reserve_result(
    uint64_t handle, uint64_t *result) {
  error.clear();
  if (result) *result = 0;
  if (getpid() != process) return bad("resident tensor result cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = tensorPrograms.find(handle);
  if (found == tensorPrograms.end() || !result)
    return bad("resident tensor result reservation owner is closed or unknown");
  auto &owner = *found->second;
  if (!tensorProgramContext(owner)) return 1;
  try {
    auto lease = std::make_unique<TensorResultLease>();
    lease->context = owner.context;
    lease->identity = owner.identity;
    lease->ownerHandle = handle;
    const size_t first = owner.buffers.size() - 2;
    std::array<size_t, 2> offsets{};
    for (size_t i = 0; i < 2; ++i) {
      const auto &expected = owner.buffers[first + i];
      if (expected.dtype != 1 || expected.rank != 2 ||
          expected.shape[0] <= 0 || expected.shape[1] <= 0 ||
          lease->bytes > SIZE_MAX - 255 ||
          expected.bytes > SIZE_MAX - lease->bytes - 255)
        return bad("resident tensor result output contract mismatch");
      offsets[i] = lease->bytes;
      lease->bytes = (lease->bytes + expected.bytes + 255) & ~size_t(255);
      lease->outputs[i] = expected;
    }
    if (!owner.arena->grow(owner.offsets[first]) ||
        !owner.arena->growHost(owner.offsets[first]) ||
        !ok(cuMemAlloc(&lease->base, lease->bytes), "allocate resident tensor result") ||
        !ok(cuStreamCreate(&lease->stream, CU_STREAM_NON_BLOCKING), "create resident tensor result stream"))
      return 1;
    for (size_t i = 0; i < 2; ++i)
      lease->outputs[i].data = reinterpret_cast<void *>(lease->base + offsets[i]);
    uint64_t id = nextHandle++;
    tensorResults.emplace(id, std::move(lease));
    *result = id;
    return 0;
  } catch (...) {
    return bad("resident tensor result reservation failed");
  }
}

extern "C" int tessera_nvidia_tensor_program_submit_result(
    uint64_t handle, uint64_t result,
    const TesseraNvidiaMatmulHostView *views, size_t count,
    const uint64_t *producerStreams, size_t producerCount) {
  error.clear();
  if (getpid() != process) return bad("resident tensor result cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto program = tensorPrograms.find(handle);
  auto output = tensorResults.find(result);
  if (program == tensorPrograms.end() ||
      output == tensorResults.end() || output->second->ownerHandle != handle ||
      output->second->materialized || !views || count != 3 ||
      (producerStreams ? producerCount != 3 : producerCount != 0))
    return bad("resident tensor result submission frame differs");
  auto &owner = *program->second;
  auto &lease = *output->second;
  if (!tensorProgramContext(owner) || !tensorResultContext(lease)) return 1;
  try {
    std::vector<CUdeviceptr> bindings(owner.buffers.size());
    for (size_t i = 0; i < 3; ++i) {
      if (!tensorProgramView(views[i], owner.buffers[i]))
        return bad("resident tensor result input shape/stride/storage mismatch");
      bindings[i] = reinterpret_cast<CUdeviceptr>(views[i].data);
      if (producerStreams && !residentAllocation(bindings[i], views[i].bytes, owner.context))
        return bad("resident tensor result allocation context/capacity mismatch");
      if (producerStreams && owner.arena->base && bindings[i] < owner.arena->base + owner.arena->capacity &&
          owner.arena->base < bindings[i] + views[i].bytes)
        return bad("resident tensor result input aliases private scratch");
    }
    const size_t first = owner.buffers.size() - 2;
    // Reservation allocates all GPU storage before this operation. This path
    // must not gain an allocation fence in place of producer event ordering.
    if (!owner.arena->base || owner.arena->capacity < owner.offsets[first])
      return bad("resident tensor result private reservation is unavailable");
    for (size_t i = 3; i < first; ++i)
      bindings[i] = owner.arena->base + owner.offsets[i];
    for (size_t i = 0; i < 2; ++i)
      bindings[first + i] = reinterpret_cast<CUdeviceptr>(lease.outputs[i].data);
    owner.generation = 0;
    ++owner.arena->generation;
    ProducerEvents ordering;
    if (producerStreams) {
      if (!ordering.wait(owner.context, lease.stream, producerStreams, producerCount)) return 1;
    } else {
      auto *host = static_cast<unsigned char *>(owner.arena->host);
      for (size_t i = 0; i < 3; ++i) {
        std::memcpy(host + owner.offsets[i], views[i].data, views[i].bytes);
        bindings[i] = owner.arena->base + owner.offsets[i];
        if (!ok(cuMemcpyHtoDAsync(bindings[i], host + owner.offsets[i],
                                 views[i].bytes, lease.stream), "upload tensor result root")) {
          const std::string saved = error;
          cuStreamSynchronize(lease.stream);
          error = saved;
          return 1;
        }
      }
    }
    bool submitted = submitTensorProgram(owner, 1, nullptr, false, bindings.data(), lease.stream);
    const std::string saved = error;
    bool completed = ok(cuStreamSynchronize(lease.stream), "complete resident tensor result");
    if (!submitted) { error = saved; return 1; }
    if (!completed) return 1;
    lease.materialized = true;
    return 0;
  } catch (...) {
    return bad("resident tensor result submission failed");
  }
}

extern "C" int tessera_nvidia_tensor_result_query(
    uint64_t handle, size_t ordinal, TesseraNvidiaMatmulHostView *view,
    uint64_t *stream, uint64_t *identity) {
  error.clear();
  if (view) *view = {};
  if (stream) *stream = 0;
  if (identity) *identity = 0;
  if (getpid() != process) return bad("resident tensor result cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = tensorResults.find(handle);
  if (found == tensorResults.end() || !found->second->materialized || ordinal >= 2 || !view || !stream || !identity)
    return bad("resident tensor result query is closed or malformed");
  auto &lease = *found->second;
  if (!tensorResultContext(lease)) return 1;
  *view = lease.outputs[ordinal];
  *stream = reinterpret_cast<uint64_t>(lease.stream);
  *identity = lease.identity;
  return 0;
}

extern "C" int tessera_nvidia_tensor_result_read(
    uint64_t handle, size_t ordinal, const TesseraNvidiaMatmulHostView *output) {
  error.clear();
  if (getpid() != process) return bad("resident tensor result cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = tensorResults.find(handle);
  if (found == tensorResults.end() || !found->second->materialized || ordinal >= 2 || !output)
    return bad("resident tensor result read is closed or malformed");
  auto &lease = *found->second;
  if (!tensorResultContext(lease)) return 1;
  if (!tensorProgramView(*output, lease.outputs[ordinal]))
    return bad("resident tensor result host output differs");
  bool submitted = ok(cuMemcpyDtoHAsync(output->data,
      reinterpret_cast<CUdeviceptr>(lease.outputs[ordinal].data),
      output->bytes, lease.stream), "read resident tensor result");
  const std::string saved = error;
  bool completed = ok(cuStreamSynchronize(lease.stream), "complete resident tensor result read");
  if (!submitted) { error = saved; return 1; }
  return completed ? 0 : 1;
}

extern "C" int tessera_nvidia_tensor_result_record_consumer(uint64_t handle, uint64_t stream) {
  error.clear();
  if (getpid() != process) return bad("resident tensor result cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = tensorResults.find(handle);
  if (found == tensorResults.end() || !found->second->materialized || !stream)
    return bad("resident tensor result consumer is closed or malformed");
  auto &lease = *found->second;
  if (!tensorResultContext(lease)) return 1;
  // Match ProducerEvents: only 1/2 are CUDA default-stream sentinels.
  // The owning driver dereferences malformed low opaque handles instead of
  // returning INVALID_HANDLE, so refuse them before calling into CUDA.
  if (stream > 2 && stream < 4096)
    return bad("resident tensor result consumer stream token is invalid");
  CUstream consumer = reinterpret_cast<CUstream>(stream);
  CUcontext context = nullptr;
  if (!ok(cuStreamGetCtx(consumer, &context), "get resident result consumer context") ||
      context != lease.context)
    return bad("resident tensor result consumer context differs");
  // If retaining a completion event fails after a consumer was enqueued,
  // drain that stream before releasing its Python generation pin.
  auto drainFailure = [&]() {
    const std::string saved = error;
    cuStreamSynchronize(consumer);
    error = saved;
    return 1;
  };
  for (auto it = lease.consumers.begin(); it != lease.consumers.end();) {
    CUresult status = cuEventQuery(*it);
    if (status == CUDA_SUCCESS) {
      cuEventDestroy(*it);
      it = lease.consumers.erase(it);
    } else if (status == CUDA_ERROR_NOT_READY) ++it;
    else { ok(status, "query resident result consumer"); return drainFailure(); }
  }
  if (lease.consumers.size() >= 256) {
    bad("resident tensor result has too many pending consumer streams");
    return drainFailure();
  }
  CUevent event = nullptr;
  if (!ok(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING), "create resident result consumer event"))
    return drainFailure();
  if (!ok(cuEventRecord(event, consumer), "record resident result consumer completion")) {
    cuEventDestroy(event);
    return drainFailure();
  }
  try { lease.consumers.push_back(event); }
  catch (...) { cuEventSynchronize(event); cuEventDestroy(event); return bad("retain resident result consumer failed"); }
  return 0;
}

extern "C" int tessera_nvidia_tensor_result_close(uint64_t handle) {
  error.clear();
  if (getpid() != process) return bad("resident tensor result cannot cross fork");
  std::lock_guard<std::mutex> lock(mutex);
  auto found = tensorResults.find(handle);
  if (found == tensorResults.end()) return bad("resident tensor result is closed or unknown");
  tensorResults.erase(found);
  return 0;
}
