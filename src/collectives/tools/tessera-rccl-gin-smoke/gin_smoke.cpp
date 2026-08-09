#include "tessera/Dialect/Collective/Runtime/Adapters.h"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

using tessera::collective::CommunicatorProperties;
using tessera::collective::NativeRegisteredWindow;
using tessera::collective::RCCLAdapter;
using tessera::collective::WireDType;

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr int kBlocked = 77;
constexpr int kSignalIndex = 0;
constexpr int kStandaloneSignalIndex = 1;
constexpr int kRmaContext = 0;

struct Launch {
  int rank = -1;
  int world = 0;
  int localRank = 0;
  int warmup = 10;
  int iterations = 100;
  int timeoutSeconds = 120;
  size_t bytes = 1U << 20;
  std::string runId;
  std::string rendezvous;
  std::string artifactDigest;
  std::string communicatorDigest;
};

struct RankResult {
  int rank = -1;
  bool correct = false;
  double hostNs = 0.0;
  double hipEventNs = 0.0;
  CommunicatorProperties properties;
  std::string host;
  std::string arch;
};

void checkHip(hipError_t result, const char *where) {
  if (result != hipSuccess)
    throw std::runtime_error(std::string(where) + ": " +
                             hipGetErrorString(result));
}

void checkRccl(ncclResult_t result, const char *where) {
  if (result != ncclSuccess)
    throw std::runtime_error(std::string(where) + ": " +
                             ncclGetErrorString(result));
}

std::string jsonEscape(const std::string &value) {
  std::string out;
  for (const char c : value) {
    if (c == '\\' || c == '"') out.push_back('\\');
    if (c == '\n') out += "\\n";
    else if (c != '\r') out.push_back(c);
  }
  return out;
}

const char *firstEnvironment(
    std::initializer_list<const char *> candidates) {
  for (const char *name : candidates) {
    const char *value = std::getenv(name);
    if (value && *value) return value;
  }
  return nullptr;
}

int parsePositive(const char *value, const char *name, bool allowZero = false) {
  if (!value) throw std::runtime_error(std::string("missing ") + name);
  char *end = nullptr;
  const long parsed = std::strtol(value, &end, 10);
  const long minimum = allowZero ? 0 : 1;
  if (*value == '\0' || *end != '\0' || parsed < minimum ||
      parsed > std::numeric_limits<int>::max())
    throw std::runtime_error(std::string("invalid ") + name);
  return static_cast<int>(parsed);
}

size_t parseBytes(const char *value) {
  if (!value) return 1U << 20;
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (*value == '\0' || *end != '\0' || parsed < sizeof(float) ||
      parsed % sizeof(float) != 0)
    throw std::runtime_error(
        "TESSERA_GIN_BYTES must be a positive multiple of four");
  return static_cast<size_t>(parsed);
}

bool validToken(const std::string &value) {
  if (value.empty() || value.size() > 128) return false;
  for (const char c : value)
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_')
      return false;
  return true;
}

bool validDigest(const std::string &value) {
  if (value.size() != 64) return false;
  for (const char c : value)
    if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
  return true;
}

Launch discoverLaunch() {
  Launch launch;
  launch.rank = parsePositive(firstEnvironment({
      "TESSERA_GIN_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK",
      "SLURM_PROCID"}), "launcher rank", true);
  launch.world = parsePositive(firstEnvironment({
      "TESSERA_GIN_WORLD_SIZE", "OMPI_COMM_WORLD_SIZE", "PMI_SIZE",
      "SLURM_NTASKS"}), "launcher world size");
  const char *localRank = firstEnvironment({
      "TESSERA_GIN_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK",
      "MPI_LOCALRANKID", "SLURM_LOCALID"});
  launch.localRank = localRank ? parsePositive(localRank, "local rank", true) : 0;
  const char *runId = std::getenv("TESSERA_GIN_RUN_ID");
  const char *rendezvous = std::getenv("TESSERA_GIN_RENDEZVOUS");
  if (!runId || !validToken(runId))
    throw std::runtime_error("TESSERA_GIN_RUN_ID must be an alphanumeric run token");
  if (!rendezvous || !*rendezvous)
    throw std::runtime_error("TESSERA_GIN_RENDEZVOUS must name a shared directory");
  launch.runId = runId;
  launch.rendezvous = rendezvous;
  if (const char *value = std::getenv("TESSERA_GIN_WARMUP"))
    launch.warmup = parsePositive(value, "warmup", true);
  if (const char *value = std::getenv("TESSERA_GIN_ITERATIONS"))
    launch.iterations = parsePositive(value, "iterations");
  if (const char *value = std::getenv("TESSERA_GIN_TIMEOUT_SECONDS"))
    launch.timeoutSeconds = parsePositive(value, "timeout");
  launch.bytes = parseBytes(std::getenv("TESSERA_GIN_BYTES"));
  if (const char *value = std::getenv("TESSERA_GIN_ARTIFACT_DIGEST"))
    launch.artifactDigest = value;
  if (const char *value = std::getenv("TESSERA_GIN_COMMUNICATOR_DIGEST"))
    launch.communicatorDigest = value;
  if (launch.rank >= launch.world)
    throw std::runtime_error("launcher rank must be smaller than world size");
  return launch;
}

fs::path runPath(const Launch &launch, const std::string &suffix) {
  return fs::path(launch.rendezvous) / ("tessera-gin-" + launch.runId + suffix);
}

void atomicWrite(const fs::path &path, const void *data, size_t bytes,
                 int rank) {
  const fs::path temporary = path.string() + ".tmp." + std::to_string(rank);
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create rendezvous file");
    stream.write(static_cast<const char *>(data), bytes);
    stream.flush();
    if (!stream) throw std::runtime_error("cannot write rendezvous file");
  }
  fs::rename(temporary, path);
}

void waitFor(const fs::path &path, int timeoutSeconds) {
  const auto deadline = Clock::now() + std::chrono::seconds(timeoutSeconds);
  while (!fs::exists(path)) {
    if (Clock::now() >= deadline)
      throw std::runtime_error("rendezvous timeout waiting for " + path.string());
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
}

void barrier(const Launch &launch, const std::string &phase) {
  const fs::path mine = runPath(
      launch, "." + phase + ".rank" + std::to_string(launch.rank));
  const char ready = 1;
  atomicWrite(mine, &ready, sizeof(ready), launch.rank);
  for (int rank = 0; rank < launch.world; ++rank)
    waitFor(runPath(launch, "." + phase + ".rank" + std::to_string(rank)),
            launch.timeoutSeconds);
}

ncclUniqueId exchangeUniqueId(const Launch &launch) {
  const fs::path path = runPath(launch, ".nccl-id");
  ncclUniqueId id{};
  if (launch.rank == 0) {
    checkRccl(ncclGetUniqueId(&id), "unique-id creation");
    atomicWrite(path, &id, sizeof(id), launch.rank);
  } else {
    waitFor(path, launch.timeoutSeconds);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char *>(&id), sizeof(id));
    if (!stream) throw std::runtime_error("invalid RCCL unique-id rendezvous file");
  }
  return id;
}

std::string hostname() {
  char value[256]{};
  if (gethostname(value, sizeof(value) - 1) != 0) return "unknown";
  return value;
}

void writeRankResult(const Launch &launch, const RankResult &result) {
  std::ostringstream row;
  row.precision(17);
  row << result.rank << '\n' << (result.correct ? 1 : 0) << '\n'
      << result.hostNs << '\n' << result.hipEventNs << '\n'
      << result.properties.gin_type << '\n'
      << (result.properties.host_rma_support ? 1 : 0) << '\n'
      << result.host << '\n' << result.arch << '\n';
  const std::string text = row.str();
  atomicWrite(runPath(launch, ".result.rank" + std::to_string(result.rank)),
              text.data(), text.size(), result.rank);
}

RankResult readRankResult(const Launch &launch, int rank) {
  RankResult result;
  int correct = 0;
  int hostRma = 0;
  std::ifstream stream(runPath(launch, ".result.rank" + std::to_string(rank)));
  stream >> result.rank >> correct >> result.hostNs >> result.hipEventNs
         >> result.properties.gin_type >> hostRma >> result.host >> result.arch;
  if (!stream || result.rank != rank)
    throw std::runtime_error("invalid rank evidence file");
  result.correct = correct != 0;
  result.properties.host_rma_support = hostRma != 0;
  return result;
}

void emitBlocked(const std::string &reason) {
  std::cout << "{\"schema\":\"tessera.rccl_gin_packet.v1\","
               "\"status\":\"blocked\",\"reason\":\""
            << jsonEscape(reason) << "\"}\n";
}

void emitPacket(const Launch &launch, const std::vector<RankResult> &results,
                int rcclVersion, int hipRuntimeVersion, int hipDriverVersion) {
  bool correct = true;
  bool capabilities = true;
  double maxHostNs = 0.0;
  double maxEventNs = 0.0;
  for (const auto &row : results) {
    correct &= row.correct;
    capabilities &= row.properties.host_rma_support && row.properties.gin_type != 0;
    maxHostNs = std::max(maxHostNs, row.hostNs);
    maxEventNs = std::max(maxEventNs, row.hipEventNs);
  }
  const bool digests = validDigest(launch.artifactDigest) &&
                       validDigest(launch.communicatorDigest);
  const bool promotable = correct && capabilities && digests && maxEventNs > 0.0;
  std::cout.precision(17);
  std::cout << "{\"schema\":\"tessera.rccl_gin_packet.v1\","
            << "\"status\":\"" << (correct ? "pass" : "fail") << "\","
            << "\"eligible_for_promotion\":" << (promotable ? "true" : "false") << ','
            << "\"run_id\":\"" << jsonEscape(launch.runId) << "\","
            << "\"world_size\":" << launch.world << ','
            << "\"bytes\":" << launch.bytes << ','
            << "\"warmup\":" << launch.warmup << ','
            << "\"iterations\":" << launch.iterations << ','
            << "\"operations\":[\"put_signal\",\"signal\",\"wait_signal\"],"
            << "\"artifact_digest\":\"" << jsonEscape(launch.artifactDigest) << "\","
            << "\"communicator_digest\":\"" << jsonEscape(launch.communicatorDigest) << "\","
            << "\"rccl_version\":" << rcclVersion << ','
            << "\"hip_runtime_version\":" << hipRuntimeVersion << ','
            << "\"hip_driver_version\":" << hipDriverVersion << ','
            << "\"timing\":{\"host_wall_ns_per_iteration\":" << maxHostNs
            << ",\"hip_event_ns_per_iteration\":" << maxEventNs
            << ",\"aggregation\":\"maximum_rank\"},\"ranks\":[";
  for (size_t index = 0; index < results.size(); ++index) {
    if (index) std::cout << ',';
    const auto &row = results[index];
    std::cout << "{\"rank\":" << row.rank << ",\"host\":\""
              << jsonEscape(row.host) << "\",\"arch\":\""
              << jsonEscape(row.arch) << "\",\"correct\":"
              << (row.correct ? "true" : "false") << ",\"gin_type\":"
              << row.properties.gin_type << ",\"host_rma_support\":"
              << (row.properties.host_rma_support ? "true" : "false") << '}';
  }
  std::cout << "]}\n";
}

} // namespace

int main() {
#if !defined(TESSERA_HAS_RCCL) || NCCL_VERSION_CODE < 23000
  emitBlocked("requires RCCL 2.30 or newer with public GIN/RMA APIs");
  return kBlocked;
#else
  Launch launch;
  try {
    launch = discoverLaunch();
  } catch (const std::exception &error) {
    emitBlocked(error.what());
    return kBlocked;
  }
  if (launch.world < 2) {
    emitBlocked("GIN evidence requires at least two launcher ranks");
    return kBlocked;
  }

  ncclComm_t communicator = nullptr;
  hipStream_t stream = nullptr;
  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  void *source = nullptr;
  void *destination = nullptr;
  try {
    fs::create_directories(launch.rendezvous);
    int deviceCount = 0;
    checkHip(hipGetDeviceCount(&deviceCount), "device count");
    if (launch.localRank >= deviceCount)
      throw std::runtime_error("local rank has no visible GPU ordinal");
    checkHip(hipSetDevice(launch.localRank), "set local device");
    hipDeviceProp_t deviceProperties{};
    checkHip(hipGetDeviceProperties(&deviceProperties, launch.localRank),
             "device properties");
    const std::string arch = deviceProperties.gcnArchName;
    if (arch.find("gfx1151") == std::string::npos) {
      emitBlocked("exact-device GIN packet requires gfx1151 on every rank");
      return kBlocked;
    }

    const ncclUniqueId uniqueId = exchangeUniqueId(launch);
    checkRccl(ncclCommInitRank(&communicator, launch.world, uniqueId, launch.rank),
              "multi-node communicator initialization");
    checkHip(hipStreamCreate(&stream), "stream creation");
    RCCLAdapter adapter(communicator, stream);
    const CommunicatorProperties properties = adapter.communicatorProperties();
    if (!properties.query_available || properties.rank != launch.rank ||
        properties.world_size != launch.world)
      throw std::runtime_error("RCCL communicator properties disagree with launcher");
    if (!properties.host_rma_support || properties.gin_type == 0) {
      std::ostringstream reason;
      reason << "communicator lacks GIN/RMA: host_rma_support="
             << (properties.host_rma_support ? "true" : "false")
             << ", gin_type=" << properties.gin_type;
      emitBlocked(reason.str());
      return kBlocked;
    }

    checkHip(hipMalloc(&source, launch.bytes), "source allocation");
    checkRccl(ncclMemAlloc(&destination, launch.bytes),
              "registered destination allocation");
    std::vector<float> input(launch.bytes / sizeof(float));
    for (size_t index = 0; index < input.size(); ++index)
      input[index] = static_cast<float>(launch.rank * 1000) +
                     static_cast<float>(index % 251);
    checkHip(hipMemcpy(source, input.data(), launch.bytes, hipMemcpyHostToDevice),
             "source upload");
    checkHip(hipMemset(destination, 0, launch.bytes), "destination initialization");
    auto window = adapter.registerSymmetricWindow(
        destination, launch.bytes, /*strictOrdering=*/true);
    barrier(launch, "registered");

    const int peer = (launch.rank + 1) % launch.world;
    const int previous = (launch.rank + launch.world - 1) % launch.world;
    int operationCount = 0;
    for (int index = 0; index < launch.warmup; ++index) {
      adapter.putSignalAsync(source, input.size(), WireDType::FP32, peer,
                             window, 0, kSignalIndex, kRmaContext);
      adapter.waitSignalAsync(previous, ++operationCount,
                              kSignalIndex, kRmaContext);
    }
    checkHip(hipStreamSynchronize(stream), "warmup synchronization");
    adapter.signalAsync(peer, kStandaloneSignalIndex, kRmaContext);
    adapter.waitSignalAsync(previous, 1, kStandaloneSignalIndex, kRmaContext);
    checkHip(hipStreamSynchronize(stream), "standalone signal synchronization");
    barrier(launch, "timed");

    checkHip(hipEventCreate(&start), "start event creation");
    checkHip(hipEventCreate(&stop), "stop event creation");
    const auto hostStart = Clock::now();
    checkHip(hipEventRecord(start, stream), "start event record");
    for (int index = 0; index < launch.iterations; ++index) {
      adapter.putSignalAsync(source, input.size(), WireDType::FP32, peer,
                             window, 0, kSignalIndex, kRmaContext);
      adapter.waitSignalAsync(previous, ++operationCount,
                              kSignalIndex, kRmaContext);
    }
    checkHip(hipEventRecord(stop, stream), "stop event record");
    checkHip(hipEventSynchronize(stop), "timed synchronization");
    const auto hostStop = Clock::now();
    float elapsedMs = 0.0F;
    checkHip(hipEventElapsedTime(&elapsedMs, start, stop), "event elapsed time");

    std::vector<float> observed(input.size());
    checkHip(hipMemcpy(observed.data(), destination, launch.bytes,
                       hipMemcpyDeviceToHost), "destination readback");
    bool correct = true;
    for (size_t index = 0; index < observed.size(); ++index) {
      const float expected = static_cast<float>(previous * 1000) +
                             static_cast<float>(index % 251);
      if (observed[index] != expected) {
        correct = false;
        break;
      }
    }

    RankResult result;
    result.rank = launch.rank;
    result.correct = correct;
    result.hostNs = std::chrono::duration<double, std::nano>(
                        hostStop - hostStart).count() / launch.iterations;
    result.hipEventNs = static_cast<double>(elapsedMs) * 1.0e6 /
                        launch.iterations;
    result.properties = properties;
    result.host = hostname();
    result.arch = arch;
    writeRankResult(launch, result);
    barrier(launch, "results");

    if (launch.rank == 0) {
      std::vector<RankResult> results;
      for (int rank = 0; rank < launch.world; ++rank)
        results.push_back(readRankResult(launch, rank));
      int rcclVersion = 0;
      int hipRuntimeVersion = 0;
      int hipDriverVersion = 0;
      checkRccl(ncclGetVersion(&rcclVersion), "RCCL version query");
      checkHip(hipRuntimeGetVersion(&hipRuntimeVersion), "HIP runtime version query");
      checkHip(hipDriverGetVersion(&hipDriverVersion), "HIP driver version query");
      emitPacket(launch, results, rcclVersion, hipRuntimeVersion,
                 hipDriverVersion);
    }

    window.close();
    checkRccl(ncclMemFree(destination), "destination free");
    destination = nullptr;
    checkHip(hipFree(source), "source free");
    source = nullptr;
    checkHip(hipEventDestroy(start), "start event destruction");
    start = nullptr;
    checkHip(hipEventDestroy(stop), "stop event destruction");
    stop = nullptr;
    checkHip(hipStreamDestroy(stream), "stream destruction");
    stream = nullptr;
    checkRccl(ncclCommDestroy(communicator), "communicator destruction");
    communicator = nullptr;
    return correct ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "RCCL GIN packet failure on rank " << launch.rank << ": "
              << error.what() << '\n';
    if (start) (void)hipEventDestroy(start);
    if (stop) (void)hipEventDestroy(stop);
    if (source) (void)hipFree(source);
    if (destination) ncclMemFree(destination);
    if (stream) (void)hipStreamDestroy(stream);
    if (communicator) ncclCommAbort(communicator);
    return 2;
  }
#endif
}
