#include "tessera/Dialect/Collective/Runtime/Adapters.h"

#include <nccl_device.h>

#include <array>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

using tessera::collective::CommunicatorProperties;
using tessera::collective::NativeRegisteredWindow;
using tessera::collective::RCCLAdapter;

namespace {

constexpr int kRanks = 2;
constexpr size_t kWindowBytes = 4096;
constexpr int kStampBase = 0x115100;

class EvidenceBlocked : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

__global__ void writeEveryLsaPeer(ncclWindow_t window,
                                  struct ncclDevComm deviceComm) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const ncclTeam lsa = ncclTeamLsa(deviceComm);
  for (int peer = 0; peer < lsa.nRanks; ++peer) {
    int *remote = static_cast<int *>(ncclGetLsaPointer(window, 0, peer));
    remote[lsa.rank] = kStampBase + lsa.rank;
  }
}

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

struct RankState {
  int device = -1;
  hipStream_t stream = nullptr;
  void *buffer = nullptr;
  std::unique_ptr<RCCLAdapter> adapter;
  std::unique_ptr<NativeRegisteredWindow> window;
  CommunicatorProperties properties;
  ncclDevComm deviceComm{};
  bool deviceCommCreated = false;
};

template <typename Fn>
void runRanks(Fn fn) {
  std::array<std::thread, kRanks> threads;
  std::array<std::exception_ptr, kRanks> failures{};
  for (int rank = 0; rank < kRanks; ++rank) {
    threads[rank] = std::thread([&, rank] {
      try {
        fn(rank);
      } catch (...) {
        failures[rank] = std::current_exception();
      }
    });
  }
  for (auto &thread : threads) thread.join();
  for (const auto &failure : failures)
    if (failure) std::rethrow_exception(failure);
}

bool isGfx1151(int device, std::string &name) {
  hipDeviceProp_t properties{};
  checkHip(hipGetDeviceProperties(&properties, device), "device properties");
  name = properties.gcnArchName;
  return name.find("gfx1151") != std::string::npos;
}

void validateSingleRankFoundation() {
  int device = 0;
  ncclComm_t communicator = nullptr;
  hipStream_t stream = nullptr;
  void *buffer = nullptr;
  checkHip(hipSetDevice(device), "foundation set device");
  checkRccl(ncclCommInitAll(&communicator, 1, &device),
            "foundation communicator initialization");
  checkHip(hipStreamCreate(&stream), "foundation stream creation");
  checkRccl(ncclMemAlloc(&buffer, kWindowBytes),
            "foundation symmetric allocation");
  RCCLAdapter adapter(communicator, stream);
  const auto properties = adapter.communicatorProperties();
  if (!properties.query_available || properties.rank != 0 ||
      properties.world_size != 1 || properties.device != 0)
    throw std::runtime_error("single-rank communicator properties are inconsistent");
  if (!properties.device_api_support) {
    std::ostringstream reason;
    reason << "communicator reports device_api_support=false"
           << ", lsa_team_count=" << properties.lsa_team_count
           << ", gin_type=" << properties.gin_type
           << ", host_rma_support="
           << (properties.host_rma_support ? "true" : "false");
    throw EvidenceBlocked(reason.str());
  }
  auto window = adapter.registerSymmetricWindow(
      buffer, kWindowBytes, /*strictOrdering=*/true);
  if (!window.valid() || window.buffer() != buffer ||
      window.bytes() != kWindowBytes ||
      !(window.flags() & NCCL_WIN_COLL_SYMMETRIC) ||
      !(window.flags() & NCCL_WIN_STRICT_ORDERING))
    throw std::runtime_error("registered-window ownership contract is inconsistent");
  window.close();
  checkRccl(ncclMemFree(buffer), "foundation symmetric free");
  checkHip(hipStreamDestroy(stream), "foundation stream destruction");
  checkRccl(ncclCommDestroy(communicator),
            "foundation communicator destruction");
}

} // namespace

int main() {
  int deviceCount = 0;
  if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1) {
    std::cout << "{\"schema\":\"tessera.rccl_lsa_packet.v1\","
                 "\"status\":\"blocked\",\"reason\":"
                 "\"requires a visible gfx1151 device\","
              << "\"visible_devices\":" << deviceCount << "}\n";
    return 77;
  }

  std::array<std::string, kRanks> architectures;
  if (!isGfx1151(0, architectures[0])) {
    std::cout << "{\"schema\":\"tessera.rccl_lsa_packet.v1\","
                 "\"status\":\"blocked\",\"reason\":"
                 "\"device zero must be gfx1151\"}\n";
    return 77;
  }
  try {
    validateSingleRankFoundation();
  } catch (const EvidenceBlocked &blocked) {
    std::cout << "{\"schema\":\"tessera.rccl_lsa_packet.v1\","
                 "\"status\":\"blocked\","
                 "\"communicator_properties\":\"pass\","
                 "\"symmetric_window\":\"blocked\","
                 "\"reason\":\"" << blocked.what() << "\","
              << "\"visible_devices\":" << deviceCount << "}\n";
    return 77;
  } catch (const std::exception &error) {
    std::cerr << "RCCL communicator/window foundation failure: "
              << error.what() << "\n";
    return 2;
  }
  if (deviceCount < kRanks) {
    std::cout << "{\"schema\":\"tessera.rccl_lsa_packet.v1\","
                 "\"status\":\"blocked\",\"foundation\":\"pass\","
                 "\"reason\":\"requires two distinct visible gfx1151 devices\","
              << "\"visible_devices\":" << deviceCount << "}\n";
    return 77;
  }
  if (!isGfx1151(1, architectures[1])) {
    std::cout << "{\"schema\":\"tessera.rccl_lsa_packet.v1\","
                 "\"status\":\"blocked\",\"foundation\":\"pass\","
                 "\"reason\":\"first two devices must both be gfx1151\"}\n";
    return 77;
  }

  std::array<int, kRanks> devices{0, 1};
  std::array<ncclComm_t, kRanks> communicators{};
  std::array<RankState, kRanks> states{};
  try {
    checkRccl(ncclCommInitAll(communicators.data(), kRanks, devices.data()),
              "communicator initialization");
    runRanks([&](int rank) {
      auto &state = states[rank];
      state.device = devices[rank];
      checkHip(hipSetDevice(state.device), "set device");
      checkHip(hipStreamCreate(&state.stream), "stream creation");
      checkRccl(ncclMemAlloc(&state.buffer, kWindowBytes), "symmetric allocation");
      checkHip(hipMemsetAsync(state.buffer, 0, kWindowBytes, state.stream),
               "window initialization");
      checkHip(hipStreamSynchronize(state.stream), "window initialization sync");
      state.adapter = std::make_unique<RCCLAdapter>(communicators[rank],
                                                     state.stream);
      state.properties = state.adapter->communicatorProperties();
      if (!state.properties.query_available ||
          !state.properties.device_api_support ||
          state.properties.world_size != kRanks ||
          state.properties.lsa_team_count != 1)
        throw std::runtime_error(
            "communicator is not one device-API-capable LSA team");
      auto window = state.adapter->registerSymmetricWindow(
          state.buffer, kWindowBytes, /*strictOrdering=*/true);
      state.window = std::make_unique<NativeRegisteredWindow>(std::move(window));
      ncclDevCommRequirements requirements =
          NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      checkRccl(ncclDevCommCreate(communicators[rank], &requirements,
                                  &state.deviceComm),
                "device communicator creation");
      state.deviceCommCreated = true;
    });

    for (int rank = 0; rank < kRanks; ++rank) {
      auto &state = states[rank];
      checkHip(hipSetDevice(state.device), "set launch device");
      hipLaunchKernelGGL(writeEveryLsaPeer, dim3(1), dim3(64), 0,
                         state.stream, state.window->nativeHandle(),
                         state.deviceComm);
    }
    for (auto &state : states) {
      checkHip(hipSetDevice(state.device), "set synchronization device");
      checkHip(hipStreamSynchronize(state.stream), "LSA kernel synchronization");
    }

    for (auto &state : states) {
      std::array<int, kRanks> observed{};
      checkHip(hipSetDevice(state.device), "set readback device");
      checkHip(hipMemcpy(observed.data(), state.buffer, sizeof(observed),
                         hipMemcpyDeviceToHost),
               "LSA readback");
      for (int rank = 0; rank < kRanks; ++rank)
        if (observed[rank] != kStampBase + rank)
          throw std::runtime_error("LSA peer write produced incorrect data");
    }

  } catch (const std::exception &error) {
    std::cerr << "RCCL LSA correctness failure: " << error.what() << "\n";
    return 2;
  }

  try {
    runRanks([&](int rank) {
      auto &state = states[rank];
      checkHip(hipSetDevice(state.device), "set cleanup device");
      if (state.deviceCommCreated)
        checkRccl(ncclDevCommDestroy(communicators[rank], &state.deviceComm),
                  "device communicator destruction");
      if (state.window) state.window->close();
      state.window.reset();
      state.adapter.reset();
      if (state.buffer) checkRccl(ncclMemFree(state.buffer), "symmetric free");
      if (state.stream) checkHip(hipStreamDestroy(state.stream), "stream destruction");
    });
    for (auto comm : communicators)
      if (comm) checkRccl(ncclCommDestroy(comm), "communicator destruction");
  } catch (const std::exception &error) {
    std::cerr << "RCCL LSA cleanup failure: " << error.what() << "\n";
    return 3;
  }
  std::cout << "{\"schema\":\"tessera.rccl_lsa_packet.v1\","
               "\"status\":\"pass\",\"ranks\":2,"
               "\"architecture\":\"gfx1151\","
               "\"window\":\"symmetric_strict\","
               "\"device_api_support\":true,\"lsa_teams\":1}\n";
  return 0;
}
