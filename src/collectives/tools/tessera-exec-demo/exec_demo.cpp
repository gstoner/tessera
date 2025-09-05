#include "tessera/Dialect/Collective/Runtime/Execution.h"
#include "tessera/Dialect/Collective/Runtime/Packing.h"
#include <vector>
#include <thread>
#include <random>
#include <iostream>

using namespace tessera::collective;

int main(){
  Policy p = Policy::fromEnv();
  ExecRuntime rt(/*maxInflight=*/2, p, /*pidBase=*/2000);
  // Roofline bands as counters per device stream
  rt.trace().counter("NVLink_BW_GBs", 300.0, 2000, 0);
  rt.trace().counter("PCIe_BW_GBs",   32.0,  2000, 0);

  // Synthesize data and pack to FP8 with optional RLE
  std::vector<float> data(1<<16);
  std::mt19937 gen(0); std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto &x : data) x = dist(gen);

  auto packed = pack_cast_fp32(data.data(), data.size(), WireDType::FP8, /*rle=*/true);
  std::cout << "Packed bytes: " << packed.out_bytes << "\n";

  // Submit several chunks on two streams
  for (int i=0;i<8;++i) {
    ChunkDesc d{packed.bytes.data(), (uint64_t)packed.out_bytes, /*device=*/0, /*stream=*/i%2, /*intraNode=*/true};
    rt.submit(d);
    // Stagger submissions slightly
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

  // Drain: naive wait
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  rt.trace().write("tessera_exec_trace.json");
  std::cout << "Trace written to tessera_exec_trace.json\n";
  return 0;
}
