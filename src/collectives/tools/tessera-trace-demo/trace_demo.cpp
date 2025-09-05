#include "tessera/Dialect/Collective/Runtime/PerfettoTrace.h"
#include <thread>
#include <chrono>
using namespace tessera::collective;
int main(){
  PerfettoTraceWriter w;
  w.annotate("topology", "NVLink+PCIe");
  w.counter("NVLink_BW_GBs", 300.0, 100, 0);
  w.counter("PCIe_BW_GBs",   32.0,  100, 0);
  w.begin("ComputeTile", "compute", 100, 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  w.begin("CommChunk", "comm", 100, 1); w.counter("chunk_bytes", 65536.0, 100, 1);
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  w.end("CommChunk", "comm", 100, 1);
  w.end("ComputeTile", "compute", 100, 0);
  w.write("tessera_trace.json");
  return 0;
}
