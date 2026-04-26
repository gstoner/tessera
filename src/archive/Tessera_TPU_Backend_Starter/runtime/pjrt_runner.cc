#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
  // Starter stub: we only print environment hints for PJRT TPU.
  std::cout << "PJRT TPU Runner (starter)" << std::endl;
  const char* dev = std::getenv("PJRT_DEVICE");
  std::cout << "PJRT_DEVICE=" << (dev ? dev : "(unset)") << std::endl;
  std::cout << "Hint: set PJRT_DEVICE=TPU on a TPU VM with libtpu installed." << std::endl;
  std::cout << "Next: link PJRT C API and enumerate devices, compile StableHLO." << std::endl;
  return 0;
}
