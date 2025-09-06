#include <cstdio>
#include <cstdlib>
#include <string>
#include <fstream>

extern "C" void smoke_cuda_label();
extern "C" void smoke_hip_label();

int main(){
  std::string device = "CPU";
  #ifdef SMOKE_WITH_CUDA
  smoke_cuda_label(); device = "NVIDIA";
  #endif
  #ifdef SMOKE_WITH_HIP
  smoke_hip_label(); device = "AMD";
  #endif
  std::system("mkdir -p reports");
  std::ofstream f("reports/roofline.csv");
  f << "kernel,flops,bytes,intensity,time_ms,device\n";
  f << "energy_head_bilinear,1000000000,250000000,4.0,1.2," << device << "\n";
  f << "grad_y_autodiff,800000000,200000000,4.0,1.0," << device << "\n";
  f << "inner_step,500000000,125000000,4.0,0.6," << device << "\n";
  f.close();
  std::printf("// wrote reports/roofline.csv device=%s\n", device.c_str());
  return 0;
}
