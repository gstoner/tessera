#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cmath>
#include <iostream>

struct Timer {
  using clk = std::chrono::high_resolution_clock;
  clk::time_point t0;
  void start(){ t0 = clk::now(); }
  double stop_ms(){ auto dt = clk::now() - t0; return std::chrono::duration<double,std::milli>(dt).count(); }
};

// Toy kernels
static void energy_head_bilinear(const std::vector<float>& y,
                                 const std::vector<float>& W,
                                 const std::vector<float>& h,
                                 int D, float& Eacc) {
  // y^T W h (collapsed over tokens); super-simplified
  std::vector<float> Wy(D,0.f);
  for (int i=0;i<D;i++) for (int j=0;j<D;j++) Wy[i]+= W[i*D+j]*y[j];
  float dot=0.f; for (int i=0;i<D;i++) dot += Wy[i]*h[i];
  Eacc += dot;
}

static void grad_y_autodiff(const std::vector<float>& y,
                            const std::vector<float>& W,
                            const std::vector<float>& h,
                            int D, std::vector<float>& g) {
  // Gradient wrt y: (W^T h)  (toy analogue)
  for (int i=0;i<D;i++){ g[i]=0.f; for (int j=0;j<D;j++) g[i]+= W[j*D+i]*h[j]; }
}

static void inner_step(std::vector<float>& y, const std::vector<float>& g, float eta) {
  for (size_t i=0;i<y.size();++i) y[i] -= eta*g[i];
}

int main(int argc, char** argv){
  int D=256, K=4, T=6;
  std::string device="CPU";
  if (const char* s = std::getenv("EBT_K")) K = std::atoi(s);
  if (const char* s = std::getenv("EBT_T")) T = std::atoi(s);
  if (const char* s = std::getenv("EBT_D")) D = std::atoi(s);
  if (const char* s = std::getenv("EBT_DEVICE")) device = s;

  std::vector<float> W(D*D,0.f), h(D,0.01f);
  for (int i=0;i<D;i++) W[i*D+i] = 1.0f + 0.001f*i;
  std::vector<float> y(D,0.5f), g(D,0.f);

  FILE* f = std::fopen("reports/roofline.csv","w");
  if (!f) { std::perror("open roofline.csv"); return 1; }
  std::fprintf(f,"kernel,flops,bytes,intensity,time_ms,device\n");

  Timer tm; double ms;

  // energy
  tm.start();
  float Eacc=0.f;
  for (int k=0;k<K;k++) energy_head_bilinear(y,W,h,D,Eacc);
  ms = tm.stop_ms();
  // naive flop/byte estimates
  double flops = double(K)*D*D*2.0;
  double bytes = double(K)*(D*D + 2*D)*sizeof(float);
  std::fprintf(f,"energy_head_bilinear,%.0f,%.0f,%.3f,%.3f,%s\n", flops, bytes, flops/bytes, ms, device.c_str());

  // grad
  tm.start();
  for (int k=0;k<K;k++) grad_y_autodiff(y,W,h,D,g);
  ms = tm.stop_ms();
  flops = double(K)*D*D;
  bytes = double(K)*(D*D + 2*D)*sizeof(float);
  std::fprintf(f,"grad_y_autodiff,%.0f,%.0f,%.3f,%.3f,%s\n", flops, bytes, flops/bytes, ms, device.c_str());

  // inner steps
  tm.start();
  for (int k=0;k<K;k++) for (int t=0;t<T;t++) inner_step(y,g,0.1f);
  ms = tm.stop_ms();
  flops = double(K)*T*D;
  bytes = double(K)*T*2*D*sizeof(float);
  std::fprintf(f,"inner_step,%.0f,%.0f,%.3f,%.3f,%s\n", flops, bytes, flops/bytes, ms, device.c_str());

  std::fclose(f);
  std::cout << "// wrote reports/roofline.csv\n";
  return 0;
}
