// SPDX-License-Identifier: MIT
// Truthful, optional-library FFT baseline benchmark.
//
// This binary deliberately does not contain a synthetic fallback.  A result is
// emitted only after the named FFT library created a plan, executed the actual
// transform, synchronized its timing domain, and passed a forward/inverse
// round-trip check.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(TESSERA_HAVE_FFTW)
#include <fftw3.h>
#endif

#if defined(TESSERA_HAVE_ROCFFT)
#include <hip/hip_runtime_api.h>
#include <rocfft/rocfft.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct Args {
  int64_t n = 1 << 20;
  int64_t batch = 1;
  int repeats = 50;
  int warmup = 10;
  std::string backend = "all";
};

[[noreturn]] void usage(int code) {
  std::ostream &os = code == 0 ? std::cout : std::cerr;
  os << "spectral_fft_bench [--backend all|fftw|rocfft] [--N <int>] "
        "[--batch <int>] [--repeats <int>] [--warmup <int>]\n";
  std::exit(code);
}

int64_t parse_positive(const char *flag, const char *value) {
  char *end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (!end || *end != '\0' || parsed <= 0)
    throw std::invalid_argument(std::string(flag) + " requires a positive integer");
  return static_cast<int64_t>(parsed);
}

Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag(argv[i]);
    if (flag == "--help" || flag == "-h")
      usage(0);
    if (i + 1 >= argc)
      usage(2);
    const char *value = argv[++i];
    if (flag == "--backend")
      args.backend = value;
    else if (flag == "--N")
      args.n = parse_positive("--N", value);
    else if (flag == "--batch")
      args.batch = parse_positive("--batch", value);
    else if (flag == "--repeats")
      args.repeats = static_cast<int>(parse_positive("--repeats", value));
    else if (flag == "--warmup")
      args.warmup = static_cast<int>(parse_positive("--warmup", value));
    else
      usage(2);
  }
  if (args.backend != "all" && args.backend != "fftw" &&
      args.backend != "rocfft")
    throw std::invalid_argument("--backend must be all, fftw, or rocfft");
  if (args.n > std::numeric_limits<int>::max())
    throw std::invalid_argument("--N exceeds the supported library ABI");
  return args;
}

std::vector<float> make_input(int64_t n, int64_t batch) {
  std::vector<float> values(static_cast<size_t>(2 * n * batch));
  for (int64_t row = 0; row < batch; ++row) {
    for (int64_t i = 0; i < n; ++i) {
      const double phase = 0.013 * static_cast<double>(i) +
                           0.071 * static_cast<double>(row);
      const size_t offset = static_cast<size_t>(2 * (row * n + i));
      values[offset] = static_cast<float>(std::sin(phase) + 0.25 * std::cos(3 * phase));
      values[offset + 1] = static_cast<float>(0.5 * std::cos(phase) -
                                              0.125 * std::sin(5 * phase));
    }
  }
  return values;
}

double max_roundtrip_error(const std::vector<float> &original,
                           const float *roundtrip, int64_t complex_count,
                           int64_t n) {
  double maximum = 0.0;
  const double scale = 1.0 / static_cast<double>(n);
  for (int64_t i = 0; i < 2 * complex_count; ++i)
    maximum = std::max(maximum,
                       std::abs(static_cast<double>(roundtrip[i]) * scale -
                                static_cast<double>(original[static_cast<size_t>(i)])));
  return maximum;
}

void emit_result(const char *backend, const Args &args, double plan_ms,
                 double average_ms, size_t workspace_bytes,
                 double roundtrip_error, const char *timing_domain) {
  const double operations = 5.0 * static_cast<double>(args.batch) *
                            static_cast<double>(args.n) *
                            std::log2(static_cast<double>(args.n));
  const double gflops = operations / (average_ms * 1.0e6);
  std::cout << std::setprecision(10)
            << "{\"schema\":\"tessera.fft_benchmark.v1\","
            << "\"backend\":\"" << backend << "\","
            << "\"dtype\":\"complex64\",\"transform\":\"c2c_forward\","
            << "\"n\":" << args.n << ",\"batch\":" << args.batch << ','
            << "\"warmup\":" << args.warmup << ",\"repeats\":"
            << args.repeats << ",\"plan_ms\":" << plan_ms
            << ",\"average_ms\":" << average_ms << ",\"gflops\":"
            << gflops << ",\"workspace_bytes\":" << workspace_bytes
            << ",\"max_roundtrip_error\":" << roundtrip_error
            << ",\"timing_domain\":\"" << timing_domain << "\"}\n";
}

#if defined(TESSERA_HAVE_FFTW)
void run_fftw(const Args &args) {
  const int n = static_cast<int>(args.n);
  const int batch = static_cast<int>(args.batch);
  const size_t count = static_cast<size_t>(args.n * args.batch);
  auto *input = static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * count));
  auto *output = static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * count));
  auto *roundtrip = static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * count));
  if (!input || !output || !roundtrip)
    throw std::runtime_error("FFTW allocation failed");
  const std::vector<float> source = make_input(args.n, args.batch);
  std::copy(source.begin(), source.end(), reinterpret_cast<float *>(input));

  int dims[] = {n};
  const auto plan_begin = Clock::now();
  fftwf_plan forward = fftwf_plan_many_dft(
      1, dims, batch, input, nullptr, 1, n, output, nullptr, 1, n,
      FFTW_FORWARD, FFTW_ESTIMATE);
  fftwf_plan inverse = fftwf_plan_many_dft(
      1, dims, batch, output, nullptr, 1, n, roundtrip, nullptr, 1, n,
      FFTW_BACKWARD, FFTW_ESTIMATE);
  const double plan_ms = elapsed_ms(plan_begin, Clock::now());
  if (!forward || !inverse)
    throw std::runtime_error("FFTW plan creation failed");

  for (int i = 0; i < args.warmup; ++i)
    fftwf_execute(forward);
  const auto begin = Clock::now();
  for (int i = 0; i < args.repeats; ++i)
    fftwf_execute(forward);
  const double average_ms = elapsed_ms(begin, Clock::now()) / args.repeats;
  fftwf_execute(inverse);
  const double error = max_roundtrip_error(
      source, reinterpret_cast<float *>(roundtrip), args.n * args.batch, args.n);
  emit_result("fftw3f", args, plan_ms, average_ms, 0, error, "host_wall");

  fftwf_destroy_plan(inverse);
  fftwf_destroy_plan(forward);
  fftwf_free(roundtrip);
  fftwf_free(output);
  fftwf_free(input);
}
#endif

#if defined(TESSERA_HAVE_ROCFFT)
void require_hip(hipError_t status, const char *operation) {
  if (status != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

void require_rocfft(rocfft_status status, const char *operation) {
  if (status != rocfft_status_success)
    throw std::runtime_error(std::string(operation) + " failed with rocFFT status " +
                             std::to_string(static_cast<int>(status)));
}

struct RocfftResources {
  rocfft_plan forward = nullptr;
  rocfft_plan inverse = nullptr;
  rocfft_execution_info forward_info = nullptr;
  rocfft_execution_info inverse_info = nullptr;
  void *input = nullptr;
  void *output = nullptr;
  void *roundtrip = nullptr;
  void *workspace = nullptr;

  ~RocfftResources() {
    if (workspace) (void)hipFree(workspace);
    if (roundtrip) (void)hipFree(roundtrip);
    if (output) (void)hipFree(output);
    if (input) (void)hipFree(input);
    if (inverse_info) rocfft_execution_info_destroy(inverse_info);
    if (forward_info) rocfft_execution_info_destroy(forward_info);
    if (inverse) rocfft_plan_destroy(inverse);
    if (forward) rocfft_plan_destroy(forward);
  }
};

struct RocfftContext {
  RocfftContext() { require_rocfft(rocfft_setup(), "rocfft_setup"); }
  ~RocfftContext() { (void)rocfft_cleanup(); }
};

void run_rocfft(const Args &args) {
  RocfftContext context;
  RocfftResources resources;
  const size_t length = static_cast<size_t>(args.n);
  const size_t count = static_cast<size_t>(args.n * args.batch);
  const size_t bytes = 2 * count * sizeof(float);
  const std::vector<float> source = make_input(args.n, args.batch);

  const auto plan_begin = Clock::now();
  require_rocfft(rocfft_plan_create(
                     &resources.forward, rocfft_placement_notinplace,
                     rocfft_transform_type_complex_forward, rocfft_precision_single,
                     1, &length, static_cast<size_t>(args.batch), nullptr),
                 "rocfft_plan_create(forward)");
  require_rocfft(rocfft_plan_create(
                     &resources.inverse, rocfft_placement_notinplace,
                     rocfft_transform_type_complex_inverse, rocfft_precision_single,
                     1, &length, static_cast<size_t>(args.batch), nullptr),
                 "rocfft_plan_create(inverse)");
  require_rocfft(rocfft_execution_info_create(&resources.forward_info),
                 "rocfft_execution_info_create(forward)");
  require_rocfft(rocfft_execution_info_create(&resources.inverse_info),
                 "rocfft_execution_info_create(inverse)");
  size_t forward_workspace = 0, inverse_workspace = 0;
  require_rocfft(rocfft_plan_get_work_buffer_size(resources.forward, &forward_workspace),
                 "rocfft_plan_get_work_buffer_size(forward)");
  require_rocfft(rocfft_plan_get_work_buffer_size(resources.inverse, &inverse_workspace),
                 "rocfft_plan_get_work_buffer_size(inverse)");
  const size_t workspace_bytes = std::max(forward_workspace, inverse_workspace);
  if (workspace_bytes) {
    require_hip(hipMalloc(&resources.workspace, workspace_bytes), "hipMalloc(workspace)");
    require_rocfft(rocfft_execution_info_set_work_buffer(
                       resources.forward_info, resources.workspace, workspace_bytes),
                   "rocfft_execution_info_set_work_buffer(forward)");
    require_rocfft(rocfft_execution_info_set_work_buffer(
                       resources.inverse_info, resources.workspace, workspace_bytes),
                   "rocfft_execution_info_set_work_buffer(inverse)");
  }
  const double plan_ms = elapsed_ms(plan_begin, Clock::now());

  require_hip(hipMalloc(&resources.input, bytes), "hipMalloc(input)");
  require_hip(hipMalloc(&resources.output, bytes), "hipMalloc(output)");
  require_hip(hipMalloc(&resources.roundtrip, bytes), "hipMalloc(roundtrip)");
  require_hip(hipMemcpy(resources.input, source.data(), bytes, hipMemcpyHostToDevice),
              "hipMemcpy(input)");
  void *forward_input[] = {resources.input};
  void *forward_output[] = {resources.output};
  for (int i = 0; i < args.warmup; ++i)
    require_rocfft(rocfft_execute(resources.forward, forward_input, forward_output,
                                  resources.forward_info),
                   "rocfft_execute(warmup)");
  require_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(warmup)");

  const auto begin = Clock::now();
  for (int i = 0; i < args.repeats; ++i)
    require_rocfft(rocfft_execute(resources.forward, forward_input, forward_output,
                                  resources.forward_info),
                   "rocfft_execute(forward)");
  require_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(timed)");
  const double average_ms = elapsed_ms(begin, Clock::now()) / args.repeats;

  void *inverse_input[] = {resources.output};
  void *inverse_output[] = {resources.roundtrip};
  require_rocfft(rocfft_execute(resources.inverse, inverse_input, inverse_output,
                                resources.inverse_info),
                 "rocfft_execute(inverse)");
  require_hip(hipDeviceSynchronize(), "hipDeviceSynchronize(inverse)");
  std::vector<float> roundtrip(2 * count);
  require_hip(hipMemcpy(roundtrip.data(), resources.roundtrip, bytes,
                        hipMemcpyDeviceToHost),
              "hipMemcpy(roundtrip)");
  const double error = max_roundtrip_error(source, roundtrip.data(),
                                           args.n * args.batch, args.n);
  emit_result("rocfft", args, plan_ms, average_ms, workspace_bytes, error,
              "host_wall_synchronized");
}
#endif

} // namespace

int main(int argc, char **argv) {
  try {
    const Args args = parse_args(argc, argv);
    bool ran = false;
    if (args.backend == "all" || args.backend == "fftw") {
#if defined(TESSERA_HAVE_FFTW)
      run_fftw(args);
      ran = true;
#else
      if (args.backend == "fftw")
        throw std::runtime_error("FFTW3f support was not compiled in");
#endif
    }
    if (args.backend == "all" || args.backend == "rocfft") {
#if defined(TESSERA_HAVE_ROCFFT)
      run_rocfft(args);
      ran = true;
#else
      if (args.backend == "rocfft")
        throw std::runtime_error("rocFFT support was not compiled in");
#endif
    }
    if (!ran)
      throw std::runtime_error("no FFT baseline library was compiled in");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "spectral_fft_bench: " << error.what() << '\n';
    return 1;
  }
}
